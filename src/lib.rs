//! A proof-of-concept vector database built on Vortex.
//!
//! This crate provides functionality to write and read vector data using
//! IVF partitioning and binary projections for approximate nearest neighbor search.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    fmt::Display,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use anyhow::Result;
use futures_util::{Stream, StreamExt, stream};
use rand::Rng;
use tokio::sync::Mutex;
use uuid::Uuid;
use vortex::{
    array::{
        ArrayRef, IntoArray, ToCanonical,
        arrays::{BoolArray, FixedSizeListArray, PrimitiveArray, StructArray, VarBinViewArray},
        session::ArraySession,
        stream::{ArrayStream, ArrayStreamExt},
        validity::Validity,
    },
    buffer::Buffer,
    compute::{Operator, compare},
    dtype::{DType, Nullability, StructFields},
    encodings::sequence::SequenceArray,
    error::VortexResult,
    expr::{and_collect, col, eq, lit, lt, or_collect, root, select, session::ExprSession},
    file::{OpenOptionsSessionExt, WriteOptionsSessionExt},
    io::session::RuntimeSession,
    layout::session::LayoutSession,
    metrics::{Metric, VortexMetrics},
    scan::Selection,
    session::VortexSession,
};

// Column name constants
pub const ROW_IDX_COL: &str = "row_idx";
pub const ID_COL: &str = "id";
pub const VECTOR_COL: &str = "vector";
pub const PROJECTION_COL: &str = "projection";
pub const IVF_PARTITION_IDX_COL: &str = "ivf_partition_idx";
pub const RAND_FLOAT_COL: &str = "rand_float";
pub const RAND_CATEGORICAL_1_COL: &str = "rand_categorical_1";
pub const RAND_CATEGORICAL_2_COL: &str = "rand_categorical_2";

/// Shared database configuration for vector storage parameters.
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub rows: usize,
    pub dimension: usize,
    /// Projection bits as a ratio of dimension (e.g., 1.0 means dimension bits).
    pub projection_bits: f64,
    pub rand_categorical_cardinality: u32,
}

/// Configuration for writing data to a Vortex file.
#[derive(Debug, Clone)]
pub struct WriteConfig {
    pub path: PathBuf,
    pub db: DatabaseConfig,
    pub chunk_size: usize,
    pub progress: bool,
}

/// Configuration for reading and querying data from a Vortex file.
#[derive(Debug, Clone)]
pub struct ReadConfig {
    pub path: PathBuf,
    pub db: DatabaseConfig,
    pub top_k: usize,
    pub queries: usize,
    pub tombstones: f64,
    pub include_values: bool,
    pub include_metadata: bool,
    pub progress: bool,
    pub rand_float_selectivity: f64,
    pub print_results: bool,
}

/// Creates a new Vortex session with all required components.
pub fn create_session() -> VortexSession {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<VortexMetrics>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>();

    vortex::file::register_default_encodings(&session);
    session
}

/// Writes synthetic vector data to a Vortex file.
pub async fn write_command(config: WriteConfig) -> Result<()> {
    let WriteConfig {
        path,
        db:
            DatabaseConfig {
                rows,
                dimension,
                projection_bits,
                rand_categorical_cardinality,
            },
        chunk_size,
        progress,
    } = config;

    let session = create_session();

    let write_stage_start = Instant::now();

    let projection_bits = (projection_bits * dimension as f64) as usize;

    let ivf_partitions = rows.isqrt();
    let ivf_partition_size = rows.div_ceil(ivf_partitions);

    let pbar = progress.then(|| Arc::new(Mutex::new(tqdm::pbar(Some(rows)))));

    struct StreamState {
        rows_written: usize,
        pbar: Option<Arc<Mutex<tqdm::Tqdm<()>>>>,
    }

    let chunk_stream = stream::try_unfold(
        StreamState {
            rows_written: 0,
            pbar,
        },
        move |state| async move {
            let StreamState {
                mut rows_written,
                pbar,
            } = state;

            if rows_written >= rows {
                return Ok(None);
            }

            let chunk_size = chunk_size.min(rows - rows_written);

            // Scope the RNG to ensure it's dropped before any await points
            let struct_array = {
                let mut rng = rand::rng();

                let row_idxs = SequenceArray::typed_new(
                    rows_written as u64,
                    1,
                    Nullability::NonNullable,
                    chunk_size,
                )?
                .into_array();

                let ids = VarBinViewArray::from_iter_str(
                    (0..chunk_size).map(|_| Uuid::new_v4().to_string()),
                );

                let vectors = FixedSizeListArray::try_new(
                    PrimitiveArray::from_iter(
                        (0..chunk_size * dimension).map(|_| rng.random_range(-1.0f32..1.0)),
                    )
                    .into_array(),
                    dimension as u32,
                    Validity::NonNullable,
                    chunk_size,
                )?;

                let projections = FixedSizeListArray::try_new(
                    BoolArray::from_iter(
                        (0..chunk_size * projection_bits).map(|_| rng.random_bool(0.5)),
                    )
                    .into_array(),
                    projection_bits as u32,
                    Validity::NonNullable,
                    chunk_size,
                )?;

                let ivf_partition_idxs = PrimitiveArray::from_iter(
                    (0..chunk_size).map(|i| ((rows_written + i) / ivf_partition_size) as u32),
                );

                let rand_floats = PrimitiveArray::from_iter(
                    (0..chunk_size).map(|_| rng.random_range(0.0f64..1.0)),
                );

                let rand_categorical_1 = PrimitiveArray::from_iter(
                    (0..chunk_size).map(|_| rng.random_range(0..rand_categorical_cardinality)),
                );

                let rand_categorical_2 = PrimitiveArray::from_iter(
                    (0..chunk_size).map(|_| rng.random_range(0..rand_categorical_cardinality)),
                );

                StructArray::from_fields(&[
                    (ROW_IDX_COL, row_idxs.into_array()),
                    (ID_COL, ids.into_array()),
                    (VECTOR_COL, vectors.into_array()),
                    (PROJECTION_COL, projections.into_array()),
                    (IVF_PARTITION_IDX_COL, ivf_partition_idxs.into_array()),
                    (RAND_FLOAT_COL, rand_floats.into_array()),
                    (RAND_CATEGORICAL_1_COL, rand_categorical_1.into_array()),
                    (RAND_CATEGORICAL_2_COL, rand_categorical_2.into_array()),
                ])?
            };

            rows_written += chunk_size;

            if let Some(pbar) = &pbar {
                let _ = pbar.lock().await.update(chunk_size);
            }

            Ok(Some((
                struct_array.into_array(),
                StreamState { rows_written, pbar },
            )))
        },
    )
    .boxed();

    let dtype = DType::Struct(
        StructFields::new(
            [
                ROW_IDX_COL,
                ID_COL,
                VECTOR_COL,
                PROJECTION_COL,
                IVF_PARTITION_IDX_COL,
                RAND_FLOAT_COL,
                RAND_CATEGORICAL_1_COL,
                RAND_CATEGORICAL_2_COL,
            ]
            .into(),
            vec![
                DType::Primitive(vortex::dtype::PType::U64, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
                DType::FixedSizeList(
                    Arc::new(DType::Primitive(
                        vortex::dtype::PType::F32,
                        Nullability::NonNullable,
                    )),
                    dimension as u32,
                    Nullability::NonNullable,
                ),
                DType::FixedSizeList(
                    Arc::new(DType::Bool(Nullability::NonNullable)),
                    projection_bits as u32,
                    Nullability::NonNullable,
                ),
                DType::Primitive(vortex::dtype::PType::U32, Nullability::NonNullable),
                DType::Primitive(vortex::dtype::PType::F64, Nullability::NonNullable),
                DType::Primitive(vortex::dtype::PType::U32, Nullability::NonNullable),
                DType::Primitive(vortex::dtype::PType::U32, Nullability::NonNullable),
            ],
        ),
        Nullability::NonNullable,
    );

    let array_stream = StreamArrayStream {
        inner: chunk_stream,
        dtype,
    };

    let mut file = tokio::fs::File::create(&path).await?;

    let write_summary = session
        .write_options()
        .write(&mut file, array_stream)
        .await?;

    let expected_uncompressed_size = rows
        * (
            size_of::<u64>() // row_idx
            + Uuid::new_v4().to_string().len() // id
            + size_of::<f32>() * dimension // vector
            + projection_bits.div_ceil(u8::BITS as usize) // projection
            + size_of::<u32>() // ivf_partition_idx
            + size_of::<f64>() // rand_float
            + 2 * size_of::<u32>()
            // rand_categorical_1, rand_categorical_2
        );
    let actual_compressed_size = write_summary.size();
    let ratio = expected_uncompressed_size as f64 / actual_compressed_size as f64;
    println!(
        "expected size: {:.2} MB, actual size: {:.2} MB, ratio: {:.2}",
        expected_uncompressed_size as f64 / (1 << 20) as f64,
        actual_compressed_size as f64 / (1 << 20) as f64,
        ratio
    );

    println!(
        "write stage elapsed time: {:?}",
        write_stage_start.elapsed()
    );

    Ok(())
}

/// Reads and queries vector data from a Vortex file.
pub async fn read_command(config: ReadConfig) -> Result<()> {
    let ReadConfig {
        path,
        db:
            DatabaseConfig {
                rows,
                dimension,
                projection_bits,
                rand_categorical_cardinality,
            },
        top_k,
        queries,
        tombstones,
        include_values,
        include_metadata,
        progress,
        rand_float_selectivity,
        print_results,
    } = config;

    let session = create_session();

    let read_stage_start = Instant::now();

    let projection_bits = (projection_bits * dimension as f64) as usize;

    let ivf_partitions = rows.isqrt();
    // nprobe = sqrt(partitions) = sqrt(sqrt(rows)), balancing recall vs. performance
    let nprobe = ivf_partitions.isqrt();

    let file = session.open_options().open(path).await?;

    let mut rng = rand::rng();

    let mut tombstone_idxs =
        rand::seq::index::sample(&mut rng, rows, (rows as f64 * tombstones) as usize)
            .into_iter()
            .map(|idx| idx as u64)
            .collect::<Vec<_>>();
    tombstone_idxs.sort();
    let tombstone_idxs = Buffer::from_iter(tombstone_idxs);

    let metrics = file.metrics();

    let pbar = progress.then(|| Arc::new(Mutex::new(tqdm::pbar(Some(queries)))));

    for _ in 0..queries {
        let query_start = Instant::now();

        let query_projection =
            BoolArray::from_iter((0..projection_bits).map(|_| rng.random_bool(0.5)));
        let query_ivf_partition_idxs: Vec<_> = (0..nprobe)
            .map(|_| rng.random_range(0..ivf_partitions))
            .collect();
        let query_rand_categorical_1 = rng.random_range(0..rand_categorical_cardinality);
        let query_rand_categorical_2 = rng.random_range(0..rand_categorical_cardinality);

        let stream = file
            .scan()?
            .with_selection(Selection::ExcludeByIndex(tombstone_idxs.clone()))
            .with_filter(
                and_collect(vec![
                    or_collect(
                        query_ivf_partition_idxs
                            .into_iter()
                            .map(|idx| eq(col(IVF_PARTITION_IDX_COL), lit(idx as u32))),
                    )
                    .ok_or_else(|| anyhow::anyhow!("empty IVF partition filter"))?,
                    eq(col(RAND_CATEGORICAL_1_COL), lit(query_rand_categorical_1)),
                    eq(col(RAND_CATEGORICAL_2_COL), lit(query_rand_categorical_2)),
                    lt(col(RAND_FLOAT_COL), lit(rand_float_selectivity)),
                ])
                .ok_or_else(|| anyhow::anyhow!("empty filter"))?,
            )
            .with_projection(select([ROW_IDX_COL, ID_COL, PROJECTION_COL], root()))
            .into_array_stream()?;

        let mut stream = Box::pin(stream);

        let mut heap = BinaryHeap::<HeapElement>::new();

        while let Some(array) = stream.next().await {
            let array = array?;

            let s = array.to_struct();
            let row_idxs = s.field_by_name(ROW_IDX_COL)?.to_primitive();
            let ids = s.field_by_name(ID_COL)?.to_varbinview();
            let projections = s.field_by_name(PROJECTION_COL)?.to_fixed_size_list();

            for i in 0..s.len() {
                let projection_array = projections.fixed_size_list_elements_at(i);
                let projection = projection_array.to_bool();

                let distance = compare(
                    query_projection.as_ref(),
                    projection.as_ref(),
                    Operator::NotEq,
                )?
                .to_bool()
                .as_bool_typed()
                .true_count()?;

                let should_insert =
                    heap.len() < top_k || heap.peek().is_some_and(|max| distance < max.distance);

                if should_insert {
                    let row_idx: u64 = row_idxs
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .expect("row_idx should be a valid u64 value");

                    let id = ids
                        .scalar_at(i)
                        .as_utf8()
                        .value()
                        .expect("id should be a valid UTF8 value")
                        .as_str()
                        .to_string();

                    if heap.len() >= top_k {
                        heap.pop();
                    }
                    heap.push(HeapElement {
                        row_idx,
                        id,
                        distance,
                    });
                }
            }
        }

        let top_k_results = heap.into_sorted_vec();

        let id_to_distance = top_k_results
            .iter()
            .map(|h| (h.id.clone(), h.distance))
            .collect::<HashMap<_, _>>();

        let mut row_idxs = top_k_results.iter().map(|h| h.row_idx).collect::<Vec<_>>();
        row_idxs.sort();
        let selection = Selection::IncludeByIndex(Buffer::from_iter(row_idxs));

        let mut projection_mask = vec![ID_COL];
        if include_values {
            projection_mask.push(VECTOR_COL);
        }
        if include_metadata {
            projection_mask.push(RAND_FLOAT_COL);
            projection_mask.push(RAND_CATEGORICAL_1_COL);
            projection_mask.push(RAND_CATEGORICAL_2_COL);
        }

        let results = file
            .scan()?
            .with_selection(selection)
            .with_projection(select(projection_mask.as_slice(), root()))
            .into_array_stream()?
            .read_all()
            .await?;

        let s = results.to_struct();
        let ids = s.field_by_name(ID_COL)?.to_varbinview();
        let vectors = s.field_by_name(VECTOR_COL);
        let rand_floats = s.field_by_name(RAND_FLOAT_COL);
        let rand_categorical_1 = s.field_by_name(RAND_CATEGORICAL_1_COL);
        let rand_categorical_2 = s.field_by_name(RAND_CATEGORICAL_2_COL);

        let mut results = (0..s.len())
            .map(|i| {
                let id_scalar = ids.scalar_at(i);
                let id_utf8_value = id_scalar
                    .as_utf8()
                    .value()
                    .expect("id should be a valid UTF8 value");
                let id = id_utf8_value.as_str().to_string();

                let distance = *id_to_distance
                    .get(&id)
                    .expect("id should exist in distance map");

                let vector = include_values.then(|| {
                    let vectors = vectors
                        .as_ref()
                        .expect("vectors field should be present when include_values is true")
                        .to_fixed_size_list();
                    vectors.fixed_size_list_elements_at(i).to_primitive()
                });

                let metadata = include_metadata.then(|| {
                    let rand_floats = rand_floats
                        .as_ref()
                        .expect("rand_floats field should be present when include_metadata is true")
                        .to_primitive();
                    let rand_float = rand_floats
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .expect("rand_float should be a valid f64 value");
                    let rand_categorical_1 = rand_categorical_1
                        .as_ref()
                        .expect(
                            "rand_categorical_1 field should be present when include_metadata is true",
                        )
                        .to_primitive();
                    let rand_categorical_1 = rand_categorical_1
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .expect("rand_categorical_1 should be a valid u32 value");
                    let rand_categorical_2 = rand_categorical_2
                        .as_ref()
                        .expect(
                            "rand_categorical_2 field should be present when include_metadata is true",
                        )
                        .to_primitive();
                    let rand_categorical_2 = rand_categorical_2
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .expect("rand_categorical_2 should be a valid u32 value");
                    (rand_float, rand_categorical_1, rand_categorical_2)
                });

                ResultElement {
                    id,
                    distance,
                    vector,
                    metadata,
                }
            })
            .collect::<Vec<_>>();

        results.sort_by_key(|r| r.distance);

        if print_results {
            for result in results {
                println!("{}", result);
            }
        }

        metrics
            .histogram("query.duration")
            .update(query_start.elapsed().as_nanos() as i64);

        if let Some(pbar) = &pbar {
            let _ = pbar.lock().await.update(1);
        }
    }

    if let Some(pbar) = &pbar {
        let _ = pbar.lock().await.close();
    }

    println!("read stage elapsed time: {:?}", read_stage_start.elapsed());

    let snapshot = metrics.snapshot();

    for (id, metric) in snapshot.iter() {
        let name = id.name();

        match metric {
            Metric::Counter(counter) => {
                let value = counter.count();
                println!("counter {name}: {value}");
            }
            Metric::Histogram(hist) => {
                let snapshot = hist.snapshot();
                let p50 = snapshot.value(0.5);
                let p90 = snapshot.value(0.9);
                let p99 = snapshot.value(0.99);
                println!(
                    "histogram {name}: p50={:.2}ns, p90={:.2}ns, p99={:.2}ns",
                    p50, p90, p99
                );
            }
            Metric::Timer(timer) => {
                let snapshot = timer.snapshot();
                let p50 = snapshot.value(0.5);
                let p90 = snapshot.value(0.9);
                let p99 = snapshot.value(0.99);
                println!(
                    "timer {name}: p50={:.2}ns, p90={:.2}ns, p99={:.2}ns",
                    p50, p90, p99
                );
            }
            _ => {}
        }
    }

    Ok(())
}

pub struct StreamArrayStream {
    pub dtype: DType,
    pub inner: Pin<Box<dyn Stream<Item = VortexResult<ArrayRef>> + Send>>,
}

impl Stream for StreamArrayStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl ArrayStream for StreamArrayStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

#[derive(Debug, PartialEq, Eq)]
struct HeapElement {
    row_idx: u64,
    id: String,
    distance: usize,
}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

struct ResultElement {
    id: String,
    distance: usize,
    vector: Option<PrimitiveArray>,
    metadata: Option<(f64, u32, u32)>,
}

impl Display for ResultElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "id={} distance={}", self.id, self.distance)?;
        if let Some(vector) = self.vector.as_ref() {
            write!(f, " values={}", vector.display_values())?;
        }
        if let Some(metadata) = self.metadata {
            write!(
                f,
                " metadata=({}, {}, {})",
                metadata.0, metadata.1, metadata.2
            )?;
        }
        Ok(())
    }
}
