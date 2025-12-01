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

use clap::Parser;
use futures_util::{Stream, StreamExt, stream};
use rand::{Rng, rng};
use tokio::sync::Mutex;
use uuid::Uuid;
use vortex::{
    ArrayRef, ArraySession, IntoArray, ToCanonical,
    arrays::{BoolArray, FixedSizeListArray, PrimitiveArray, StructArray, VarBinViewArray},
    buffer::Buffer,
    compute::{Operator, compare},
    dtype::{DType, Nullability, StructFields},
    encodings::sequence::SequenceArray,
    error::VortexResult,
    expr::{and_collect, col, eq, lit, lt, or_collect, root, select},
    file::{OpenOptionsSessionExt, WriteOptionsSessionExt},
    io::session::RuntimeSession,
    layout::session::LayoutSession,
    metrics::{Metric, VortexMetrics},
    scan::Selection,
    session::VortexSession,
    stream::{ArrayStream, ArrayStreamExt},
    validity::Validity,
};

const ROW_IDX_COL: &str = "row_idx";
const ID_COL: &str = "id";
const VECTOR_COL: &str = "vector";
const PROJECTION_COL: &str = "projection";
const IVF_PARTITION_IDX_COL: &str = "ivf_partition_idx";
const RAND_FLOAT_COL: &str = "rand_float";
const RAND_CATEGORICAL_1_COL: &str = "rand_categorical_1";
const RAND_CATEGORICAL_2_COL: &str = "rand_categorical_2";

#[derive(Debug, Parser)]
struct Opt {
    #[arg(long, short = 'f', default_value = "db.vortex")]
    path: PathBuf,
    #[arg(long, short = 'n', default_value_t = 1024)]
    rows: usize,
    #[arg(long, short = 'd', default_value_t = 1024)]
    dimension: usize,
    #[arg(long, short = 'c', default_value_t = 1024)]
    chunk_size: usize,
    #[arg(long, short = 'k', default_value_t = 10)]
    top_k: usize,
    #[arg(long, short = 'q', default_value_t = 100)]
    queries: usize,
    #[arg(long, short = 't', default_value_t = 0.01)]
    tombstones: f64,
    #[arg(long)]
    include_values: bool,
    #[arg(long)]
    include_metadata: bool,
    #[arg(long)]
    progress: bool,
    #[arg(long, default_value_t = 5)]
    rand_categorical_cardinality: u32,
    #[arg(long, default_value_t = 0.5)]
    rand_float_selectivity: f64,
    #[arg(long)]
    print_results: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::parse();

    let Opt {
        path,
        rows,
        chunk_size,
        dimension,
        top_k,
        queries,
        tombstones,
        include_values,
        include_metadata,
        progress,
        rand_categorical_cardinality,
        rand_float_selectivity,
        print_results,
    } = opt;

    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<VortexMetrics>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex::file::register_default_encodings(&session);

    let write_stage_start = Instant::now();

    let ivf_partitions = rows.isqrt();
    let ivf_partition_size = (rows + ivf_partitions - 1) / ivf_partitions;

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

            let row_idxs = SequenceArray::typed_new(
                rows_written as u64,
                1,
                Nullability::NonNullable,
                chunk_size,
            )?
            .into_array();

            let ids =
                VarBinViewArray::from_iter_str((0..chunk_size).map(|_| Uuid::new_v4().to_string()));

            let vectors = FixedSizeListArray::try_new(
                PrimitiveArray::from_iter(
                    (0..chunk_size * dimension).map(|_| rng().random_range(-1.0f32..1.0)),
                )
                .into_array(),
                dimension as u32,
                Validity::NonNullable,
                chunk_size,
            )?;

            let projections = FixedSizeListArray::try_new(
                BoolArray::from_iter((0..chunk_size * dimension).map(|_| rng().random_bool(0.5)))
                    .into_array(),
                dimension as u32,
                Validity::NonNullable,
                chunk_size,
            )?;

            let ivf_partition_idxs = PrimitiveArray::from_iter(
                (0..chunk_size).map(|i| ((rows_written + i) / ivf_partition_size) as u32),
            );

            let rand_floats =
                PrimitiveArray::from_iter((0..chunk_size).map(|_| rng().random_range(0.0f64..1.0)));

            let rand_categorical_1 = PrimitiveArray::from_iter(
                (0..chunk_size).map(|_| rng().random_range(0..rand_categorical_cardinality)),
            );

            let rand_categorical_2 = PrimitiveArray::from_iter(
                (0..chunk_size).map(|_| rng().random_range(0..rand_categorical_cardinality)),
            );

            let struct_array = StructArray::from_fields(&[
                (ROW_IDX_COL, row_idxs.into_array()),
                (ID_COL, ids.into_array()),
                (VECTOR_COL, vectors.into_array()),
                (PROJECTION_COL, projections.into_array()),
                (IVF_PARTITION_IDX_COL, ivf_partition_idxs.into_array()),
                (RAND_FLOAT_COL, rand_floats.into_array()),
                (RAND_CATEGORICAL_1_COL, rand_categorical_1.into_array()),
                (RAND_CATEGORICAL_2_COL, rand_categorical_2.into_array()),
            ])?;

            rows_written += chunk_size;

            if let Some(pbar) = &pbar {
                _ = pbar.lock().await.update(chunk_size);
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
                    dimension as u32,
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
            + dimension / 8 // projection
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

    let read_stage_start = Instant::now();

    let file = session.open_options().open(path).await.unwrap();

    let mut tombstone_idxs =
        rand::seq::index::sample(&mut rand::rng(), rows, (rows as f64 * tombstones) as usize)
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
            BoolArray::from_iter((0..opt.dimension).map(|_| rng().random_bool(0.5)));
        let nprobe = ivf_partitions.isqrt();
        let query_ivf_partition_idxs = (0..nprobe).map(|_| rng().random_range(0..ivf_partitions));
        let query_rand_categorical_1 = rng().random_range(0..rand_categorical_cardinality);
        let query_rand_categorical_2 = rng().random_range(0..rand_categorical_cardinality);

        let stream = file
            .scan()?
            .with_selection(Selection::ExcludeByIndex(tombstone_idxs.clone()))
            .with_filter(
                and_collect(vec![
                    or_collect(
                        query_ivf_partition_idxs
                            .map(|idx| eq(col(IVF_PARTITION_IDX_COL), lit(idx as u32))),
                    )
                    .unwrap(),
                    eq(col(RAND_CATEGORICAL_1_COL), lit(query_rand_categorical_1)),
                    eq(col(RAND_CATEGORICAL_2_COL), lit(query_rand_categorical_2)),
                    lt(col(RAND_FLOAT_COL), lit(rand_float_selectivity)),
                ])
                .unwrap(),
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

                if heap.len() < top_k {
                    let row_idx = row_idxs.scalar_at(i);
                    let row_idx = row_idx.as_primitive().typed_value().unwrap();

                    let id = ids.scalar_at(i);
                    let id = id.as_utf8().value().unwrap();
                    let id = id.as_str();

                    heap.push(HeapElement {
                        row_idx,
                        id: id.to_string(),
                        distance,
                    });
                } else if let Some(min) = heap.peek()
                    && distance < min.distance
                {
                    let row_idx = row_idxs.scalar_at(i);
                    let row_idx = row_idx.as_primitive().typed_value().unwrap();

                    let id = ids.scalar_at(i);
                    let id = id.as_utf8().value().unwrap();
                    let id = id.as_str();

                    heap.pop();
                    heap.push(HeapElement {
                        row_idx,
                        id: id.to_string(),
                        distance,
                    });
                }
            }
        }

        let top_k = heap.into_sorted_vec();

        let id_to_distance = top_k
            .iter()
            .map(|h| (h.id.clone(), h.distance))
            .collect::<HashMap<_, _>>();

        let mut row_idxs = top_k.iter().map(|h| h.row_idx).collect::<Vec<_>>();
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
                let id_utf8_value = id_scalar.as_utf8().value().unwrap();
                let id = id_utf8_value.as_str().to_string();

                let distance = *id_to_distance.get(&id).unwrap();

                let vector = include_values.then(|| {
                    let vectors = vectors.as_ref().unwrap().to_fixed_size_list();
                    vectors.fixed_size_list_elements_at(i).to_primitive()
                });

                let metadata = include_metadata.then(|| {
                    let rand_floats = rand_floats.as_ref().unwrap().to_primitive();
                    let rand_float = rand_floats
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .unwrap();
                    let rand_categorical_1 = rand_categorical_1.as_ref().unwrap().to_primitive();
                    let rand_categorical_1 = rand_categorical_1
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .unwrap();
                    let rand_categorical_2 = rand_categorical_2.as_ref().unwrap().to_primitive();
                    let rand_categorical_2 = rand_categorical_2
                        .scalar_at(i)
                        .as_primitive()
                        .typed_value()
                        .unwrap();
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
            _ = pbar.lock().await.update(1);
        }
    }

    if let Some(pbar) = &pbar {
        _ = pbar.lock().await.close();
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
                println!("histogram {name}: p50={p50}, p90={p90}, p99={p99}");
            }
            Metric::Timer(timer) => {
                let snapshot = timer.snapshot();
                let p50 = snapshot.value(0.5);
                let p90 = snapshot.value(0.9);
                let p99 = snapshot.value(0.99);
                println!("timer {name}: p50={p50}, p90={p90}, p99={p99}");
            }
            _ => {}
        }
    }

    Ok(())
}

struct StreamArrayStream {
    dtype: DType,
    inner: Pin<Box<dyn Stream<Item = VortexResult<ArrayRef>> + Send>>,
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

#[derive(Debug, PartialEq, Eq, Ord)]
struct HeapElement {
    row_idx: u64,
    id: String,
    distance: usize,
}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.distance.cmp(&other.distance))
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
