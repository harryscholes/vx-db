use std::{
    path::Path,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures_util::{Stream, StreamExt, stream};
use rand::{Rng, rng};
use tokio::sync::Mutex;
use uuid::Uuid;
use vortex::{
    array::{
        ArrayRef, IntoArray,
        arrays::{BoolArray, FixedSizeListArray, PrimitiveArray, StructArray, VarBinViewArray},
        session::ArraySession,
        stream::ArrayStream,
        validity::Validity,
    },
    dtype::{DType, Nullability, StructFields},
    encodings::sequence::SequenceArray,
    error::VortexResult,
    expr::session::ExprSession,
    file::{OpenOptionsSessionExt, WriteOptionsSessionExt},
    io::session::RuntimeSession,
    layout::session::LayoutSession,
    metrics::VortexMetrics,
    session::VortexSession,
};

pub const ROW_IDX_COL: &str = "row_idx";
pub const ID_COL: &str = "id";
pub const VECTOR_COL: &str = "vector";
pub const PROJECTION_COL: &str = "projection";
pub const IVF_PARTITION_IDX_COL: &str = "ivf_partition_idx";
pub const RAND_FLOAT_COL: &str = "rand_float";
pub const RAND_CATEGORICAL_1_COL: &str = "rand_categorical_1";
pub const RAND_CATEGORICAL_2_COL: &str = "rand_categorical_2";

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

#[derive(Debug, Clone)]
pub struct WriteConfig {
    pub rows: usize,
    pub dimension: usize,
    pub projection_bits: f64,
    pub chunk_size: usize,
    pub rand_categorical_cardinality: u32,
    pub progress: bool,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            rows: 1024,
            dimension: 1024,
            projection_bits: 1.0,
            chunk_size: 1024,
            rand_categorical_cardinality: 5,
            progress: false,
        }
    }
}

pub async fn write_data(
    path: impl AsRef<Path>,
    config: WriteConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let WriteConfig {
        rows,
        dimension,
        projection_bits,
        chunk_size,
        rand_categorical_cardinality,
        progress,
    } = config;

    let session = create_session();

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
                BoolArray::from_iter(
                    (0..chunk_size * projection_bits).map(|_| rng().random_bool(0.5)),
                )
                .into_array(),
                projection_bits as u32,
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

    let mut file = tokio::fs::File::create(path).await?;

    session
        .write_options()
        .write(&mut file, array_stream)
        .await?;

    Ok(())
}

pub async fn open_file(
    session: &VortexSession,
    path: std::path::PathBuf,
) -> Result<vortex::file::VortexFile, Box<dyn std::error::Error>> {
    let file = session.open_options().open(path).await?;
    Ok(file)
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
