use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use clap::Parser;
use futures_util::StreamExt;
use rand::{Rng, rng};
use uuid::Uuid;
use vortex::{
    ArraySession, IntoArray, ToCanonical,
    arrays::{BoolArray, FixedSizeListArray, PrimitiveArray, StructArray, VarBinViewArray},
    compute::{Operator, compare},
    dtype::Nullability,
    expr::{col, list_contains, lit, lt, root, select},
    file::{OpenOptionsSessionExt, WriteOptionsSessionExt},
    io::session::RuntimeSession,
    layout::session::LayoutSession,
    metrics::VortexMetrics,
    scalar::Scalar,
    session::VortexSession,
    stream::ArrayStreamExt,
    validity::Validity,
};

#[derive(Debug, Parser)]
struct Opt {
    #[arg(long, short = 'n', default_value_t = 1_000)]
    rows: usize,
    #[arg(long, short = 'd', default_value_t = 8)]
    dimension: usize,
    #[arg(long, short = 'k', default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    include_values: bool,
    #[arg(long)]
    include_metadata: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::parse();

    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<VortexMetrics>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex::file::register_default_encodings(&session);

    let ids = VarBinViewArray::from_iter_str((0..opt.rows).map(|_| Uuid::new_v4().to_string()));

    let vectors = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter(
            (0..opt.rows * opt.dimension).map(|_| rng().random_range(-1.0..1.0)),
        )
        .into_array(),
        opt.dimension as u32,
        Validity::NonNullable,
        opt.rows,
    )?;

    let projections = FixedSizeListArray::try_new(
        BoolArray::from_iter((0..opt.rows * opt.dimension).map(|_| rng().random_bool(0.5)))
            .into_array(),
        opt.dimension as u32,
        Validity::NonNullable,
        opt.rows,
    )?;

    let rand_floats =
        PrimitiveArray::from_iter((0..opt.rows).map(|_| rng().random_range(0.0..1.0)));

    let records = StructArray::from_fields(&[
        ("id", ids.into_array()),
        ("vector", vectors.into_array()),
        ("projection", projections.into_array()),
        ("rand_float", rand_floats.into_array()),
    ])?;

    let mut file = tokio::fs::File::create("test.vortex").await?;

    session
        .write_options()
        .write(&mut file, records.to_array_stream())
        .await?;

    let file = session.open_options().open("test.vortex").await?;

    let stream = file
        .scan()?
        .with_filter(lt(col("rand_float"), lit(0.1)))
        .with_projection(select(["id", "projection"], root()))
        .into_array_stream()?;

    let mut stream = Box::pin(stream);

    let query_projection = BoolArray::from_iter((0..opt.dimension).map(|_| rng().random_bool(0.5)));

    let mut heap = BinaryHeap::<HeapElement>::new();

    while let Some(array) = stream.next().await {
        let array = array?;

        let s = array.to_struct();
        let ids = s.field_by_name("id")?.to_varbinview();
        let projections = s.field_by_name("projection")?.to_fixed_size_list();

        let len = ids.len();

        for i in 0..len {
            let id = ids.scalar_at(i);
            let id = id.as_utf8().value().unwrap();
            let id = id.as_str();

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

            if heap.len() < opt.top_k {
                heap.push(HeapElement {
                    id: id.to_string(),
                    distance,
                });
            } else {
                if let Some(min) = heap.peek() {
                    if distance < min.distance {
                        heap.pop();
                        heap.push(HeapElement {
                            id: id.to_string(),
                            distance,
                        });
                    }
                }
            }
        }
    }

    let top_k = heap
        .into_sorted_vec()
        .into_iter()
        .map(|h| (h.id.clone(), h.distance))
        .collect::<HashMap<_, _>>();

    let mut projection_mask = vec!["id"];
    if opt.include_values {
        projection_mask.push("vector");
    }
    if opt.include_metadata {
        projection_mask.push("rand_float");
    }

    let filter = if top_k.is_empty() {
        lit(false)
    } else {
        let scalars = top_k
            .keys()
            .map(|id| Scalar::from(id.as_str()))
            .collect::<Vec<_>>();
        let list = Scalar::list(
            scalars.first().unwrap().dtype().clone(),
            scalars,
            Nullability::NonNullable,
        );
        list_contains(lit(list), col("id"))
    };

    let results = file
        .scan()?
        .with_filter(filter)
        .with_projection(select(projection_mask.as_slice(), root()))
        .into_array_stream()?
        .read_all()
        .await?;

    let s = results.to_struct();
    let ids = s.field_by_name("id")?.to_varbinview();

    for i in 0..s.len() {
        let mut result_string = String::new();

        let id_scalar = ids.scalar_at(i);
        let id_utf8_value = id_scalar.as_utf8().value().unwrap();
        let id = id_utf8_value.as_str();
        result_string.push_str(&format!("id={}", id));

        let distance = top_k.get(id).unwrap();
        result_string.push_str(&format!(" distance={}", distance));

        if opt.include_values {
            let vectors = s.field_by_name("vector")?.to_listview();
            let vector = vectors.list_elements_at(i).to_primitive();
            let vector = vector.display_values();
            result_string.push_str(&format!(" values={}", vector));
        }

        if opt.include_metadata {
            let rand_floats = s.field_by_name("rand_float")?.to_primitive();
            let rand_float: f64 = rand_floats
                .scalar_at(i)
                .as_primitive()
                .typed_value()
                .unwrap();
            result_string.push_str(&format!(" metadata={:?}", rand_float));
        }

        println!("{}", result_string);
    }

    Ok(())
}

#[derive(Debug, PartialEq, Eq, Ord)]
struct HeapElement {
    id: String,
    distance: usize,
}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.distance.cmp(&other.distance))
    }
}
