use std::process::Command;
use tempfile::tempdir;
use vortex::array::{Array, ToCanonical, stream::ArrayStreamExt};
use vortex::dtype::DType;
use vx_db::{
    ID_COL, IVF_PARTITION_IDX_COL, PROJECTION_COL, RAND_CATEGORICAL_1_COL, RAND_CATEGORICAL_2_COL,
    RAND_FLOAT_COL, ROW_IDX_COL, VECTOR_COL, WriteConfig, create_session, open_file, write_data,
};

#[test]
fn test_write_then_read_cli() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test.vortex");
    let db_path_str = db_path.to_str().unwrap();

    // Write a small dataset
    let write_output = Command::new(env!("CARGO_BIN_EXE_vx-db"))
        .args([
            "write",
            "-f",
            db_path_str,
            "-n",
            "100", // Small number of rows for fast test
            "-d",
            "64", // Small dimension for fast test
            "-c",
            "50", // Small chunk size
        ])
        .output()
        .expect("Failed to execute write command");

    assert!(
        write_output.status.success(),
        "Write command failed with stderr: {}",
        String::from_utf8_lossy(&write_output.stderr)
    );

    // Verify the file was created
    assert!(db_path.exists(), "Database file was not created");

    // Read and query the data
    let read_output = Command::new(env!("CARGO_BIN_EXE_vx-db"))
        .args([
            "read",
            "-f",
            db_path_str,
            "-n",
            "100", // Same row count as write
            "-d",
            "64", // Same dimension as write
            "-k",
            "5", // Top-k results
            "-q",
            "10", // Number of queries
            "-t",
            "0.0", // No tombstones for simplicity
        ])
        .output()
        .expect("Failed to execute read command");

    assert!(
        read_output.status.success(),
        "Read command failed with stderr: {}",
        String::from_utf8_lossy(&read_output.stderr)
    );

    // Verify output contains expected metrics
    let stdout = String::from_utf8_lossy(&read_output.stdout);
    assert!(
        stdout.contains("read stage elapsed time"),
        "Expected metrics output not found in: {}",
        stdout
    );
}

#[tokio::test]
async fn test_write_read_invariants() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test.vortex");

    let rows = 100;
    let dimension = 64;
    let projection_bits = 1.0;
    let rand_categorical_cardinality = 5;

    // Write data using the library function
    let config = WriteConfig {
        rows,
        dimension,
        projection_bits,
        chunk_size: 50,
        rand_categorical_cardinality,
        progress: false,
    };

    write_data(&db_path, config)
        .await
        .expect("Failed to write data");

    // Open and read all data
    let session = create_session();
    let file = open_file(&session, db_path).await.expect("Failed to open file");

    let all_data = file
        .scan()
        .expect("Failed to create scan")
        .into_array_stream()
        .expect("Failed to create array stream")
        .read_all()
        .await
        .expect("Failed to read all data");

    let struct_array = all_data.to_struct();

    // Assert: row count matches
    assert_eq!(struct_array.len(), rows, "Row count mismatch");

    // Assert: all expected columns exist
    let row_idxs = struct_array.field_by_name(ROW_IDX_COL).expect("Missing row_idx column");
    let _ids = struct_array.field_by_name(ID_COL).expect("Missing id column");
    let vectors = struct_array.field_by_name(VECTOR_COL).expect("Missing vector column");
    let projections = struct_array.field_by_name(PROJECTION_COL).expect("Missing projection column");
    let ivf_partitions = struct_array.field_by_name(IVF_PARTITION_IDX_COL).expect("Missing ivf_partition_idx column");
    let rand_floats = struct_array.field_by_name(RAND_FLOAT_COL).expect("Missing rand_float column");
    let rand_cat1 = struct_array.field_by_name(RAND_CATEGORICAL_1_COL).expect("Missing rand_categorical_1 column");
    let rand_cat2 = struct_array.field_by_name(RAND_CATEGORICAL_2_COL).expect("Missing rand_categorical_2 column");

    // Assert: row indices are sequential 0 to N-1
    let row_idxs = row_idxs.to_primitive();
    for i in 0..rows {
        let val: u64 = row_idxs.scalar_at(i).as_primitive().typed_value().unwrap();
        assert_eq!(val, i as u64, "Row index mismatch at position {}", i);
    }

    // Assert: vectors have correct dimension
    let vector_size = match vectors.dtype() {
        DType::FixedSizeList(_, size, _) => *size as usize,
        _ => panic!("Expected FixedSizeList dtype for vectors"),
    };
    assert_eq!(vector_size, dimension, "Vector dimension mismatch");

    // Assert: projections have correct size
    let expected_projection_bits = (projection_bits * dimension as f64) as usize;
    let projection_size = match projections.dtype() {
        DType::FixedSizeList(_, size, _) => *size as usize,
        _ => panic!("Expected FixedSizeList dtype for projections"),
    };
    assert_eq!(
        projection_size,
        expected_projection_bits,
        "Projection bits mismatch"
    );

    // Assert: IVF partition indices are within expected range
    let ivf_partitions_prim = ivf_partitions.to_primitive();
    let num_ivf_partitions = rows.isqrt() as u32;
    for i in 0..rows {
        let val: u32 = ivf_partitions_prim.scalar_at(i).as_primitive().typed_value().unwrap();
        assert!(
            val < num_ivf_partitions,
            "IVF partition {} out of range [0, {}) at row {}",
            val,
            num_ivf_partitions,
            i
        );
    }

    // Assert: random floats are in [0, 1) range
    let rand_floats_prim = rand_floats.to_primitive();
    for i in 0..rows {
        let val: f64 = rand_floats_prim.scalar_at(i).as_primitive().typed_value().unwrap();
        assert!(
            (0.0..1.0).contains(&val),
            "Random float {} out of range [0, 1) at row {}",
            val,
            i
        );
    }

    // Assert: categorical values are within cardinality
    let rand_cat1_prim = rand_cat1.to_primitive();
    let rand_cat2_prim = rand_cat2.to_primitive();
    for i in 0..rows {
        let val1: u32 = rand_cat1_prim.scalar_at(i).as_primitive().typed_value().unwrap();
        let val2: u32 = rand_cat2_prim.scalar_at(i).as_primitive().typed_value().unwrap();
        assert!(
            val1 < rand_categorical_cardinality,
            "Categorical 1 value {} out of range [0, {}) at row {}",
            val1,
            rand_categorical_cardinality,
            i
        );
        assert!(
            val2 < rand_categorical_cardinality,
            "Categorical 2 value {} out of range [0, {}) at row {}",
            val2,
            rand_categorical_cardinality,
            i
        );
    }
}
