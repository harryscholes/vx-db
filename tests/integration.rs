use tempfile::TempDir;
use vx_db::{ReadConfig, WriteConfig, read_command, write_command};

#[tokio::test]
async fn test_write_then_read() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.vortex");

    let rows = 100;
    let dimension = 64;
    let projection_bits = 1.0;
    let rand_categorical_cardinality = 3;

    // Write data
    write_command(WriteConfig {
        path: db_path.clone(),
        rows,
        dimension,
        projection_bits,
        chunk_size: 50,
        rand_categorical_cardinality,
        progress: false,
    })
    .await
    .expect("write_command should succeed");

    // Assert: file was created
    assert!(db_path.exists(), "database file should exist after write");

    // Assert: file has content
    let metadata = std::fs::metadata(&db_path).expect("should read file metadata");
    assert!(metadata.len() > 0, "database file should have content");

    // Read data back - use matching parameters
    read_command(ReadConfig {
        path: db_path,
        rows,
        dimension,
        projection_bits,
        top_k: 5,
        queries: 1,
        tombstones: 0.0,
        include_values: true,
        include_metadata: true,
        progress: false,
        rand_categorical_cardinality,
        rand_float_selectivity: 1.0,
        print_results: false,
    })
    .await
    .expect("read_command should succeed on written data");
}
