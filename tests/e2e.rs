use std::process::Command;
use tempfile::tempdir;

#[test]
fn test_write_then_read() {
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
