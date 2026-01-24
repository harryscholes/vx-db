use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use vx_db::{DatabaseConfig, ReadConfig, WriteConfig, read_command, write_command};

#[derive(Debug, Parser)]
struct Opt {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Write data to a Vortex file
    Write(WriteArgs),
    /// Read and query data from a Vortex file
    Read(ReadArgs),
}

/// Shared database configuration arguments.
#[derive(Debug, Args)]
struct DatabaseArgs {
    #[arg(long, short = 'n', default_value_t = 1024)]
    rows: usize,
    #[arg(long, short = 'd', default_value_t = 1024)]
    dimension: usize,
    #[arg(long, short = 'b', default_value_t = 1.0)]
    projection_bits: f64,
    #[arg(long, default_value_t = 5)]
    rand_categorical_cardinality: u32,
}

impl From<DatabaseArgs> for DatabaseConfig {
    fn from(args: DatabaseArgs) -> Self {
        DatabaseConfig {
            rows: args.rows,
            dimension: args.dimension,
            projection_bits: args.projection_bits,
            rand_categorical_cardinality: args.rand_categorical_cardinality,
        }
    }
}

#[derive(Debug, Parser)]
struct WriteArgs {
    #[arg(long, short = 'f', default_value = "db.vortex")]
    path: PathBuf,
    #[command(flatten)]
    db: DatabaseArgs,
    #[arg(long, short = 'c', default_value_t = 1024)]
    chunk_size: usize,
    #[arg(long)]
    progress: bool,
}

#[derive(Debug, Parser)]
struct ReadArgs {
    #[arg(long, short = 'f', default_value = "db.vortex")]
    path: PathBuf,
    #[command(flatten)]
    db: DatabaseArgs,
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
    #[arg(long, default_value_t = 0.5)]
    rand_float_selectivity: f64,
    #[arg(long)]
    print_results: bool,
}

impl From<WriteArgs> for WriteConfig {
    fn from(args: WriteArgs) -> Self {
        WriteConfig {
            path: args.path,
            db: args.db.into(),
            chunk_size: args.chunk_size,
            progress: args.progress,
        }
    }
}

impl From<ReadArgs> for ReadConfig {
    fn from(args: ReadArgs) -> Self {
        ReadConfig {
            path: args.path,
            db: args.db.into(),
            top_k: args.top_k,
            queries: args.queries,
            tombstones: args.tombstones,
            include_values: args.include_values,
            include_metadata: args.include_metadata,
            progress: args.progress,
            rand_float_selectivity: args.rand_float_selectivity,
            print_results: args.print_results,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::parse();

    match opt.command {
        Commands::Write(args) => write_command(args.into()).await,
        Commands::Read(args) => read_command(args.into()).await,
    }
}
