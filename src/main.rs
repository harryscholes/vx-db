use std::path::PathBuf;

use clap::{Parser, Subcommand};
use vx_db::{ReadConfig, WriteConfig, read_command, write_command};

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

#[derive(Debug, Parser)]
struct WriteArgs {
    #[arg(long, short = 'f', default_value = "db.vortex")]
    path: PathBuf,
    #[arg(long, short = 'n', default_value_t = 1024)]
    rows: usize,
    #[arg(long, short = 'd', default_value_t = 1024)]
    dimension: usize,
    #[arg(long, short = 'b', default_value_t = 1.0)]
    projection_bits: f64,
    #[arg(long, short = 'c', default_value_t = 1024)]
    chunk_size: usize,
    #[arg(long, default_value_t = 5)]
    rand_categorical_cardinality: u32,
    #[arg(long)]
    progress: bool,
}

#[derive(Debug, Parser)]
struct ReadArgs {
    #[arg(long, short = 'f', default_value = "db.vortex")]
    path: PathBuf,
    #[arg(long, short = 'n', default_value_t = 1024)]
    rows: usize,
    #[arg(long, short = 'd', default_value_t = 1024)]
    dimension: usize,
    #[arg(long, short = 'b', default_value_t = 1.0)]
    projection_bits: f64,
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

impl From<WriteArgs> for WriteConfig {
    fn from(args: WriteArgs) -> Self {
        WriteConfig {
            path: args.path,
            rows: args.rows,
            dimension: args.dimension,
            projection_bits: args.projection_bits,
            chunk_size: args.chunk_size,
            rand_categorical_cardinality: args.rand_categorical_cardinality,
            progress: args.progress,
        }
    }
}

impl From<ReadArgs> for ReadConfig {
    fn from(args: ReadArgs) -> Self {
        ReadConfig {
            path: args.path,
            rows: args.rows,
            dimension: args.dimension,
            projection_bits: args.projection_bits,
            top_k: args.top_k,
            queries: args.queries,
            tombstones: args.tombstones,
            include_values: args.include_values,
            include_metadata: args.include_metadata,
            progress: args.progress,
            rand_categorical_cardinality: args.rand_categorical_cardinality,
            rand_float_selectivity: args.rand_float_selectivity,
            print_results: args.print_results,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::parse();

    match opt.command {
        Commands::Write(args) => write_command(args.into()).await,
        Commands::Read(args) => read_command(args.into()).await,
    }
}
