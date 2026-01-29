//! # Run a single backtest
//! mahler-backtest run --config config/default.toml --data data/orats
//!
//! # Run walk-forward optimization
//! mahler-backtest optimize --config config/default.toml --data data/orats
//!
//! # Validate parameters on test data
//! mahler-backtest validate --params results/best_params.toml --data data/orats

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "mahler-backtest")]
#[command(about = "High-performance options backtesting engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single backtest with given configuration
    Run {
        /// Path to configuration file
        #[arg(short, long)]
        config: String,

        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,
    },

    /// Run walk-forward parameter optimization
    Optimize {
        /// Path to configuration file
        #[arg(short, long)]
        config: String,

        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,

        /// Output directory for results
        #[arg(short, long, default_value = "results")]
        output: String,
    },

    /// Validate parameters on test data
    Validate {
        /// Path to parameters file
        #[arg(short, long)]
        params: String,

        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { config, data } => {
            println!("Running backtest...");
            println!("  Config: {}", config);
            println!("  Data: {}", data);
            println!("\nNot yet implemented. Use 'orats-download' to download data first.");
        }
        Commands::Optimize {
            config,
            data,
            output,
        } => {
            println!("Running walk-forward optimization...");
            println!("  Config: {}", config);
            println!("  Data: {}", data);
            println!("  Output: {}", output);
            println!("\nNot yet implemented.");
        }
        Commands::Validate { params, data } => {
            println!("Validating parameters...");
            println!("  Params: {}", params);
            println!("  Data: {}", data);
            println!("\nNot yet implemented.");
        }
    }
}
