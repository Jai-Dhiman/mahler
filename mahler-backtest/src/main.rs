//! Mahler Backtest CLI
//!
//! High-performance options backtesting engine.
//!
//! # Usage
//!
//! ## Run a single backtest
//! ```bash
//! mahler-backtest run --ticker SPY --start 2020-01-01 --end 2024-12-31
//! ```
//!
//! ## Run walk-forward optimization
//! ```bash
//! mahler-backtest optimize --ticker SPY --start 2020-01-01 --end 2024-12-31
//! ```
//!
//! ## Validate data integrity
//! ```bash
//! mahler-backtest validate --ticker SPY
//! ```

use std::time::Instant;

use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use mahler_backtest::{
    BacktestConfig, BacktestEngine, DataIntegrityValidator, GreeksValidator, MetricsCalculator,
    ParameterGrid, SlippageModel, WalkForwardOptimizer,
};
use mahler_backtest::data::DataLoader;
use mahler_backtest::walkforward::periods::WalkForwardPeriodsConfig;

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
    /// Run a single backtest with given parameters
    Run {
        /// Ticker symbol (e.g., SPY)
        #[arg(short, long)]
        ticker: String,

        /// Start date (YYYY-MM-DD)
        #[arg(short, long)]
        start: String,

        /// End date (YYYY-MM-DD)
        #[arg(short, long)]
        end: String,

        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,

        /// Initial equity
        #[arg(long, default_value = "100000")]
        equity: u64,

        /// Profit target percentage
        #[arg(long, default_value = "50")]
        profit_target: f64,

        /// Stop loss percentage
        #[arg(long, default_value = "125")]
        stop_loss: f64,

        /// Minimum DTE for trades
        #[arg(long, default_value = "30")]
        min_dte: i32,

        /// Maximum DTE for trades
        #[arg(long, default_value = "45")]
        max_dte: i32,

        /// Minimum delta for short strike (absolute value)
        #[arg(long, default_value = "0.20")]
        min_delta: f64,

        /// Maximum delta for short strike (absolute value)
        #[arg(long, default_value = "0.30")]
        max_delta: f64,

        /// Minimum IV percentile for entry (0-100)
        #[arg(long, default_value = "50")]
        min_iv_percentile: f64,

        /// Maximum risk per trade as percentage of equity (e.g., 2.0 = 2%)
        #[arg(long, default_value = "2.0")]
        max_risk_per_trade: f64,

        /// Maximum portfolio risk as percentage of equity (e.g., 10.0 = 10%)
        #[arg(long, default_value = "10.0")]
        max_portfolio_risk: f64,

        /// Slippage model: "orats" (default, 66% for 2-leg), "pessimistic" (100%), "zero" (50%), or a percentage (e.g., "75" for 75%)
        #[arg(long, default_value = "orats")]
        slippage: String,

        /// Enable scaled position sizing (calculates contracts based on risk)
        #[arg(long)]
        use_scaled_sizing: bool,

        /// Maximum exposure to equity-correlated assets (SPY/QQQ/IWM) as %
        #[arg(long, default_value = "50.0")]
        max_correlated_exposure: f64,

        /// Maximum single position risk as % of equity (e.g., 5.0 = 5%)
        #[arg(long, default_value = "5.0")]
        max_single_position: f64,
    },

    /// Run walk-forward parameter optimization
    Optimize {
        /// Ticker symbol (e.g., SPY)
        #[arg(short, long)]
        ticker: String,

        /// Start date (YYYY-MM-DD)
        #[arg(short, long)]
        start: String,

        /// End date (YYYY-MM-DD)
        #[arg(short, long)]
        end: String,

        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,

        /// Training period months
        #[arg(long, default_value = "6")]
        train_months: u32,

        /// Validation period months
        #[arg(long, default_value = "1")]
        validate_months: u32,

        /// Test period months
        #[arg(long, default_value = "1")]
        test_months: u32,
    },

    /// Validate data integrity and Greeks
    Validate {
        /// Ticker symbol (e.g., SPY) or "all" for all tickers
        #[arg(short, long)]
        ticker: String,

        /// Year to validate (optional, validates all years if not specified)
        #[arg(short, long)]
        year: Option<i32>,

        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,

        /// Skip Greeks validation
        #[arg(long)]
        skip_greeks: bool,
    },

    /// Show data summary
    Info {
        /// Path to data directory
        #[arg(short, long, default_value = "data/orats")]
        data: String,
    },
}

fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            ticker,
            start,
            end,
            data,
            equity,
            profit_target,
            stop_loss,
            min_dte,
            max_dte,
            min_delta,
            max_delta,
            min_iv_percentile,
            max_risk_per_trade,
            max_portfolio_risk,
            slippage,
            use_scaled_sizing,
            max_correlated_exposure,
            max_single_position,
        } => {
            run_backtest(
                &ticker,
                &start,
                &end,
                &data,
                equity,
                profit_target,
                stop_loss,
                min_dte,
                max_dte,
                min_delta,
                max_delta,
                min_iv_percentile,
                max_risk_per_trade,
                max_portfolio_risk,
                &slippage,
                use_scaled_sizing,
                max_correlated_exposure,
                max_single_position,
            );
        }
        Commands::Optimize {
            ticker,
            start,
            end,
            data,
            train_months,
            validate_months,
            test_months,
        } => {
            run_optimize(
                &ticker,
                &start,
                &end,
                &data,
                train_months,
                validate_months,
                test_months,
            );
        }
        Commands::Validate {
            ticker,
            year,
            data,
            skip_greeks,
        } => {
            run_validate(&ticker, year, &data, skip_greeks);
        }
        Commands::Info { data } => {
            show_info(&data);
        }
    }
}

fn run_backtest(
    ticker: &str,
    start: &str,
    end: &str,
    data_dir: &str,
    equity: u64,
    profit_target: f64,
    stop_loss: f64,
    min_dte: i32,
    max_dte: i32,
    min_delta: f64,
    max_delta: f64,
    min_iv_percentile: f64,
    max_risk_per_trade: f64,
    max_portfolio_risk: f64,
    slippage_arg: &str,
    use_scaled_sizing: bool,
    max_correlated_exposure: f64,
    max_single_position: f64,
) {
    info!("Running backtest for {}", ticker);

    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .expect("Invalid start date format. Use YYYY-MM-DD");
    let end_date =
        NaiveDate::parse_from_str(end, "%Y-%m-%d").expect("Invalid end date format. Use YYYY-MM-DD");

    // Parse slippage model
    let slippage = match slippage_arg.to_lowercase().as_str() {
        "orats" => SlippageModel::orats(),
        "pessimistic" => SlippageModel::pessimistic(),
        "zero" => SlippageModel::zero(),
        other => {
            // Try to parse as a percentage (e.g., "75" for 75%)
            if let Ok(pct) = other.parse::<f64>() {
                let fill_pct = pct / 100.0;
                SlippageModel::new(vec![fill_pct, fill_pct, fill_pct, fill_pct])
            } else {
                eprintln!("Invalid slippage model '{}', using ORATS default", other);
                SlippageModel::orats()
            }
        }
    };

    let config = BacktestConfig {
        initial_equity: equity.into(),
        profit_target_pct: profit_target,
        stop_loss_pct: stop_loss,
        min_dte,
        max_dte,
        min_delta,
        max_delta,
        min_iv_percentile,
        max_risk_per_trade_pct: max_risk_per_trade,
        max_portfolio_risk_pct: max_portfolio_risk,
        slippage,
        use_scaled_position_sizing: use_scaled_sizing,
        max_correlated_exposure_pct: max_correlated_exposure,
        max_single_position_pct: max_single_position,
        ..Default::default()
    };

    let start_time = Instant::now();
    let mut engine = BacktestEngine::new(config, data_dir);

    match engine.run(ticker, start_date, end_date) {
        Ok(result) => {
            let elapsed = start_time.elapsed();
            println!("\n{}", result.summary());
            println!("\nBacktest completed in {:.2?}", elapsed);

            // Calculate detailed metrics
            let metrics = MetricsCalculator::calculate(&result);
            println!("\n{}", metrics.summary());
        }
        Err(e) => {
            eprintln!("Backtest failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_optimize(
    ticker: &str,
    start: &str,
    end: &str,
    data_dir: &str,
    train_months: u32,
    validate_months: u32,
    test_months: u32,
) {
    info!("Running walk-forward optimization for {}", ticker);

    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .expect("Invalid start date format. Use YYYY-MM-DD");
    let end_date =
        NaiveDate::parse_from_str(end, "%Y-%m-%d").expect("Invalid end date format. Use YYYY-MM-DD");

    let periods_config = WalkForwardPeriodsConfig {
        train_months,
        validate_months,
        test_months,
        roll_months: 1,
    };

    let param_grid = ParameterGrid::default();
    println!(
        "Parameter combinations to test: {}",
        param_grid.total_combinations()
    );

    let optimizer = WalkForwardOptimizer::new(data_dir)
        .with_periods_config(periods_config)
        .with_param_grid(param_grid);

    let start_time = Instant::now();

    match optimizer.optimize(ticker, start_date, end_date) {
        Ok(result) => {
            let elapsed = start_time.elapsed();

            // Print the comprehensive LLM analysis
            println!("\n{}", result.llm_analysis());

            println!("Optimization completed in {:.2?}", elapsed);
        }
        Err(e) => {
            eprintln!("Optimization failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_validate(ticker: &str, year: Option<i32>, data_dir: &str, skip_greeks: bool) {
    info!("Validating data for {}", ticker);

    let validator = DataIntegrityValidator::new(data_dir);

    let start_time = Instant::now();

    let reports = if ticker.to_lowercase() == "all" {
        validator
            .validate_all()
            .expect("Failed to run validation")
    } else if let Some(y) = year {
        vec![validator
            .validate(ticker, y)
            .expect("Failed to run validation")]
    } else {
        validator
            .validate_ticker(ticker)
            .expect("Failed to run validation")
    };

    println!("\nData Integrity Results:");
    println!("========================");

    let mut total_passed = 0;
    let mut total_failed = 0;

    for report in &reports {
        let status = if report.all_passed() {
            total_passed += 1;
            "PASS"
        } else {
            total_failed += 1;
            "FAIL"
        };

        println!("[{}] {}", status, report.summary());

        for check in report.failed_checks() {
            println!("  - {}: {}", check.name, check.message);
            if let Some(details) = &check.details {
                println!("    {}", details);
            }
        }
    }

    println!(
        "\nTotal: {} passed, {} failed",
        total_passed, total_failed
    );

    // Greeks validation
    if !skip_greeks {
        println!("\nGreeks Validation:");
        println!("==================");

        let greeks_validator = GreeksValidator::new(data_dir);

        if ticker.to_lowercase() == "all" {
            let loader = DataLoader::new(data_dir);
            for t in loader.available_tickers().unwrap_or_default() {
                validate_greeks_for_ticker(&greeks_validator, &t);
            }
        } else {
            validate_greeks_for_ticker(&greeks_validator, ticker);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\nValidation completed in {:.2?}", elapsed);
}

fn validate_greeks_for_ticker(validator: &GreeksValidator, ticker: &str) {
    match validator.validate_ticker(ticker) {
        Ok(reports) => {
            for report in reports {
                let pass_threshold = 0.90; // 90% within tolerance
                let status = if report.all_pass(pass_threshold) {
                    "PASS"
                } else {
                    "WARN"
                };
                println!("[{}] {}", status, report.summary());
            }
        }
        Err(e) => {
            println!("[SKIP] {}: {}", ticker, e);
        }
    }
}

fn show_info(data_dir: &str) {
    info!("Data directory: {}", data_dir);

    let loader = DataLoader::new(data_dir);

    match loader.available_tickers() {
        Ok(tickers) => {
            println!("\nAvailable Tickers: {}", tickers.len());
            println!("==================");

            for ticker in tickers {
                if let Ok(years) = loader.available_years(&ticker) {
                    let mut total_rows = 0;
                    let mut total_days = 0;

                    for &year in &years {
                        if let Ok(rows) = loader.row_count(&ticker, year) {
                            total_rows += rows;
                        }
                        if let Ok(days) = loader.trading_day_count(&ticker, year) {
                            total_days += days;
                        }
                    }

                    let year_range = if !years.is_empty() {
                        format!("{}-{}", years.first().unwrap(), years.last().unwrap())
                    } else {
                        "N/A".to_string()
                    };

                    println!(
                        "{}: {} years ({}), {} rows, {} trading days",
                        ticker,
                        years.len(),
                        year_range,
                        total_rows,
                        total_days
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to list tickers: {}", e);
        }
    }
}
