//! ORATS Data Downloader
//!
//! Downloads historical options data from ORATS API and stores in Parquet format.
//!
//! # Usage
//!
//! ```bash
//! # Set API token
//! export ORATS_API_KEY=your-token
//!
//! # Explore API and data availability
//! orats-download explore --tickers SPY,QQQ,IWM
//!
//! # Download all data
//! orats-download download --tickers SPY,QQQ,IWM
//!
//! # Download specific date range
//! orats-download download --tickers SPY --start 2020-01-01 --end 2020-12-31
//!
//! # Resume interrupted download
//! orats-download download --resume
//!
//! # Validate downloaded data
//! orats-download validate --tickers SPY,QQQ,IWM
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{Datelike, Duration, NaiveDate, Utc};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::warn;

use mahler_backtest::data::{ORATSClient, RawStrikeRecord, records_to_snapshot};

const SEPARATOR: &str = "============================================================";

/// ORATS data downloader CLI.
#[derive(Parser)]
#[command(name = "orats-download")]
#[command(about = "Download historical options data from ORATS API")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Data output directory
    #[arg(long, default_value = "data/orats")]
    data_dir: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Explore API and check data availability
    Explore {
        /// Comma-separated list of tickers
        #[arg(long, default_value = "SPY,QQQ,IWM")]
        tickers: String,
    },

    /// Download historical data
    Download {
        /// Comma-separated list of tickers
        #[arg(long, default_value = "SPY,QQQ,IWM")]
        tickers: String,

        /// Start date (YYYY-MM-DD)
        #[arg(long, default_value = "2007-01-01")]
        start: String,

        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: Option<String>,

        /// Resume from previous progress
        #[arg(long)]
        resume: bool,
    },

    /// Validate downloaded data
    Validate {
        /// Comma-separated list of tickers
        #[arg(long, default_value = "SPY,QQQ,IWM")]
        tickers: String,
    },
}

/// Download progress tracking.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DownloadProgress {
    tickers: Vec<String>,
    completed_dates: HashMap<String, Vec<String>>, // ticker -> dates
    total_rows_downloaded: u64,
    total_requests_made: u64,
    last_updated: String,
    errors: Vec<DownloadError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DownloadError {
    ticker: String,
    date: String,
    error: String,
    timestamp: String,
}

impl DownloadProgress {
    fn load(path: &PathBuf) -> Self {
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
                Err(_) => Self::default(),
            }
        } else {
            Self::default()
        }
    }

    fn save(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

/// Generate list of trading days (Mon-Fri).
fn trading_days(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut days = Vec::new();
    let mut current = start;
    while current <= end {
        // Monday = 0, Friday = 4
        if current.weekday().num_days_from_monday() < 5 {
            days.push(current);
        }
        current += Duration::days(1);
    }
    days
}

/// Convert raw records to a DataFrame for Parquet storage.
fn records_to_dataframe(records: &[RawStrikeRecord]) -> Result<DataFrame> {
    // Build column vectors
    let mut ticker: Vec<&str> = Vec::with_capacity(records.len() * 2);
    let mut trade_date: Vec<&str> = Vec::with_capacity(records.len() * 2);
    let mut expir_date: Vec<&str> = Vec::with_capacity(records.len() * 2);
    let mut dte: Vec<i32> = Vec::with_capacity(records.len() * 2);
    let mut strike: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut option_type: Vec<&str> = Vec::with_capacity(records.len() * 2);
    let mut stock_price: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut bid: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut ask: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut volume: Vec<i64> = Vec::with_capacity(records.len() * 2);
    let mut open_interest: Vec<i64> = Vec::with_capacity(records.len() * 2);
    let mut bid_iv: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut mid_iv: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut ask_iv: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut smv_vol: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut delta: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut gamma: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut theta: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut vega: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut rho: Vec<f64> = Vec::with_capacity(records.len() * 2);
    let mut theoretical_value: Vec<f64> = Vec::with_capacity(records.len() * 2);

    for record in records {
        // Add call record
        if record.call_bid_price.is_some() && record.call_ask_price.is_some() {
            ticker.push(&record.ticker);
            trade_date.push(&record.trade_date);
            expir_date.push(&record.expir_date);
            dte.push(record.dte);
            strike.push(record.strike);
            option_type.push("C");
            stock_price.push(record.stock_price);
            bid.push(record.call_bid_price.unwrap_or(0.0));
            ask.push(record.call_ask_price.unwrap_or(0.0));
            volume.push(record.call_volume.unwrap_or(0));
            open_interest.push(record.call_open_interest.unwrap_or(0));
            bid_iv.push(record.call_bid_iv.unwrap_or(0.0));
            mid_iv.push(record.call_mid_iv.unwrap_or(0.0));
            ask_iv.push(record.call_ask_iv.unwrap_or(0.0));
            smv_vol.push(record.smv_vol.unwrap_or(0.0));
            delta.push(record.delta.unwrap_or(0.0));
            gamma.push(record.gamma.unwrap_or(0.0));
            theta.push(record.theta.unwrap_or(0.0));
            vega.push(record.vega.unwrap_or(0.0));
            rho.push(record.rho.unwrap_or(0.0));
            theoretical_value.push(record.call_value.unwrap_or(0.0));
        }

        // Add put record
        if record.put_bid_price.is_some() && record.put_ask_price.is_some() {
            ticker.push(&record.ticker);
            trade_date.push(&record.trade_date);
            expir_date.push(&record.expir_date);
            dte.push(record.dte);
            strike.push(record.strike);
            option_type.push("P");
            stock_price.push(record.stock_price);
            bid.push(record.put_bid_price.unwrap_or(0.0));
            ask.push(record.put_ask_price.unwrap_or(0.0));
            volume.push(record.put_volume.unwrap_or(0));
            open_interest.push(record.put_open_interest.unwrap_or(0));
            bid_iv.push(record.put_bid_iv.unwrap_or(0.0));
            mid_iv.push(record.put_mid_iv.unwrap_or(0.0));
            ask_iv.push(record.put_ask_iv.unwrap_or(0.0));
            smv_vol.push(record.smv_vol.unwrap_or(0.0));
            // Store delta as negative for puts
            delta.push(-record.delta.unwrap_or(0.0).abs());
            gamma.push(record.gamma.unwrap_or(0.0));
            theta.push(record.theta.unwrap_or(0.0));
            vega.push(record.vega.unwrap_or(0.0));
            rho.push(record.rho.unwrap_or(0.0));
            theoretical_value.push(record.put_value.unwrap_or(0.0));
        }
    }

    let df = DataFrame::new(vec![
        Series::new("ticker".into(), ticker).into(),
        Series::new("trade_date".into(), trade_date).into(),
        Series::new("expir_date".into(), expir_date).into(),
        Series::new("dte".into(), dte).into(),
        Series::new("strike".into(), strike).into(),
        Series::new("option_type".into(), option_type).into(),
        Series::new("stock_price".into(), stock_price).into(),
        Series::new("bid".into(), bid).into(),
        Series::new("ask".into(), ask).into(),
        Series::new("volume".into(), volume).into(),
        Series::new("open_interest".into(), open_interest).into(),
        Series::new("bid_iv".into(), bid_iv).into(),
        Series::new("mid_iv".into(), mid_iv).into(),
        Series::new("ask_iv".into(), ask_iv).into(),
        Series::new("smv_vol".into(), smv_vol).into(),
        Series::new("delta".into(), delta).into(),
        Series::new("gamma".into(), gamma).into(),
        Series::new("theta".into(), theta).into(),
        Series::new("vega".into(), vega).into(),
        Series::new("rho".into(), rho).into(),
        Series::new("theoretical_value".into(), theoretical_value).into(),
    ])?;

    Ok(df)
}

/// Save DataFrame to Parquet file (append if exists).
fn save_to_parquet(
    data_dir: &PathBuf,
    ticker: &str,
    year: i32,
    df: DataFrame,
) -> Result<PathBuf> {
    let output_dir = data_dir.join("strikes").join(ticker);
    fs::create_dir_all(&output_dir)?;

    let output_path = output_dir.join(format!("{}_{}.parquet", ticker, year));

    // If file exists, read and concatenate
    let final_df = if output_path.exists() {
        let existing_df = LazyFrame::scan_parquet(&output_path, ScanArgsParquet::default())?
            .collect()?;

        // Concatenate and deduplicate
        let combined = concat(
            [existing_df.lazy(), df.lazy()],
            UnionArgs::default(),
        )?
        .unique(
            Some(vec![
                "ticker".into(),
                "trade_date".into(),
                "expir_date".into(),
                "strike".into(),
                "option_type".into(),
            ]),
            UniqueKeepStrategy::Last,
        )
        .collect()?;

        combined
    } else {
        df
    };

    // Write to Parquet with compression
    let file = fs::File::create(&output_path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(Some(ZstdLevel::try_new(3)?)))
        .finish(&mut final_df.clone())?;

    Ok(output_path)
}

async fn cmd_explore(tickers: Vec<&str>) -> Result<()> {
    let token = std::env::var("ORATS_API_KEY")
        .context("ORATS_API_KEY environment variable not set")?;

    let mut client = ORATSClient::new(token);

    println!("{}", SEPARATOR);
    println!("ORATS API Exploration");
    println!("{}", SEPARATOR);

    // 1. Check ticker availability
    println!("\n1. Checking ticker availability...");
    for ticker in &tickers {
        match client.get_tickers(Some(ticker)).await {
            Ok(info) => {
                if let Some(t) = info.first() {
                    println!(
                        "   {}: {} to {}",
                        t.ticker,
                        t.min_date().map(|d| d.to_string()).unwrap_or("?".into()),
                        t.max_date().map(|d| d.to_string()).unwrap_or("?".into())
                    );
                } else {
                    println!("   {}: NOT FOUND", ticker);
                }
            }
            Err(e) => {
                println!("   {}: ERROR - {}", ticker, e);
            }
        }
    }

    // 2. Sample data request
    println!("\n2. Sample strikes data (recent date)...");
    let sample_ticker = tickers.first().unwrap_or(&"SPY");
    let today = Utc::now().date_naive();

    for offset in 1..10 {
        let try_date = today - Duration::days(offset);
        match client.get_strikes_history(sample_ticker, try_date).await {
            Ok(records) if !records.is_empty() => {
                println!("   Date: {}", try_date);
                println!("   Rows returned: {}", records.len());

                if let Some(first) = records.first() {
                    println!("\n   Sample record:");
                    println!("     ticker: {}", first.ticker);
                    println!("     tradeDate: {}", first.trade_date);
                    println!("     expirDate: {}", first.expir_date);
                    println!("     strike: {}", first.strike);
                    println!("     stockPrice: {}", first.stock_price);
                    println!("     callBidPrice: {:?}", first.call_bid_price);
                    println!("     callAskPrice: {:?}", first.call_ask_price);
                    println!("     delta: {:?}", first.delta);
                    println!("     smvVol: {:?}", first.smv_vol);
                }

                // Convert to snapshot and show summary
                let snapshot = records_to_snapshot(sample_ticker, try_date, records);
                println!("\n   Converted to OptionsSnapshot:");
                println!("     Underlying: {} @ {}", snapshot.ticker, snapshot.underlying_price);
                println!("     Chains: {}", snapshot.chains.len());
                println!("     Total quotes: {}", snapshot.total_quotes());

                break;
            }
            Ok(_) => continue,
            Err(e) => {
                warn!("   {}: No data ({})", try_date, e);
            }
        }
    }

    // 3. Estimate work
    println!("\n3. Download estimation for {:?}:", tickers);
    let start_date = NaiveDate::from_ymd_opt(2007, 1, 1).unwrap();
    let days = trading_days(start_date, today);
    let requests_needed = days.len() * tickers.len();

    println!("   Trading days (2007-present): ~{}", days.len());
    println!("   Requests needed: ~{}", requests_needed);
    println!("   Monthly limit: 20,000");
    println!("   Months needed: ~{:.1}", requests_needed as f64 / 20_000.0);

    println!("\n{}", SEPARATOR);

    Ok(())
}

async fn cmd_download(
    data_dir: PathBuf,
    tickers: Vec<&str>,
    start_date: NaiveDate,
    end_date: NaiveDate,
    resume: bool,
) -> Result<()> {
    let token = std::env::var("ORATS_API_KEY")
        .context("ORATS_API_KEY environment variable not set")?;

    let mut client = ORATSClient::new(token);

    // Progress tracking
    let progress_file = data_dir.join("download_progress.json");
    let mut progress = if resume {
        DownloadProgress::load(&progress_file)
    } else {
        DownloadProgress::default()
    };
    progress.tickers = tickers.iter().map(|s| s.to_string()).collect();

    // Get trading days
    let days = trading_days(start_date, end_date);
    let total_days = days.len();

    // Estimate work
    let estimated_requests = total_days * tickers.len();
    println!("\nDownload Plan:");
    println!("  Tickers: {:?}", tickers);
    println!("  Date range: {} to {}", start_date, end_date);
    println!("  Trading days: {}", total_days);
    println!("  Estimated requests: {}", estimated_requests);
    println!("  Monthly limit: 20,000");

    if estimated_requests > 20_000 {
        println!("\n  WARNING: Estimated requests exceed monthly limit!");
        println!("  Consider downloading in smaller batches.");
    }

    println!("\nStarting download...\n");

    // Buffer for batching writes
    let mut write_buffer: HashMap<(String, i32), Vec<RawStrikeRecord>> = HashMap::new();
    let mut buffer_rows = 0;
    let max_buffer_rows = 50_000;

    let start_time = Instant::now();

    for ticker in &tickers {
        let ticker_str = ticker.to_string();

        // Initialize completed dates
        if !progress.completed_dates.contains_key(*ticker) {
            progress.completed_dates.insert(ticker_str.clone(), Vec::new());
        }

        let completed_set: HashSet<String> = progress
            .completed_dates
            .get(*ticker)
            .unwrap_or(&Vec::new())
            .iter()
            .cloned()
            .collect();

        // Progress bar
        let pb = ProgressBar::new(total_days as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")?
                .progress_chars("=>-"),
        );
        pb.set_message(format!("{}", ticker));

        for &trade_date in days.iter() {
            let date_str = trade_date.to_string();

            // Skip if already completed
            if completed_set.contains(&date_str) {
                pb.inc(1);
                continue;
            }

            pb.set_message(format!("{} {}", ticker, trade_date));

            // Download data with retries
            let mut records: Option<Vec<RawStrikeRecord>> = None;
            for attempt in 0..3 {
                match client.get_strikes_history(ticker, trade_date).await {
                    Ok(r) => {
                        records = Some(r);
                        break;
                    }
                    Err(e) => {
                        if attempt < 2 {
                            tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt))).await;
                        } else {
                            progress.errors.push(DownloadError {
                                ticker: ticker_str.clone(),
                                date: date_str.clone(),
                                error: format!("{}", e),
                                timestamp: Utc::now().to_rfc3339(),
                            });
                        }
                    }
                }
            }

            if let Some(recs) = records {
                if !recs.is_empty() {
                    // Add to buffer
                    let year = trade_date.year();
                    let key = (ticker_str.clone(), year);
                    write_buffer.entry(key).or_default().extend(recs.iter().cloned());
                    buffer_rows += recs.len() * 2; // calls + puts
                    progress.total_rows_downloaded += recs.len() as u64 * 2;
                }

                // Mark as completed
                progress
                    .completed_dates
                    .get_mut(*ticker)
                    .unwrap()
                    .push(date_str);
                progress.total_requests_made += 1;
            }

            pb.inc(1);

            // Flush buffer if large enough
            if buffer_rows >= max_buffer_rows {
                pb.set_message(format!("Flushing {} rows...", buffer_rows));
                for ((t, y), recs) in write_buffer.drain() {
                    let df = records_to_dataframe(&recs)?;
                    save_to_parquet(&data_dir, &t, y, df)?;
                }
                buffer_rows = 0;
            }

            // Save progress periodically
            if progress.total_requests_made % 100 == 0 {
                progress.last_updated = Utc::now().to_rfc3339();
                progress.save(&progress_file)?;
            }
        }

        pb.finish_with_message(format!("{} complete", ticker));
    }

    // Final buffer flush
    if !write_buffer.is_empty() {
        println!("\nFlushing final {} rows...", buffer_rows);
        for ((t, y), recs) in write_buffer.drain() {
            let df = records_to_dataframe(&recs)?;
            save_to_parquet(&data_dir, &t, y, df)?;
        }
    }

    // Save final progress
    progress.last_updated = Utc::now().to_rfc3339();
    progress.save(&progress_file)?;

    let elapsed = start_time.elapsed();
    println!("\nDownload Complete!");
    println!("  Total requests: {}", progress.total_requests_made);
    println!("  Total rows: {}", progress.total_rows_downloaded);
    println!("  Elapsed time: {:.1} minutes", elapsed.as_secs_f64() / 60.0);
    println!("  Errors: {}", progress.errors.len());

    Ok(())
}

async fn cmd_validate(data_dir: PathBuf, tickers: Vec<&str>) -> Result<()> {
    println!("Validating downloaded data...\n");

    for ticker in &tickers {
        let ticker_dir = data_dir.join("strikes").join(ticker);
        if !ticker_dir.exists() {
            println!("{}: No data found", ticker);
            continue;
        }

        let parquet_files: Vec<_> = fs::read_dir(&ticker_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "parquet").unwrap_or(false))
            .collect();

        let mut total_rows = 0u64;
        let mut all_dates: Vec<String> = Vec::new();

        for entry in &parquet_files {
            let path = entry.path();
            let df = LazyFrame::scan_parquet(&path, ScanArgsParquet::default())?
                .collect()?;

            total_rows += df.height() as u64;

            // Get unique dates
            let dates = df
                .column("trade_date")?
                .unique()?
                .str()?
                .into_iter()
                .filter_map(|x| x.map(|s| s.to_string()))
                .collect::<Vec<_>>();

            all_dates.extend(dates);
        }

        all_dates.sort();
        all_dates.dedup();

        let min_date = all_dates.first().cloned().unwrap_or_default();
        let max_date = all_dates.last().cloned().unwrap_or_default();

        println!("{}:", ticker);
        println!("  Files: {}", parquet_files.len());
        println!("  Total rows: {}", total_rows);
        println!("  Date range: {} to {}", min_date, max_date);
        println!("  Unique dates: {}", all_dates.len());
        println!();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mahler_backtest=info".parse()?),
        )
        .init();

    let cli = Cli::parse();

    // Ensure data directory exists
    fs::create_dir_all(&cli.data_dir)?;

    match cli.command {
        Commands::Explore { tickers } => {
            let ticker_list: Vec<&str> = tickers.split(',').map(|s| s.trim()).collect();
            cmd_explore(ticker_list).await?;
        }
        Commands::Download {
            tickers,
            start,
            end,
            resume,
        } => {
            let ticker_list: Vec<&str> = tickers.split(',').map(|s| s.trim()).collect();
            let start_date = NaiveDate::parse_from_str(&start, "%Y-%m-%d")
                .context("Invalid start date format")?;
            let end_date = match end {
                Some(e) => NaiveDate::parse_from_str(&e, "%Y-%m-%d")
                    .context("Invalid end date format")?,
                None => Utc::now().date_naive(),
            };

            cmd_download(cli.data_dir, ticker_list, start_date, end_date, resume).await?;
        }
        Commands::Validate { tickers } => {
            let ticker_list: Vec<&str> = tickers.split(',').map(|s| s.trim()).collect();
            cmd_validate(cli.data_dir, ticker_list).await?;
        }
    }

    Ok(())
}
