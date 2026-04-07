//! Data loader for ORATS parquet files.
//!
//! Loads historical options data from parquet files into the type system
//! for backtesting and validation.
//!
//! The parquet files contain normalized rows with one row per option contract
//! (separate rows for calls and puts), with the following schema:
//! - ticker, trade_date, expir_date, dte, strike, option_type
//! - stock_price, bid, ask, volume, open_interest
//! - bid_iv, mid_iv, ask_iv, smv_vol
//! - delta, gamma, theta, vega, rho, theoretical_value

use std::collections::HashMap;
use std::path::Path;

use chrono::{Datelike, NaiveDate};
use polars::prelude::*;
use rust_decimal::Decimal;
use thiserror::Error;

use super::types::{Greeks, OptionQuote, OptionType, OptionsChain, OptionsSnapshot};

/// Expected columns in the parquet files.
pub const EXPECTED_COLUMNS: &[&str] = &[
    "ticker",
    "trade_date",
    "expir_date",
    "dte",
    "strike",
    "option_type",
    "stock_price",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "bid_iv",
    "mid_iv",
    "ask_iv",
    "smv_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "theoretical_value",
];

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Parquet data loader for ORATS options data.
pub struct DataLoader {
    data_dir: String,
}

impl DataLoader {
    /// Create a new data loader pointing to the ORATS data directory.
    pub fn new(data_dir: &str) -> Self {
        Self {
            data_dir: data_dir.to_string(),
        }
    }

    /// Get the path to a ticker's parquet file for a given year.
    fn parquet_path(&self, ticker: &str, year: i32) -> String {
        format!(
            "{}/strikes/{}/{}_{}.parquet",
            self.data_dir, ticker, ticker, year
        )
    }

    /// List available years for a ticker.
    pub fn available_years(&self, ticker: &str) -> Result<Vec<i32>, LoaderError> {
        let dir_path = format!("{}/strikes/{}", self.data_dir, ticker);
        let path = Path::new(&dir_path);

        if !path.exists() {
            return Ok(vec![]);
        }

        let mut years = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if name.ends_with(".parquet") {
                if let Some(year_str) = name.strip_prefix(&format!("{}_", ticker)) {
                    if let Some(year_str) = year_str.strip_suffix(".parquet") {
                        if let Ok(year) = year_str.parse::<i32>() {
                            years.push(year);
                        }
                    }
                }
            }
        }
        years.sort();
        Ok(years)
    }

    /// List available tickers.
    pub fn available_tickers(&self) -> Result<Vec<String>, LoaderError> {
        let dir_path = format!("{}/strikes", self.data_dir);
        let path = Path::new(&dir_path);

        if !path.exists() {
            return Ok(vec![]);
        }

        let mut tickers = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let name = entry.file_name();
                tickers.push(name.to_string_lossy().to_string());
            }
        }
        tickers.sort();
        Ok(tickers)
    }

    /// Load raw parquet data for a ticker and year as a LazyFrame.
    pub fn load_lazy(&self, ticker: &str, year: i32) -> Result<LazyFrame, LoaderError> {
        let path = self.parquet_path(ticker, year);
        if !Path::new(&path).exists() {
            return Err(LoaderError::FileNotFound(path));
        }
        let lf = LazyFrame::scan_parquet(&path, ScanArgsParquet::default())?;
        Ok(lf)
    }

    /// Load raw parquet data for a ticker and year as a DataFrame.
    pub fn load_dataframe(&self, ticker: &str, year: i32) -> Result<DataFrame, LoaderError> {
        Ok(self.load_lazy(ticker, year)?.collect()?)
    }

    /// Load all data for a ticker across multiple years.
    pub fn load_ticker_range(
        &self,
        ticker: &str,
        start_year: i32,
        end_year: i32,
    ) -> Result<LazyFrame, LoaderError> {
        let mut frames = Vec::new();
        for year in start_year..=end_year {
            match self.load_lazy(ticker, year) {
                Ok(lf) => frames.push(lf),
                Err(LoaderError::FileNotFound(_)) => continue,
                Err(e) => return Err(e),
            }
        }

        if frames.is_empty() {
            return Err(LoaderError::InvalidData(format!(
                "No data found for {} in years {}-{}",
                ticker, start_year, end_year
            )));
        }

        let combined = concat(&frames, UnionArgs::default())?;
        Ok(combined)
    }

    /// Load data for a specific date range.
    pub fn load_date_range(
        &self,
        ticker: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<DataFrame, LoaderError> {
        let start_year = start_date.year();
        let end_year = end_date.year();

        let lf = self.load_ticker_range(ticker, start_year, end_year)?;

        // Filter by date range using string comparison (dates stored as strings)
        let filtered = lf.filter(
            col("trade_date")
                .gt_eq(lit(start_date.to_string()))
                .and(col("trade_date").lt_eq(lit(end_date.to_string()))),
        );

        Ok(filtered.collect()?)
    }

    /// Get unique trading dates from a DataFrame.
    pub fn get_trading_dates(&self, df: &DataFrame) -> Result<Vec<NaiveDate>, LoaderError> {
        let dates_col = df.column("trade_date")?;

        // Handle both string and date column types
        let dates: Vec<NaiveDate> = if let Ok(str_col) = dates_col.str() {
            str_col
                .into_iter()
                .filter_map(|s| s.and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()))
                .collect()
        } else if let Ok(date_col) = dates_col.date() {
            date_col
                .into_iter()
                .filter_map(|d| d.map(date_from_days))
                .collect()
        } else {
            return Err(LoaderError::InvalidData(
                "trade_date column has unexpected type".to_string(),
            ));
        };

        // Deduplicate and sort
        let mut unique_dates: Vec<_> = dates.into_iter().collect();
        unique_dates.sort();
        unique_dates.dedup();

        Ok(unique_dates)
    }

    /// Load options snapshot for a specific date.
    pub fn load_snapshot(
        &self,
        ticker: &str,
        date: NaiveDate,
    ) -> Result<OptionsSnapshot, LoaderError> {
        let year = date.year();
        let lf = self.load_lazy(ticker, year)?;

        let df = lf
            .filter(col("trade_date").eq(lit(date.to_string())))
            .collect()?;

        if df.height() == 0 {
            return Err(LoaderError::InvalidData(format!(
                "No data for {} on {}",
                ticker, date
            )));
        }

        dataframe_to_snapshot(df, ticker, date)
    }

    /// Load multiple snapshots for a date range.
    pub fn load_snapshots(
        &self,
        ticker: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<OptionsSnapshot>, LoaderError> {
        let df = self.load_date_range(ticker, start_date, end_date)?;
        let dates = self.get_trading_dates(&df)?;

        let mut snapshots = Vec::with_capacity(dates.len());

        for date in dates {
            let day_df = df
                .clone()
                .lazy()
                .filter(col("trade_date").eq(lit(date.to_string())))
                .collect()?;

            if day_df.height() > 0 {
                let snapshot = dataframe_to_snapshot(day_df, ticker, date)?;
                snapshots.push(snapshot);
            }
        }

        Ok(snapshots)
    }

    /// Get schema information for validation.
    pub fn get_schema(&self, ticker: &str, year: i32) -> Result<Schema, LoaderError> {
        let path = self.parquet_path(ticker, year);
        if !Path::new(&path).exists() {
            return Err(LoaderError::FileNotFound(path));
        }

        let mut lf = LazyFrame::scan_parquet(&path, ScanArgsParquet::default())?;
        Ok(lf.collect_schema()?.as_ref().clone())
    }

    /// Get row count for a parquet file.
    pub fn row_count(&self, ticker: &str, year: i32) -> Result<usize, LoaderError> {
        let path = self.parquet_path(ticker, year);
        if !Path::new(&path).exists() {
            return Err(LoaderError::FileNotFound(path));
        }

        let lf = LazyFrame::scan_parquet(&path, ScanArgsParquet::default())?;
        let df = lf.select([col("trade_date")]).collect()?;
        Ok(df.height())
    }

    /// Get date range covered by a parquet file.
    pub fn date_range(
        &self,
        ticker: &str,
        year: i32,
    ) -> Result<(NaiveDate, NaiveDate), LoaderError> {
        let lf = self.load_lazy(ticker, year)?;

        let stats = lf
            .select([
                col("trade_date").min().alias("min_date"),
                col("trade_date").max().alias("max_date"),
            ])
            .collect()?;

        let min_str = stats
            .column("min_date")?
            .str()?
            .get(0)
            .ok_or_else(|| LoaderError::InvalidData("No min date".to_string()))?;

        let max_str = stats
            .column("max_date")?
            .str()?
            .get(0)
            .ok_or_else(|| LoaderError::InvalidData("No max date".to_string()))?;

        let min_date = NaiveDate::parse_from_str(min_str, "%Y-%m-%d")
            .map_err(|e| LoaderError::InvalidData(format!("Invalid min date: {}", e)))?;
        let max_date = NaiveDate::parse_from_str(max_str, "%Y-%m-%d")
            .map_err(|e| LoaderError::InvalidData(format!("Invalid max date: {}", e)))?;

        Ok((min_date, max_date))
    }

    /// Count unique trading days in a parquet file.
    pub fn trading_day_count(&self, ticker: &str, year: i32) -> Result<usize, LoaderError> {
        let lf = self.load_lazy(ticker, year)?;

        let df = lf
            .select([col("trade_date").n_unique().alias("count")])
            .collect()?;

        let count = df
            .column("count")?
            .u32()?
            .get(0)
            .ok_or_else(|| LoaderError::InvalidData("No count".to_string()))?;

        Ok(count as usize)
    }
}

/// Convert days since Unix epoch to NaiveDate.
fn date_from_days(days: i32) -> NaiveDate {
    NaiveDate::from_num_days_from_ce_opt(days + 719163).unwrap_or_default()
}

/// Convert a DataFrame to an OptionsSnapshot.
///
/// The DataFrame should contain rows for a single trading day with
/// columns: ticker, trade_date, expir_date, dte, strike, option_type,
/// stock_price, bid, ask, volume, open_interest, bid_iv, mid_iv, ask_iv,
/// smv_vol, delta, gamma, theta, vega, rho, theoretical_value
fn dataframe_to_snapshot(
    df: DataFrame,
    ticker: &str,
    date: NaiveDate,
) -> Result<OptionsSnapshot, LoaderError> {
    // Get underlying price from first row
    let stock_price = df
        .column("stock_price")
        .ok()
        .and_then(|c| c.f64().ok())
        .and_then(|c| c.get(0))
        .map(|p| Decimal::from_f64_retain(p).unwrap_or_default())
        .unwrap_or_default();

    let mut snapshot = OptionsSnapshot::new(date, ticker.to_string(), stock_price);

    // Group by expiration date
    let mut chains_map: HashMap<NaiveDate, OptionsChain> = HashMap::new();

    // Get column references
    let expir_col = df.column("expir_date")?;
    let dte_col = df.column("dte")?;
    let strike_col = df.column("strike")?;
    let opt_type_col = df.column("option_type")?;
    let stock_price_col = df.column("stock_price")?;
    let bid_col = df.column("bid")?;
    let ask_col = df.column("ask")?;
    let volume_col = df.column("volume")?;
    let oi_col = df.column("open_interest")?;
    let bid_iv_col = df.column("bid_iv")?;
    let mid_iv_col = df.column("mid_iv")?;
    let ask_iv_col = df.column("ask_iv")?;
    let smv_col = df.column("smv_vol")?;
    let delta_col = df.column("delta")?;
    let gamma_col = df.column("gamma")?;
    let theta_col = df.column("theta")?;
    let vega_col = df.column("vega")?;
    let rho_col = df.column("rho")?;
    let tv_col = df.column("theoretical_value")?;

    for idx in 0..df.height() {
        // Parse expiration date
        let expiration = expir_col
            .str()
            .ok()
            .and_then(|c| c.get(idx))
            .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            .unwrap_or(date);

        let dte = dte_col.i32().ok().and_then(|c| c.get(idx)).unwrap_or(0);

        let chain = chains_map
            .entry(expiration)
            .or_insert_with(|| OptionsChain::new(expiration, dte));

        // Parse option type
        let opt_type_str = opt_type_col
            .str()
            .ok()
            .and_then(|c| c.get(idx))
            .unwrap_or("C");
        let option_type = match opt_type_str {
            "P" | "p" | "PUT" | "put" => OptionType::Put,
            _ => OptionType::Call,
        };

        // Get numeric values
        let strike = strike_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let stock_price_val = stock_price_col
            .f64()
            .ok()
            .and_then(|c| c.get(idx))
            .unwrap_or(0.0);
        let bid = bid_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let ask = ask_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let volume = volume_col.i64().ok().and_then(|c| c.get(idx)).unwrap_or(0);
        let oi = oi_col.i64().ok().and_then(|c| c.get(idx)).unwrap_or(0);
        let bid_iv = bid_iv_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let mid_iv = mid_iv_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let ask_iv = ask_iv_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let smv = smv_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let delta = delta_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let gamma = gamma_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let theta = theta_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let vega = vega_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let rho = rho_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);
        let tv = tv_col.f64().ok().and_then(|c| c.get(idx)).unwrap_or(0.0);

        let mid = (bid + ask) / 2.0;

        chain.add_quote(OptionQuote {
            ticker: ticker.to_string(),
            trade_date: date,
            expiration,
            dte,
            strike: Decimal::from_f64_retain(strike).unwrap_or_default(),
            option_type,
            stock_price: Decimal::from_f64_retain(stock_price_val).unwrap_or_default(),
            bid: Decimal::from_f64_retain(bid).unwrap_or_default(),
            ask: Decimal::from_f64_retain(ask).unwrap_or_default(),
            mid: Decimal::from_f64_retain(mid).unwrap_or_default(),
            theoretical_value: Decimal::from_f64_retain(tv).unwrap_or_default(),
            volume,
            open_interest: oi,
            bid_iv,
            mid_iv,
            ask_iv,
            smv_vol: smv,
            greeks: Greeks {
                delta,
                gamma,
                theta,
                vega,
                rho,
            },
            residual_rate: 0.0, // Not stored in normalized format
        });
    }

    // Convert HashMap to sorted Vec
    let mut chains: Vec<_> = chains_map.into_values().collect();
    chains.sort_by_key(|c| c.expiration);
    snapshot.chains = chains;

    Ok(snapshot)
}

/// Iterator over daily snapshots for efficient memory usage.
pub struct SnapshotIterator {
    loader: DataLoader,
    ticker: String,
    dates: Vec<NaiveDate>,
    current_idx: usize,
    df_cache: Option<DataFrame>,
}

impl SnapshotIterator {
    /// Create a new snapshot iterator.
    pub fn new(
        loader: DataLoader,
        ticker: String,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Self, LoaderError> {
        let df = loader.load_date_range(&ticker, start_date, end_date)?;
        let dates = loader.get_trading_dates(&df)?;

        Ok(Self {
            loader,
            ticker,
            dates,
            current_idx: 0,
            df_cache: Some(df),
        })
    }

    /// Get the total number of trading days.
    pub fn len(&self) -> usize {
        self.dates.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }

    /// Get the list of trading dates.
    pub fn dates(&self) -> &[NaiveDate] {
        &self.dates
    }
}

impl Iterator for SnapshotIterator {
    type Item = Result<OptionsSnapshot, LoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dates.len() {
            return None;
        }

        let date = self.dates[self.current_idx];
        self.current_idx += 1;

        if let Some(ref df) = self.df_cache {
            let day_df = df
                .clone()
                .lazy()
                .filter(col("trade_date").eq(lit(date.to_string())))
                .collect();

            match day_df {
                Ok(day_df) if day_df.height() > 0 => {
                    Some(dataframe_to_snapshot(day_df, &self.ticker, date))
                }
                Ok(_) => self.next(),
                Err(e) => Some(Err(LoaderError::Polars(e))),
            }
        } else {
            Some(self.loader.load_snapshot(&self.ticker, date))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_from_days() {
        let date = date_from_days(18262);
        assert_eq!(date, NaiveDate::from_ymd_opt(2020, 1, 1).unwrap());
    }

    #[test]
    fn test_loader_creation() {
        let loader = DataLoader::new("data/orats");
        assert_eq!(loader.data_dir, "data/orats");
    }

    #[test]
    fn test_parquet_path() {
        let loader = DataLoader::new("data/orats");
        let path = loader.parquet_path("SPY", 2020);
        assert_eq!(path, "data/orats/strikes/SPY/SPY_2020.parquet");
    }

    #[test]
    fn test_expected_columns() {
        assert_eq!(EXPECTED_COLUMNS.len(), 21);
        assert!(EXPECTED_COLUMNS.contains(&"ticker"));
        assert!(EXPECTED_COLUMNS.contains(&"delta"));
        assert!(EXPECTED_COLUMNS.contains(&"option_type"));
    }
}
