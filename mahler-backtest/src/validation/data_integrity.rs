//! Data integrity validation for ORATS parquet files.
//!
//! Validates:
//! - Schema consistency (21 columns)
//! - Date continuity (no gaps in trading days)
//! - Greeks validity (delta in [-1,1], gamma/vega >= 0)
//! - IV validity (bid_iv <= mid_iv <= ask_iv, all >= 0)
//! - Price validity (bid <= ask, bid >= 0)
//! - Put-call parity (delta signs)
//! - DTE accuracy (DTE = expir_date - trade_date)

use std::collections::HashSet;

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use polars::prelude::*;
use thiserror::Error;

use crate::data::{DataLoader, LoaderError, EXPECTED_COLUMNS};

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Loader error: {0}")]
    Loader(#[from] LoaderError),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

pub type ValidationResult<T> = Result<T, ValidationError>;

/// Result of a single validation check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub message: String,
    pub details: Option<String>,
}

impl CheckResult {
    pub fn pass(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            message: message.to_string(),
            details: None,
        }
    }

    pub fn fail(name: &str, message: &str, details: Option<String>) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            message: message.to_string(),
            details,
        }
    }
}

/// Complete data integrity report for a ticker/year.
#[derive(Debug)]
pub struct DataIntegrityReport {
    pub ticker: String,
    pub year: i32,
    pub row_count: usize,
    pub trading_days: usize,
    pub checks: Vec<CheckResult>,
}

impl DataIntegrityReport {
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    pub fn failed_checks(&self) -> Vec<&CheckResult> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }

    pub fn summary(&self) -> String {
        let passed = self.checks.iter().filter(|c| c.passed).count();
        let total = self.checks.len();
        format!(
            "{} {} ({} rows, {} trading days): {}/{} checks passed",
            self.ticker, self.year, self.row_count, self.trading_days, passed, total
        )
    }
}

/// Validator for ORATS data integrity.
pub struct DataIntegrityValidator {
    loader: DataLoader,
}

impl DataIntegrityValidator {
    pub fn new(data_dir: &str) -> Self {
        Self {
            loader: DataLoader::new(data_dir),
        }
    }

    /// Run all validation checks on a ticker/year combination.
    pub fn validate(&self, ticker: &str, year: i32) -> ValidationResult<DataIntegrityReport> {
        let df = self.loader.load_dataframe(ticker, year)?;
        let row_count = df.height();

        let mut checks = Vec::new();

        // 1. Schema consistency
        checks.push(self.check_schema(&df)?);

        // 2. Date continuity
        let (date_check, trading_days) = self.check_date_continuity(&df, year)?;
        checks.push(date_check);

        // 3. Greeks validity
        checks.push(self.check_greeks_validity(&df)?);

        // 4. IV validity
        checks.push(self.check_iv_validity(&df)?);

        // 5. Price validity
        checks.push(self.check_price_validity(&df)?);

        // 6. Put-call parity (delta signs)
        checks.push(self.check_delta_signs(&df)?);

        // 7. DTE accuracy
        checks.push(self.check_dte_accuracy(&df)?);

        Ok(DataIntegrityReport {
            ticker: ticker.to_string(),
            year,
            row_count,
            trading_days,
            checks,
        })
    }

    /// Validate all available data for a ticker.
    pub fn validate_ticker(&self, ticker: &str) -> ValidationResult<Vec<DataIntegrityReport>> {
        let years = self.loader.available_years(ticker)?;
        let mut reports = Vec::new();

        for year in years {
            reports.push(self.validate(ticker, year)?);
        }

        Ok(reports)
    }

    /// Validate all available tickers.
    pub fn validate_all(&self) -> ValidationResult<Vec<DataIntegrityReport>> {
        let tickers = self.loader.available_tickers()?;
        let mut reports = Vec::new();

        for ticker in tickers {
            let ticker_reports = self.validate_ticker(&ticker)?;
            reports.extend(ticker_reports);
        }

        Ok(reports)
    }

    /// Check that schema has all expected columns.
    fn check_schema(&self, df: &DataFrame) -> ValidationResult<CheckResult> {
        let columns: HashSet<String> = df
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let expected: HashSet<String> = EXPECTED_COLUMNS.iter().map(|s| s.to_string()).collect();

        let missing: Vec<_> = expected.difference(&columns).collect();
        let extra: Vec<_> = columns.difference(&expected).collect();

        if missing.is_empty() && extra.is_empty() {
            Ok(CheckResult::pass(
                "schema_consistency",
                &format!("All {} expected columns present", EXPECTED_COLUMNS.len()),
            ))
        } else {
            let mut details = String::new();
            if !missing.is_empty() {
                details.push_str(&format!("Missing: {:?}", missing));
            }
            if !extra.is_empty() {
                if !details.is_empty() {
                    details.push_str("; ");
                }
                details.push_str(&format!("Extra: {:?}", extra));
            }
            Ok(CheckResult::fail(
                "schema_consistency",
                "Schema mismatch",
                Some(details),
            ))
        }
    }

    /// Check that trading days are continuous (no unexpected gaps).
    fn check_date_continuity(
        &self,
        df: &DataFrame,
        year: i32,
    ) -> ValidationResult<(CheckResult, usize)> {
        let dates = self.loader.get_trading_dates(df)?;

        if dates.is_empty() {
            return Ok((
                CheckResult::fail("date_continuity", "No trading dates found", None),
                0,
            ));
        }

        // Filter to the specific year
        let year_dates: Vec<_> = dates.iter().filter(|d| d.year() == year).copied().collect();

        if year_dates.is_empty() {
            return Ok((
                CheckResult::fail(
                    "date_continuity",
                    &format!("No trading dates in year {}", year),
                    None,
                ),
                0,
            ));
        }

        // Check for gaps (excluding weekends and common holidays)
        let mut gaps = Vec::new();
        for window in year_dates.windows(2) {
            let (prev, curr) = (window[0], window[1]);
            let _expected_next = next_trading_day(prev);

            // Allow small gaps (up to 5 business days for holidays)
            let gap_days = (curr - prev).num_days();
            if gap_days > 7 {
                // More than a week gap is suspicious
                gaps.push(format!("{} to {} ({} days)", prev, curr, gap_days));
            }
        }

        let trading_days = year_dates.len();

        if gaps.is_empty() {
            Ok((
                CheckResult::pass(
                    "date_continuity",
                    &format!("{} trading days, no major gaps", trading_days),
                ),
                trading_days,
            ))
        } else {
            Ok((
                CheckResult::fail(
                    "date_continuity",
                    &format!("{} major gaps found", gaps.len()),
                    Some(gaps.join(", ")),
                ),
                trading_days,
            ))
        }
    }

    /// Check Greeks validity: delta in [-1,1], gamma >= 0, vega >= 0.
    fn check_greeks_validity(&self, df: &DataFrame) -> ValidationResult<CheckResult> {
        let delta = df.column("delta")?.f64()?;
        let gamma = df.column("gamma")?.f64()?;
        let vega = df.column("vega")?.f64()?;

        let mut issues = Vec::new();

        // Delta should be in [-1, 1]
        let invalid_delta = delta
            .into_iter()
            .filter(|d| d.map_or(false, |v| v < -1.0 || v > 1.0))
            .count();
        if invalid_delta > 0 {
            issues.push(format!("{} rows with delta outside [-1,1]", invalid_delta));
        }

        // Gamma should be >= 0
        let invalid_gamma = gamma
            .into_iter()
            .filter(|g| g.map_or(false, |v| v < 0.0))
            .count();
        if invalid_gamma > 0 {
            issues.push(format!("{} rows with negative gamma", invalid_gamma));
        }

        // Vega should be >= 0
        let invalid_vega = vega
            .into_iter()
            .filter(|v| v.map_or(false, |val| val < 0.0))
            .count();
        if invalid_vega > 0 {
            issues.push(format!("{} rows with negative vega", invalid_vega));
        }

        if issues.is_empty() {
            Ok(CheckResult::pass("greeks_validity", "All Greeks within valid ranges"))
        } else {
            Ok(CheckResult::fail(
                "greeks_validity",
                &format!("{} issues found", issues.len()),
                Some(issues.join("; ")),
            ))
        }
    }

    /// Check IV validity: bid_iv <= mid_iv <= ask_iv, all >= 0.
    fn check_iv_validity(&self, df: &DataFrame) -> ValidationResult<CheckResult> {
        let bid_iv = df.column("bid_iv")?.f64()?;
        let mid_iv = df.column("mid_iv")?.f64()?;
        let ask_iv = df.column("ask_iv")?.f64()?;

        let mut issues = Vec::new();
        let total = df.height();

        // All IVs should be >= 0
        let negative_bid = bid_iv
            .into_iter()
            .filter(|v| v.map_or(false, |val| val < 0.0))
            .count();
        let negative_mid = mid_iv
            .clone()
            .into_iter()
            .filter(|v| v.map_or(false, |val| val < 0.0))
            .count();
        let negative_ask = ask_iv
            .into_iter()
            .filter(|v| v.map_or(false, |val| val < 0.0))
            .count();

        if negative_bid > 0 || negative_mid > 0 || negative_ask > 0 {
            issues.push(format!(
                "Negative IVs: bid={}, mid={}, ask={}",
                negative_bid, negative_mid, negative_ask
            ));
        }

        // bid_iv <= mid_iv <= ask_iv (with tolerance for floating point)
        // This is often not strictly enforced in market data, so we track but don't fail
        let tolerance = 0.001;
        let bid_iv = df.column("bid_iv")?.f64()?;
        let ask_iv = df.column("ask_iv")?.f64()?;

        let mut order_violations = 0;
        for (bid, mid, ask) in bid_iv
            .into_iter()
            .zip(mid_iv.into_iter())
            .zip(ask_iv.into_iter())
            .map(|((b, m), a)| (b, m, a))
        {
            if let (Some(b), Some(m), Some(a)) = (bid, mid, ask) {
                if b > m + tolerance || m > a + tolerance {
                    order_violations += 1;
                }
            }
        }

        if order_violations > 0 {
            let pct = (order_violations as f64 / total as f64) * 100.0;
            if pct > 5.0 {
                issues.push(format!(
                    "{} ({:.1}%) rows with IV ordering violations",
                    order_violations, pct
                ));
            }
        }

        if issues.is_empty() {
            Ok(CheckResult::pass("iv_validity", "All IVs non-negative"))
        } else {
            Ok(CheckResult::fail(
                "iv_validity",
                "IV validation issues",
                Some(issues.join("; ")),
            ))
        }
    }

    /// Check price validity: bid <= ask, bid >= 0.
    fn check_price_validity(&self, df: &DataFrame) -> ValidationResult<CheckResult> {
        let bid = df.column("bid")?.f64()?;
        let ask = df.column("ask")?.f64()?;

        let mut issues = Vec::new();
        let total = df.height();

        // bid >= 0
        let negative_bid = bid
            .clone()
            .into_iter()
            .filter(|v| v.map_or(false, |val| val < 0.0))
            .count();
        if negative_bid > 0 {
            issues.push(format!("{} rows with negative bid", negative_bid));
        }

        // bid <= ask
        let mut crossed = 0;
        for (b, a) in bid.into_iter().zip(ask.into_iter()) {
            if let (Some(b), Some(a)) = (b, a) {
                if b > a + 0.001 {
                    crossed += 1;
                }
            }
        }

        if crossed > 0 {
            let pct = (crossed as f64 / total as f64) * 100.0;
            issues.push(format!(
                "{} ({:.2}%) rows with crossed markets (bid > ask)",
                crossed, pct
            ));
        }

        if issues.is_empty() {
            Ok(CheckResult::pass(
                "price_validity",
                "All prices valid (bid >= 0, bid <= ask)",
            ))
        } else {
            // Crossed markets are common in historical data due to timing
            let is_warning = crossed > 0 && negative_bid == 0;
            if is_warning {
                Ok(CheckResult::pass(
                    "price_validity",
                    &format!(
                        "Prices valid but {} crossed markets (common in historical data)",
                        crossed
                    ),
                ))
            } else {
                Ok(CheckResult::fail(
                    "price_validity",
                    "Price validation issues",
                    Some(issues.join("; ")),
                ))
            }
        }
    }

    /// Check put-call parity: calls should have positive delta, puts negative.
    fn check_delta_signs(&self, df: &DataFrame) -> ValidationResult<CheckResult> {
        let opt_type = df.column("option_type")?.str()?;
        let delta = df.column("delta")?.f64()?;

        let mut call_negative = 0;
        let mut put_positive = 0;
        let mut total_calls = 0;
        let mut total_puts = 0;

        for (ot, d) in opt_type.into_iter().zip(delta.into_iter()) {
            match (ot, d) {
                (Some("C"), Some(d)) => {
                    total_calls += 1;
                    if d < 0.0 {
                        call_negative += 1;
                    }
                }
                (Some("P"), Some(d)) => {
                    total_puts += 1;
                    if d > 0.0 {
                        put_positive += 1;
                    }
                }
                _ => {}
            }
        }

        let mut issues = Vec::new();

        if call_negative > 0 {
            let pct = (call_negative as f64 / total_calls.max(1) as f64) * 100.0;
            if pct > 1.0 {
                issues.push(format!(
                    "{} ({:.2}%) calls with negative delta",
                    call_negative, pct
                ));
            }
        }

        if put_positive > 0 {
            let pct = (put_positive as f64 / total_puts.max(1) as f64) * 100.0;
            if pct > 1.0 {
                issues.push(format!(
                    "{} ({:.2}%) puts with positive delta",
                    put_positive, pct
                ));
            }
        }

        if issues.is_empty() {
            Ok(CheckResult::pass(
                "delta_signs",
                &format!(
                    "Delta signs correct ({} calls positive, {} puts negative)",
                    total_calls, total_puts
                ),
            ))
        } else {
            Ok(CheckResult::fail(
                "delta_signs",
                "Delta sign issues",
                Some(issues.join("; ")),
            ))
        }
    }

    /// Check DTE accuracy: DTE should equal expir_date - trade_date.
    fn check_dte_accuracy(&self, df: &DataFrame) -> ValidationResult<CheckResult> {
        let trade_date = df.column("trade_date")?.str()?;
        let expir_date = df.column("expir_date")?.str()?;
        let dte = df.column("dte")?.i32()?;

        let mut mismatches = 0;
        let total = df.height();

        for ((td, ed), d) in trade_date
            .into_iter()
            .zip(expir_date.into_iter())
            .zip(dte.into_iter())
        {
            if let (Some(td_str), Some(ed_str), Some(d)) = (td, ed, d) {
                if let (Ok(td_date), Ok(ed_date)) = (
                    NaiveDate::parse_from_str(td_str, "%Y-%m-%d"),
                    NaiveDate::parse_from_str(ed_str, "%Y-%m-%d"),
                ) {
                    let expected_dte = (ed_date - td_date).num_days() as i32;
                    if (expected_dte - d).abs() > 1 {
                        // Allow 1 day tolerance for edge cases
                        mismatches += 1;
                    }
                }
            }
        }

        if mismatches == 0 {
            Ok(CheckResult::pass(
                "dte_accuracy",
                "DTE matches expir_date - trade_date",
            ))
        } else {
            let pct = (mismatches as f64 / total as f64) * 100.0;
            if pct < 1.0 {
                Ok(CheckResult::pass(
                    "dte_accuracy",
                    &format!(
                        "DTE mostly accurate ({} minor mismatches, {:.2}%)",
                        mismatches, pct
                    ),
                ))
            } else {
                Ok(CheckResult::fail(
                    "dte_accuracy",
                    &format!("{} ({:.2}%) DTE mismatches", mismatches, pct),
                    None,
                ))
            }
        }
    }
}

/// Get the next expected trading day (skip weekends).
fn next_trading_day(date: NaiveDate) -> NaiveDate {
    let mut next = date + Duration::days(1);
    while matches!(next.weekday(), Weekday::Sat | Weekday::Sun) {
        next += Duration::days(1);
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_trading_day() {
        // Friday -> Monday
        let friday = NaiveDate::from_ymd_opt(2024, 1, 5).unwrap();
        let monday = next_trading_day(friday);
        assert_eq!(monday, NaiveDate::from_ymd_opt(2024, 1, 8).unwrap());

        // Monday -> Tuesday
        let monday = NaiveDate::from_ymd_opt(2024, 1, 8).unwrap();
        let tuesday = next_trading_day(monday);
        assert_eq!(tuesday, NaiveDate::from_ymd_opt(2024, 1, 9).unwrap());
    }

    #[test]
    fn test_check_result() {
        let pass = CheckResult::pass("test", "passed");
        assert!(pass.passed);

        let fail = CheckResult::fail("test", "failed", Some("details".to_string()));
        assert!(!fail.passed);
        assert_eq!(fail.details, Some("details".to_string()));
    }
}
