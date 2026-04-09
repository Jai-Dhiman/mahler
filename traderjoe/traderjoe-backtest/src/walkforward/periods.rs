//! Walk-forward period generation.
//!
//! Generates train/validate/test periods for rolling optimization.

use chrono::{Datelike, Duration, NaiveDate};
use serde::{Deserialize, Serialize};

/// A single walk-forward period with train/validate/test splits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardPeriod {
    /// Period number (1-indexed).
    pub period_num: usize,
    /// Training start date.
    pub train_start: NaiveDate,
    /// Training end date.
    pub train_end: NaiveDate,
    /// Validation start date.
    pub validate_start: NaiveDate,
    /// Validation end date.
    pub validate_end: NaiveDate,
    /// Test start date.
    pub test_start: NaiveDate,
    /// Test end date.
    pub test_end: NaiveDate,
}

impl WalkForwardPeriod {
    /// Get training period length in days.
    pub fn train_days(&self) -> i64 {
        (self.train_end - self.train_start).num_days()
    }

    /// Get validation period length in days.
    pub fn validate_days(&self) -> i64 {
        (self.validate_end - self.validate_start).num_days()
    }

    /// Get test period length in days.
    pub fn test_days(&self) -> i64 {
        (self.test_end - self.test_start).num_days()
    }

    /// Get total period length in days.
    pub fn total_days(&self) -> i64 {
        (self.test_end - self.train_start).num_days()
    }
}

/// Configuration for walk-forward periods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardPeriodsConfig {
    /// Training period length in months.
    pub train_months: u32,
    /// Validation period length in months.
    pub validate_months: u32,
    /// Test period length in months.
    pub test_months: u32,
    /// Roll forward by this many months.
    pub roll_months: u32,
}

impl Default for WalkForwardPeriodsConfig {
    fn default() -> Self {
        Self {
            train_months: 6,
            validate_months: 1,
            test_months: 1,
            roll_months: 1,
        }
    }
}

/// Generator for walk-forward periods.
pub struct WalkForwardPeriods {
    config: WalkForwardPeriodsConfig,
    start_date: NaiveDate,
    end_date: NaiveDate,
}

impl WalkForwardPeriods {
    /// Create a new period generator.
    pub fn new(config: WalkForwardPeriodsConfig, start_date: NaiveDate, end_date: NaiveDate) -> Self {
        Self {
            config,
            start_date,
            end_date,
        }
    }

    /// Generate all walk-forward periods.
    pub fn generate(&self) -> Vec<WalkForwardPeriod> {
        let mut periods = Vec::new();
        let mut period_num = 1;

        let mut train_start = self.start_date;

        loop {
            // Calculate period boundaries
            let train_end = add_months(train_start, self.config.train_months as i32);
            let validate_start = train_end + Duration::days(1);
            let validate_end = add_months(validate_start, self.config.validate_months as i32);
            let test_start = validate_end + Duration::days(1);
            let test_end = add_months(test_start, self.config.test_months as i32);

            // Check if we have enough data
            if test_end > self.end_date {
                break;
            }

            periods.push(WalkForwardPeriod {
                period_num,
                train_start,
                train_end,
                validate_start,
                validate_end,
                test_start,
                test_end,
            });

            period_num += 1;

            // Roll forward
            train_start = add_months(train_start, self.config.roll_months as i32);

            // Prevent infinite loops
            if period_num > 1000 {
                break;
            }
        }

        periods
    }

    /// Get expected number of periods.
    pub fn expected_periods(&self) -> usize {
        let total_months = months_between(self.start_date, self.end_date);
        let required_months = self.config.train_months + self.config.validate_months + self.config.test_months;

        if total_months < required_months {
            return 0;
        }

        let available_months = total_months - required_months;
        (available_months / self.config.roll_months + 1) as usize
    }
}

/// Add months to a date.
fn add_months(date: NaiveDate, months: i32) -> NaiveDate {
    let mut year = date.year();
    let mut month = date.month() as i32 + months;

    while month > 12 {
        year += 1;
        month -= 12;
    }
    while month < 1 {
        year -= 1;
        month += 12;
    }

    // Handle day overflow (e.g., Jan 31 + 1 month)
    let day = date.day().min(days_in_month(year, month as u32));
    NaiveDate::from_ymd_opt(year, month as u32, day).unwrap_or(date)
}

/// Get the number of days in a month.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// Check if a year is a leap year.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Calculate months between two dates.
fn months_between(start: NaiveDate, end: NaiveDate) -> u32 {
    let years = (end.year() - start.year()) as i32;
    let months_diff = end.month() as i32 - start.month() as i32;
    (years * 12 + months_diff).max(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WalkForwardPeriodsConfig::default();
        assert_eq!(config.train_months, 6);
        assert_eq!(config.validate_months, 1);
        assert_eq!(config.test_months, 1);
        assert_eq!(config.roll_months, 1);
    }

    #[test]
    fn test_add_months() {
        let date = NaiveDate::from_ymd_opt(2020, 1, 15).unwrap();
        let result = add_months(date, 6);
        assert_eq!(result, NaiveDate::from_ymd_opt(2020, 7, 15).unwrap());
    }

    #[test]
    fn test_add_months_year_rollover() {
        let date = NaiveDate::from_ymd_opt(2020, 11, 15).unwrap();
        let result = add_months(date, 3);
        assert_eq!(result, NaiveDate::from_ymd_opt(2021, 2, 15).unwrap());
    }

    #[test]
    fn test_period_generation() {
        let config = WalkForwardPeriodsConfig::default();
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2021, 12, 31).unwrap();

        let generator = WalkForwardPeriods::new(config, start, end);
        let periods = generator.generate();

        // 24 months - 8 required = 16 available, 16 / 1 + 1 = 17 periods
        assert!(periods.len() > 10);

        // Check first period
        let first = &periods[0];
        assert_eq!(first.train_start, start);
    }

    #[test]
    fn test_period_days() {
        let period = WalkForwardPeriod {
            period_num: 1,
            train_start: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            train_end: NaiveDate::from_ymd_opt(2020, 6, 30).unwrap(),
            validate_start: NaiveDate::from_ymd_opt(2020, 7, 1).unwrap(),
            validate_end: NaiveDate::from_ymd_opt(2020, 7, 31).unwrap(),
            test_start: NaiveDate::from_ymd_opt(2020, 8, 1).unwrap(),
            test_end: NaiveDate::from_ymd_opt(2020, 8, 31).unwrap(),
        };

        assert!(period.train_days() > 150);
        assert!(period.validate_days() >= 28);
        assert!(period.test_days() >= 28);
    }

    #[test]
    fn test_months_between() {
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2021, 1, 1).unwrap();
        assert_eq!(months_between(start, end), 12);
    }
}
