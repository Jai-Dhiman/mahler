//! Circuit breakers for risk management.
//!
//! Hard-coded safety rules that cannot be overridden:
//! - Daily loss: 2% -> HALT, close all positions
//! - Weekly loss: 5% -> HALT, close all positions
//! - Max drawdown: 15% -> HALT, disable trading
//! - VIX halt: VIX > 50 -> No new trades

use chrono::{Datelike, NaiveDate};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Circuit breaker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Daily loss limit as percentage of equity.
    pub daily_loss_pct: f64,
    /// Weekly loss limit as percentage of equity.
    pub weekly_loss_pct: f64,
    /// Maximum drawdown as percentage of peak equity.
    pub max_drawdown_pct: f64,
    /// VIX level that halts new trades.
    pub vix_halt_level: f64,
    /// Whether to auto-close positions on halt.
    pub auto_close_on_halt: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            daily_loss_pct: 2.0,
            weekly_loss_pct: 5.0,
            max_drawdown_pct: 15.0,
            vix_halt_level: 50.0,
            auto_close_on_halt: true,
        }
    }
}

/// Circuit breaker status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerStatus {
    /// Normal operation, trading allowed.
    Normal,
    /// Halted due to daily loss limit.
    HaltedDailyLoss,
    /// Halted due to weekly loss limit.
    HaltedWeeklyLoss,
    /// Halted due to max drawdown.
    HaltedDrawdown,
    /// Halted due to high VIX.
    HaltedVix,
    /// Manually halted.
    HaltedManual,
}

impl CircuitBreakerStatus {
    /// Check if trading is allowed.
    pub fn is_trading_allowed(&self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Check if new positions are allowed.
    pub fn allows_new_positions(&self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Get reason string.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::Normal => "Normal operation",
            Self::HaltedDailyLoss => "Daily loss limit exceeded",
            Self::HaltedWeeklyLoss => "Weekly loss limit exceeded",
            Self::HaltedDrawdown => "Maximum drawdown exceeded",
            Self::HaltedVix => "VIX above halt level",
            Self::HaltedManual => "Manual halt",
        }
    }
}

/// Daily P&L tracking for circuit breakers.
#[derive(Debug, Clone, Default)]
pub struct DailyPnL {
    pub date: Option<NaiveDate>,
    pub starting_equity: Decimal,
    pub current_equity: Decimal,
    pub pnl: Decimal,
    pub pnl_pct: f64,
}

/// Weekly P&L tracking for circuit breakers.
#[derive(Debug, Clone, Default)]
pub struct WeeklyPnL {
    pub week_start: Option<NaiveDate>,
    pub starting_equity: Decimal,
    pub current_equity: Decimal,
    pub pnl: Decimal,
    pub pnl_pct: f64,
}

/// Circuit breaker state and checks.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    status: CircuitBreakerStatus,
    daily_pnl: DailyPnL,
    weekly_pnl: WeeklyPnL,
    peak_equity: Decimal,
    current_vix: f64,
    halt_history: Vec<(NaiveDate, CircuitBreakerStatus)>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(config: CircuitBreakerConfig, initial_equity: Decimal) -> Self {
        Self {
            config,
            status: CircuitBreakerStatus::Normal,
            daily_pnl: DailyPnL {
                starting_equity: initial_equity,
                current_equity: initial_equity,
                ..Default::default()
            },
            weekly_pnl: WeeklyPnL {
                starting_equity: initial_equity,
                current_equity: initial_equity,
                ..Default::default()
            },
            peak_equity: initial_equity,
            current_vix: 0.0,
            halt_history: Vec::new(),
        }
    }

    /// Update circuit breaker state with new data.
    pub fn update(
        &mut self,
        date: NaiveDate,
        current_equity: Decimal,
        vix: Option<f64>,
    ) -> CircuitBreakerStatus {
        // Check for day/week change
        self.check_period_change(date, current_equity);

        // Update VIX
        if let Some(v) = vix {
            self.current_vix = v;
        }

        // Update equity tracking
        self.daily_pnl.current_equity = current_equity;
        self.weekly_pnl.current_equity = current_equity;

        // Update peak equity
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }

        // Calculate P&L percentages
        self.daily_pnl.pnl = current_equity - self.daily_pnl.starting_equity;
        let daily_start: f64 = self.daily_pnl.starting_equity.try_into().unwrap_or(1.0);
        self.daily_pnl.pnl_pct = if daily_start > 0.0 {
            let pnl: f64 = self.daily_pnl.pnl.try_into().unwrap_or(0.0);
            pnl / daily_start * 100.0
        } else {
            0.0
        };

        self.weekly_pnl.pnl = current_equity - self.weekly_pnl.starting_equity;
        let weekly_start: f64 = self.weekly_pnl.starting_equity.try_into().unwrap_or(1.0);
        self.weekly_pnl.pnl_pct = if weekly_start > 0.0 {
            let pnl: f64 = self.weekly_pnl.pnl.try_into().unwrap_or(0.0);
            pnl / weekly_start * 100.0
        } else {
            0.0
        };

        // Check circuit breakers (priority order)
        let new_status = self.check_breakers(current_equity);

        // Record halt if status changed
        if new_status != self.status && new_status != CircuitBreakerStatus::Normal {
            self.halt_history.push((date, new_status));
        }

        self.status = new_status;
        self.status
    }

    /// Check if day or week has changed.
    fn check_period_change(&mut self, date: NaiveDate, current_equity: Decimal) {
        // Check day change
        if self.daily_pnl.date != Some(date) {
            // New day - reset daily tracking
            self.daily_pnl = DailyPnL {
                date: Some(date),
                starting_equity: current_equity,
                current_equity,
                pnl: Decimal::ZERO,
                pnl_pct: 0.0,
            };

            // If normal, potentially reset status
            if self.status == CircuitBreakerStatus::HaltedDailyLoss {
                self.status = CircuitBreakerStatus::Normal;
            }
        }

        // Check week change (Monday)
        let is_new_week = self.weekly_pnl.week_start.map_or(true, |start| {
            let days_diff = (date - start).num_days();
            days_diff >= 7 || (days_diff > 0 && date.weekday() == chrono::Weekday::Mon)
        });

        if is_new_week {
            self.weekly_pnl = WeeklyPnL {
                week_start: Some(date),
                starting_equity: current_equity,
                current_equity,
                pnl: Decimal::ZERO,
                pnl_pct: 0.0,
            };

            // If normal, potentially reset status
            if self.status == CircuitBreakerStatus::HaltedWeeklyLoss {
                self.status = CircuitBreakerStatus::Normal;
            }
        }
    }

    /// Check all circuit breakers and return status.
    fn check_breakers(&self, current_equity: Decimal) -> CircuitBreakerStatus {
        // 1. VIX halt (highest priority for new trades)
        if self.current_vix >= self.config.vix_halt_level {
            return CircuitBreakerStatus::HaltedVix;
        }

        // 2. Drawdown halt (non-recoverable during session)
        let peak: f64 = self.peak_equity.try_into().unwrap_or(1.0);
        let current: f64 = current_equity.try_into().unwrap_or(peak);
        let drawdown_pct = if peak > 0.0 {
            (peak - current) / peak * 100.0
        } else {
            0.0
        };

        if drawdown_pct >= self.config.max_drawdown_pct {
            return CircuitBreakerStatus::HaltedDrawdown;
        }

        // 3. Weekly loss halt
        if self.weekly_pnl.pnl_pct <= -self.config.weekly_loss_pct {
            return CircuitBreakerStatus::HaltedWeeklyLoss;
        }

        // 4. Daily loss halt
        if self.daily_pnl.pnl_pct <= -self.config.daily_loss_pct {
            return CircuitBreakerStatus::HaltedDailyLoss;
        }

        CircuitBreakerStatus::Normal
    }

    /// Get current status.
    pub fn status(&self) -> CircuitBreakerStatus {
        self.status
    }

    /// Check if trading is allowed.
    pub fn is_trading_allowed(&self) -> bool {
        self.status.is_trading_allowed()
    }

    /// Check if new positions are allowed.
    pub fn allows_new_positions(&self) -> bool {
        self.status.allows_new_positions()
    }

    /// Get current drawdown percentage.
    pub fn current_drawdown_pct(&self) -> f64 {
        let peak: f64 = self.peak_equity.try_into().unwrap_or(1.0);
        let current: f64 = self
            .daily_pnl
            .current_equity
            .try_into()
            .unwrap_or(peak);
        if peak > 0.0 {
            (peak - current) / peak * 100.0
        } else {
            0.0
        }
    }

    /// Get daily P&L percentage.
    pub fn daily_pnl_pct(&self) -> f64 {
        self.daily_pnl.pnl_pct
    }

    /// Get weekly P&L percentage.
    pub fn weekly_pnl_pct(&self) -> f64 {
        self.weekly_pnl.pnl_pct
    }

    /// Get halt history.
    pub fn halt_history(&self) -> &[(NaiveDate, CircuitBreakerStatus)] {
        &self.halt_history
    }

    /// Manually halt trading.
    pub fn manual_halt(&mut self) {
        self.status = CircuitBreakerStatus::HaltedManual;
    }

    /// Reset to normal (use with caution).
    pub fn reset(&mut self) {
        self.status = CircuitBreakerStatus::Normal;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_config() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.daily_loss_pct, 2.0);
        assert_eq!(config.weekly_loss_pct, 5.0);
        assert_eq!(config.max_drawdown_pct, 15.0);
        assert_eq!(config.vix_halt_level, 50.0);
    }

    #[test]
    fn test_normal_operation() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default(), dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();

        let status = cb.update(date, dec!(100_500), Some(20.0));
        assert_eq!(status, CircuitBreakerStatus::Normal);
        assert!(cb.is_trading_allowed());
    }

    #[test]
    fn test_daily_loss_halt() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default(), dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();

        // First update sets the day
        cb.update(date, dec!(100_000), Some(20.0));

        // 2.5% loss should trigger halt
        let status = cb.update(date, dec!(97_500), Some(20.0));
        assert_eq!(status, CircuitBreakerStatus::HaltedDailyLoss);
        assert!(!cb.is_trading_allowed());
    }

    #[test]
    fn test_vix_halt() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default(), dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();

        // VIX at 55 should trigger halt
        let status = cb.update(date, dec!(100_000), Some(55.0));
        assert_eq!(status, CircuitBreakerStatus::HaltedVix);
        assert!(!cb.allows_new_positions());
    }

    #[test]
    fn test_drawdown_halt() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default(), dec!(100_000));

        // Build up to peak
        let date1 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        cb.update(date1, dec!(110_000), Some(20.0));

        // Drawdown of 16.4% (110K -> 92K)
        let date2 = NaiveDate::from_ymd_opt(2024, 1, 16).unwrap();
        let status = cb.update(date2, dec!(92_000), Some(20.0));
        assert_eq!(status, CircuitBreakerStatus::HaltedDrawdown);
    }

    #[test]
    fn test_status_reasons() {
        assert_eq!(
            CircuitBreakerStatus::Normal.reason(),
            "Normal operation"
        );
        assert_eq!(
            CircuitBreakerStatus::HaltedDailyLoss.reason(),
            "Daily loss limit exceeded"
        );
        assert_eq!(
            CircuitBreakerStatus::HaltedVix.reason(),
            "VIX above halt level"
        );
    }

    #[test]
    fn test_day_reset() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig::default(), dec!(100_000));

        let date1 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        cb.update(date1, dec!(100_000), Some(20.0));
        cb.update(date1, dec!(97_500), Some(20.0)); // Halt

        assert_eq!(cb.status(), CircuitBreakerStatus::HaltedDailyLoss);

        // Next day should reset
        let date2 = NaiveDate::from_ymd_opt(2024, 1, 16).unwrap();
        let status = cb.update(date2, dec!(97_500), Some(20.0));
        assert_eq!(status, CircuitBreakerStatus::Normal);
    }
}
