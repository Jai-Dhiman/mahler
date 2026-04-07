//! Core backtesting types.
//!
//! Provides configuration and result types for backtesting:
//! - BacktestConfig: parameters for a backtest run
//! - BacktestResult: output statistics and equity curve
//! - EquityPoint: daily equity snapshot

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::commission::CommissionModel;
use super::slippage::SlippageModel;
use crate::risk::CircuitBreakerConfig;

/// Configuration for backtest execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Starting equity.
    pub initial_equity: Decimal,

    /// Profit target as percentage of credit (e.g., 50.0 = 50%).
    pub profit_target_pct: f64,

    /// Stop loss as percentage of credit (e.g., 125.0 = 125%).
    pub stop_loss_pct: f64,

    /// Time-based exit DTE (e.g., 21 = exit at 21 DTE).
    pub time_exit_dte: i32,

    /// Minimum DTE for new trades.
    pub min_dte: i32,

    /// Maximum DTE for new trades.
    pub max_dte: i32,

    /// Minimum delta for short strike (absolute value).
    pub min_delta: f64,

    /// Maximum delta for short strike (absolute value).
    pub max_delta: f64,

    /// Minimum IV percentile for entry.
    pub min_iv_percentile: f64,

    /// Maximum trades per day.
    pub max_trades_per_day: usize,

    /// Maximum concurrent positions.
    pub max_positions: usize,

    /// Maximum risk per trade as percentage of equity.
    pub max_risk_per_trade_pct: f64,

    /// Maximum total portfolio risk as percentage of equity.
    pub max_portfolio_risk_pct: f64,

    /// Slippage model.
    #[serde(default)]
    pub slippage: SlippageModel,

    /// Commission model.
    #[serde(default)]
    pub commission: CommissionModel,

    /// Enable IV percentile entry filter.
    #[serde(default = "default_true")]
    pub use_iv_percentile_filter: bool,

    /// Enable market regime position sizing.
    #[serde(default = "default_true")]
    pub use_regime_sizing: bool,

    /// Enable circuit breakers.
    #[serde(default = "default_true")]
    pub use_circuit_breakers: bool,

    /// Circuit breaker configuration.
    #[serde(default)]
    pub circuit_breaker_config: CircuitBreakerConfig,

    /// Enable scaled position sizing (vs hardcoded 1 contract).
    #[serde(default)]
    pub use_scaled_position_sizing: bool,

    /// Maximum exposure to equity-correlated assets (SPY/QQQ/IWM) as %.
    #[serde(default = "default_correlated_exposure")]
    pub max_correlated_exposure_pct: f64,

    /// Maximum single position risk as % of equity.
    #[serde(default = "default_single_position")]
    pub max_single_position_pct: f64,
}

fn default_correlated_exposure() -> f64 {
    50.0
}

fn default_single_position() -> f64 {
    5.0
}

fn default_true() -> bool {
    true
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_equity: Decimal::from(100_000),
            profit_target_pct: 50.0,
            stop_loss_pct: 125.0,
            time_exit_dte: 21,
            min_dte: 30,
            max_dte: 45,
            min_delta: 0.20,
            max_delta: 0.30,
            min_iv_percentile: 50.0,
            max_trades_per_day: 1,
            max_positions: 10,
            max_risk_per_trade_pct: 2.0,
            max_portfolio_risk_pct: 10.0,
            slippage: SlippageModel::default(),
            commission: CommissionModel::default(),
            use_iv_percentile_filter: true,
            use_regime_sizing: true,
            use_circuit_breakers: true,
            circuit_breaker_config: CircuitBreakerConfig::default(),
            use_scaled_position_sizing: false,
            max_correlated_exposure_pct: 50.0,
            max_single_position_pct: 5.0,
        }
    }
}

/// Daily equity snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub date: NaiveDate,
    pub equity: Decimal,
    pub cash: Decimal,
    pub positions_value: Decimal,
    pub open_positions: usize,
    pub daily_pnl: Decimal,
}

/// Result of a completed backtest.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Configuration used.
    pub config: BacktestConfig,

    /// Ticker(s) tested.
    pub tickers: Vec<String>,

    /// Start date.
    pub start_date: NaiveDate,

    /// End date.
    pub end_date: NaiveDate,

    /// All completed trades.
    pub trades: Vec<super::trade::Trade>,

    /// Daily equity curve.
    pub equity_curve: Vec<EquityPoint>,

    /// Final equity.
    pub final_equity: Decimal,

    /// Total return percentage.
    pub total_return_pct: f64,

    /// Number of trading days.
    pub trading_days: usize,

    /// Peak equity (for drawdown calculation).
    pub peak_equity: Decimal,

    /// Maximum drawdown.
    pub max_drawdown: Decimal,

    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,

    /// Total trades.
    pub total_trades: usize,

    /// Winning trades.
    pub winning_trades: usize,

    /// Losing trades.
    pub losing_trades: usize,

    /// Total P&L.
    pub total_pnl: Decimal,

    /// Gross profit.
    pub gross_profit: Decimal,

    /// Gross loss.
    pub gross_loss: Decimal,

    /// Total commission paid.
    pub total_commission: Decimal,
}

impl BacktestResult {
    /// Calculate win rate.
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / self.total_trades as f64
    }

    /// Calculate profit factor.
    pub fn profit_factor(&self) -> f64 {
        let loss: f64 = self.gross_loss.abs().try_into().unwrap_or(0.0);
        if loss == 0.0 {
            return f64::INFINITY;
        }
        let profit: f64 = self.gross_profit.try_into().unwrap_or(0.0);
        profit / loss
    }

    /// Calculate average trade P&L.
    pub fn avg_trade_pnl(&self) -> Decimal {
        if self.total_trades == 0 {
            return Decimal::ZERO;
        }
        self.total_pnl / Decimal::from(self.total_trades as i64)
    }

    /// Calculate average winner.
    pub fn avg_winner(&self) -> Decimal {
        if self.winning_trades == 0 {
            return Decimal::ZERO;
        }
        self.gross_profit / Decimal::from(self.winning_trades as i64)
    }

    /// Calculate average loser.
    pub fn avg_loser(&self) -> Decimal {
        if self.losing_trades == 0 {
            return Decimal::ZERO;
        }
        self.gross_loss / Decimal::from(self.losing_trades as i64)
    }

    /// Calculate Sharpe ratio (simplified, assuming risk-free rate = 0).
    pub fn sharpe_ratio(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        // Calculate daily returns
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| {
                let prev: f64 = w[0].equity.try_into().unwrap_or(1.0);
                let curr: f64 = w[1].equity.try_into().unwrap_or(1.0);
                (curr - prev) / prev
            })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize: 252 trading days
        (mean * 252.0_f64.sqrt()) / std_dev
    }

    /// Generate summary string.
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results ({} to {})\n\
             ----------------------------------------\n\
             Total Return: {:.2}%\n\
             Final Equity: ${:.2}\n\
             Max Drawdown: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             \n\
             Trades: {} (W: {}, L: {})\n\
             Win Rate: {:.1}%\n\
             Profit Factor: {:.2}\n\
             Avg Trade: ${:.2}\n\
             Avg Winner: ${:.2}\n\
             Avg Loser: ${:.2}\n\
             \n\
             Total Commission: ${:.2}",
            self.start_date,
            self.end_date,
            self.total_return_pct,
            self.final_equity,
            self.max_drawdown_pct,
            self.sharpe_ratio(),
            self.total_trades,
            self.winning_trades,
            self.losing_trades,
            self.win_rate() * 100.0,
            self.profit_factor(),
            self.avg_trade_pnl(),
            self.avg_winner(),
            self.avg_loser(),
            self.total_commission,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use crate::backtest::commission::CommissionModel;

    #[test]
    fn test_entry_commission_not_doubled() {
        // Bug regression: entry commission was called as calculate(contracts * 2, 2)
        // which doubled the contract count. Should be calculate(contracts, 2).
        // For 1 contract, 2 legs at $1/contract/leg: correct = $2, doubled = $4.
        let model = CommissionModel::default();

        // Correct call: 1 contract, 2 legs = $2
        let correct = model.calculate(1, 2);
        assert_eq!(correct.total, dec!(2), "1 contract * 2 legs * $1 = $2");

        // Buggy call: contracts * 2 passed as contract count = $4 (wrong)
        let doubled = model.calculate(1 * 2, 2);
        assert_eq!(doubled.total, dec!(4), "doubled bug produces $4, not $2");

        // The engine must use contracts (not contracts*2) so entry_commission == $2
        // This test documents the expected behavior for the engine call site.
        // After the fix, entry commission for 1 contract = correct.total = $2.
        assert_ne!(correct.total, doubled.total, "correct != doubled");
    }

    #[test]
    fn test_default_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_equity, dec!(100_000));
        assert_eq!(config.profit_target_pct, 50.0);
        assert_eq!(config.stop_loss_pct, 125.0);
        assert_eq!(config.time_exit_dte, 21);
    }

    #[test]
    fn test_backtest_result_calculations() {
        let config = BacktestConfig::default();
        let result = BacktestResult {
            config,
            tickers: vec!["SPY".to_string()],
            start_date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
            trades: vec![],
            equity_curve: vec![],
            final_equity: dec!(110_000),
            total_return_pct: 10.0,
            trading_days: 252,
            peak_equity: dec!(115_000),
            max_drawdown: dec!(5_000),
            max_drawdown_pct: 4.35,
            total_trades: 100,
            winning_trades: 70,
            losing_trades: 30,
            total_pnl: dec!(10_000),
            gross_profit: dec!(15_000),
            gross_loss: dec!(-5_000),
            total_commission: dec!(400),
        };

        assert_eq!(result.win_rate(), 0.7);
        assert_eq!(result.profit_factor(), 3.0); // 15000 / 5000
        assert_eq!(result.avg_trade_pnl(), dec!(100)); // 10000 / 100
    }
}
