//! Performance metrics calculator.
//!
//! Calculates comprehensive trading performance statistics.

use std::collections::HashMap;

use chrono::{Datelike, NaiveDate};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::backtest::{BacktestResult, Trade};

/// Comprehensive performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Basic statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,

    // P&L metrics
    pub total_pnl: Decimal,
    pub gross_profit: Decimal,
    pub gross_loss: Decimal,
    pub profit_factor: f64,
    pub avg_trade_pnl: Decimal,
    pub avg_winner: Decimal,
    pub avg_loser: Decimal,
    pub largest_winner: Decimal,
    pub largest_loser: Decimal,

    // Return metrics
    pub total_return_pct: f64,
    pub cagr: f64,

    // Risk metrics
    pub max_drawdown: Decimal,
    pub max_drawdown_pct: f64,
    pub avg_drawdown: f64,
    pub drawdown_duration_days: i64,

    // Risk-adjusted returns
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,

    // Time metrics
    pub trading_days: usize,
    pub avg_days_in_trade: f64,
    pub avg_days_to_profit_target: f64,
    pub avg_days_to_stop_loss: f64,

    // Commission
    pub total_commission: Decimal,
    pub commission_pct_of_pnl: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            total_pnl: Decimal::ZERO,
            gross_profit: Decimal::ZERO,
            gross_loss: Decimal::ZERO,
            profit_factor: 0.0,
            avg_trade_pnl: Decimal::ZERO,
            avg_winner: Decimal::ZERO,
            avg_loser: Decimal::ZERO,
            largest_winner: Decimal::ZERO,
            largest_loser: Decimal::ZERO,
            total_return_pct: 0.0,
            cagr: 0.0,
            max_drawdown: Decimal::ZERO,
            max_drawdown_pct: 0.0,
            avg_drawdown: 0.0,
            drawdown_duration_days: 0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            trading_days: 0,
            avg_days_in_trade: 0.0,
            avg_days_to_profit_target: 0.0,
            avg_days_to_stop_loss: 0.0,
            total_commission: Decimal::ZERO,
            commission_pct_of_pnl: 0.0,
        }
    }
}

impl PerformanceMetrics {
    /// Generate a summary report.
    pub fn summary(&self) -> String {
        format!(
            "Performance Summary\n\
             ====================\n\
             \n\
             Trades: {} (W: {}, L: {})\n\
             Win Rate: {:.1}%\n\
             Profit Factor: {:.2}\n\
             \n\
             Total P&L: ${:.2}\n\
             Avg Trade: ${:.2}\n\
             Avg Winner: ${:.2}\n\
             Avg Loser: ${:.2}\n\
             Largest Win: ${:.2}\n\
             Largest Loss: ${:.2}\n\
             \n\
             Total Return: {:.2}%\n\
             CAGR: {:.2}%\n\
             \n\
             Max Drawdown: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Sortino Ratio: {:.2}\n\
             Calmar Ratio: {:.2}\n\
             \n\
             Avg Days in Trade: {:.1}\n\
             Commission: ${:.2} ({:.2}% of P&L)",
            self.total_trades,
            self.winning_trades,
            self.losing_trades,
            self.win_rate * 100.0,
            self.profit_factor,
            self.total_pnl,
            self.avg_trade_pnl,
            self.avg_winner,
            self.avg_loser,
            self.largest_winner,
            self.largest_loser,
            self.total_return_pct,
            self.cagr,
            self.max_drawdown_pct,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
            self.avg_days_in_trade,
            self.total_commission,
            self.commission_pct_of_pnl
        )
    }
}

/// Drawdown analysis details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    pub max_drawdown: Decimal,
    pub max_drawdown_pct: f64,
    pub max_drawdown_date: Option<NaiveDate>,
    pub peak_date: Option<NaiveDate>,
    pub recovery_date: Option<NaiveDate>,
    pub duration_days: i64,
    pub avg_drawdown_pct: f64,
    pub drawdown_periods: usize,
}

/// Monthly return data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyReturn {
    pub year: i32,
    pub month: u32,
    pub return_pct: f64,
    pub trades: usize,
    pub win_rate: f64,
}

/// Metrics calculator.
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Calculate all metrics from a backtest result.
    pub fn calculate(result: &BacktestResult) -> PerformanceMetrics {
        let trades = &result.trades;

        // Basic counts
        let total_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.is_winner()).count();
        let losing_trades = total_trades - winning_trades;
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        // P&L calculations
        let total_pnl = result.total_pnl;
        let gross_profit = result.gross_profit;
        let gross_loss = result.gross_loss;
        let profit_factor = Self::calculate_profit_factor(gross_profit, gross_loss);

        let avg_trade_pnl = if total_trades > 0 {
            total_pnl / Decimal::from(total_trades as i64)
        } else {
            Decimal::ZERO
        };

        let avg_winner = if winning_trades > 0 {
            gross_profit / Decimal::from(winning_trades as i64)
        } else {
            Decimal::ZERO
        };

        let avg_loser = if losing_trades > 0 {
            gross_loss / Decimal::from(losing_trades as i64)
        } else {
            Decimal::ZERO
        };

        let largest_winner = trades
            .iter()
            .filter(|t| t.is_winner())
            .map(|t| t.pnl())
            .max()
            .unwrap_or(Decimal::ZERO);

        let largest_loser = trades
            .iter()
            .filter(|t| !t.is_winner())
            .map(|t| t.pnl())
            .min()
            .unwrap_or(Decimal::ZERO);

        // Return metrics
        let total_return_pct = result.total_return_pct;
        let cagr = Self::calculate_cagr(
            result.config.initial_equity,
            result.final_equity,
            result.trading_days,
        );

        // Drawdown
        let drawdown = Self::analyze_drawdown(&result.equity_curve);

        // Risk-adjusted returns
        let sharpe_ratio = result.sharpe_ratio();
        let sortino_ratio = Self::calculate_sortino(&result.equity_curve);
        let calmar_ratio = if drawdown.max_drawdown_pct > 0.0 {
            cagr / drawdown.max_drawdown_pct
        } else {
            0.0
        };

        // Time metrics
        let trading_days = result.trading_days;
        let avg_days_in_trade = if total_trades > 0 {
            trades.iter().map(|t| t.days_held as f64).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        let avg_days_to_profit_target = Self::avg_days_by_exit_reason(
            trades,
            crate::backtest::ExitReason::ProfitTarget,
        );
        let avg_days_to_stop_loss =
            Self::avg_days_by_exit_reason(trades, crate::backtest::ExitReason::StopLoss);

        // Commission
        let total_commission = result.total_commission;
        let commission_pct_of_pnl = if !total_pnl.is_zero() {
            let comm: f64 = total_commission.abs().try_into().unwrap_or(0.0);
            let pnl: f64 = total_pnl.abs().try_into().unwrap_or(1.0);
            comm / pnl * 100.0
        } else {
            0.0
        };

        PerformanceMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            gross_profit,
            gross_loss,
            profit_factor,
            avg_trade_pnl,
            avg_winner,
            avg_loser,
            largest_winner,
            largest_loser,
            total_return_pct,
            cagr,
            max_drawdown: drawdown.max_drawdown,
            max_drawdown_pct: drawdown.max_drawdown_pct,
            avg_drawdown: drawdown.avg_drawdown_pct,
            drawdown_duration_days: drawdown.duration_days,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            trading_days,
            avg_days_in_trade,
            avg_days_to_profit_target,
            avg_days_to_stop_loss,
            total_commission,
            commission_pct_of_pnl,
        }
    }

    /// Calculate profit factor.
    fn calculate_profit_factor(gross_profit: Decimal, gross_loss: Decimal) -> f64 {
        let loss: f64 = gross_loss.abs().try_into().unwrap_or(0.0);
        if loss == 0.0 {
            return f64::INFINITY;
        }
        let profit: f64 = gross_profit.try_into().unwrap_or(0.0);
        profit / loss
    }

    /// Calculate CAGR (Compound Annual Growth Rate).
    fn calculate_cagr(initial: Decimal, final_val: Decimal, trading_days: usize) -> f64 {
        let init: f64 = initial.try_into().unwrap_or(1.0);
        let fin: f64 = final_val.try_into().unwrap_or(1.0);

        if init <= 0.0 || trading_days == 0 {
            return 0.0;
        }

        let years = trading_days as f64 / 252.0;
        if years <= 0.0 {
            return 0.0;
        }

        ((fin / init).powf(1.0 / years) - 1.0) * 100.0
    }

    /// Analyze drawdown from equity curve.
    fn analyze_drawdown(
        equity_curve: &[crate::backtest::EquityPoint],
    ) -> DrawdownAnalysis {
        if equity_curve.is_empty() {
            return DrawdownAnalysis {
                max_drawdown: Decimal::ZERO,
                max_drawdown_pct: 0.0,
                max_drawdown_date: None,
                peak_date: None,
                recovery_date: None,
                duration_days: 0,
                avg_drawdown_pct: 0.0,
                drawdown_periods: 0,
            };
        }

        let mut peak = equity_curve[0].equity;
        let mut peak_date = equity_curve[0].date;
        let mut max_drawdown = Decimal::ZERO;
        let mut max_drawdown_pct = 0.0;
        let mut max_drawdown_date = equity_curve[0].date;
        let mut drawdown_start: Option<NaiveDate> = None;
        let mut max_duration = 0i64;
        let mut current_duration = 0i64;
        let mut drawdowns = Vec::new();
        let mut periods = 0;

        for point in equity_curve {
            if point.equity > peak {
                // New high
                if drawdown_start.is_some() {
                    // End of drawdown period
                    periods += 1;
                }
                peak = point.equity;
                peak_date = point.date;
                drawdown_start = None;
                current_duration = 0;
            } else {
                // In drawdown
                let drawdown = peak - point.equity;
                let drawdown_pct = if !peak.is_zero() {
                    let dd: f64 = drawdown.try_into().unwrap_or(0.0);
                    let pk: f64 = peak.try_into().unwrap_or(1.0);
                    dd / pk * 100.0
                } else {
                    0.0
                };

                if drawdown_start.is_none() {
                    drawdown_start = Some(point.date);
                }

                current_duration = (point.date - drawdown_start.unwrap()).num_days();

                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                    max_drawdown_pct = drawdown_pct;
                    max_drawdown_date = point.date;
                    max_duration = current_duration;
                }

                drawdowns.push(drawdown_pct);
            }
        }

        let avg_drawdown_pct = if !drawdowns.is_empty() {
            drawdowns.iter().sum::<f64>() / drawdowns.len() as f64
        } else {
            0.0
        };

        DrawdownAnalysis {
            max_drawdown,
            max_drawdown_pct,
            max_drawdown_date: Some(max_drawdown_date),
            peak_date: Some(peak_date),
            recovery_date: None, // Would need to track recovery
            duration_days: max_duration,
            avg_drawdown_pct,
            drawdown_periods: periods,
        }
    }

    /// Calculate Sortino ratio (downside deviation).
    fn calculate_sortino(equity_curve: &[crate::backtest::EquityPoint]) -> f64 {
        if equity_curve.len() < 2 {
            return 0.0;
        }

        // Calculate daily returns
        let returns: Vec<f64> = equity_curve
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

        // Downside deviation (only negative returns)
        let downside_variance = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let downside_dev = downside_variance.sqrt();

        if downside_dev == 0.0 {
            return 0.0;
        }

        // Annualize
        (mean * 252.0_f64.sqrt()) / downside_dev
    }

    /// Calculate average days in trade by exit reason.
    fn avg_days_by_exit_reason(trades: &[Trade], reason: crate::backtest::ExitReason) -> f64 {
        let matching: Vec<_> = trades
            .iter()
            .filter(|t| t.position.exit_reason == Some(reason))
            .collect();

        if matching.is_empty() {
            return 0.0;
        }

        matching.iter().map(|t| t.days_held as f64).sum::<f64>() / matching.len() as f64
    }

    /// Calculate monthly returns.
    pub fn monthly_returns(result: &BacktestResult) -> Vec<MonthlyReturn> {
        let mut monthly: HashMap<(i32, u32), Vec<&Trade>> = HashMap::new();

        for trade in &result.trades {
            if let Some(exit_date) = trade.position.exit_date {
                let key = (exit_date.year(), exit_date.month());
                monthly.entry(key).or_default().push(trade);
            }
        }

        let initial: f64 = result.config.initial_equity.try_into().unwrap_or(100_000.0);

        let mut returns: Vec<_> = monthly
            .into_iter()
            .map(|((year, month), trades)| {
                let pnl: f64 = trades.iter().map(|t| {
                    let p: f64 = t.pnl().try_into().unwrap_or(0.0);
                    p
                }).sum();
                let winners = trades.iter().filter(|t| t.is_winner()).count();
                let win_rate = if !trades.is_empty() {
                    winners as f64 / trades.len() as f64
                } else {
                    0.0
                };

                MonthlyReturn {
                    year,
                    month,
                    return_pct: pnl / initial * 100.0,
                    trades: trades.len(),
                    win_rate,
                }
            })
            .collect();

        returns.sort_by(|a, b| {
            (a.year, a.month).cmp(&(b.year, b.month))
        });

        returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profit_factor() {
        let pf = MetricsCalculator::calculate_profit_factor(
            Decimal::from(15000),
            Decimal::from(-5000),
        );
        assert_eq!(pf, 3.0);
    }

    #[test]
    fn test_cagr() {
        // 100K -> 121K over 2 years (504 days) = 10% CAGR
        let cagr = MetricsCalculator::calculate_cagr(
            Decimal::from(100000),
            Decimal::from(121000),
            504,
        );
        assert!((cagr - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.total_trades, 0);
        assert_eq!(metrics.win_rate, 0.0);
    }

    #[test]
    fn test_drawdown_analysis_empty() {
        let analysis = MetricsCalculator::analyze_drawdown(&[]);
        assert_eq!(analysis.max_drawdown_pct, 0.0);
    }
}
