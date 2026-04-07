//! Position sizing module.
//!
//! Determines the number of contracts to trade based on risk limits:
//! - Per-trade risk: max 2% of equity
//! - Portfolio risk: max 10% of equity
//! - Single position: max 5% of equity
//! - Equity correlation: max 50% in correlated assets (SPY/QQQ/IWM)

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Position sizing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizerConfig {
    /// Maximum risk per trade as percentage of equity.
    pub max_risk_per_trade_pct: f64,
    /// Maximum total portfolio risk as percentage of equity.
    pub max_portfolio_risk_pct: f64,
    /// Maximum single position risk as percentage of equity.
    pub max_single_position_pct: f64,
    /// Maximum exposure to equity-correlated assets (SPY, QQQ, IWM).
    pub max_equity_correlation_pct: f64,
    /// Minimum contracts per trade.
    pub min_contracts: i32,
    /// Maximum contracts per trade.
    pub max_contracts: i32,
}

impl Default for PositionSizerConfig {
    fn default() -> Self {
        Self {
            max_risk_per_trade_pct: 2.0,
            max_portfolio_risk_pct: 10.0,
            max_single_position_pct: 5.0,
            max_equity_correlation_pct: 50.0,
            min_contracts: 1,
            max_contracts: 100,
        }
    }
}

/// Result of position sizing calculation.
#[derive(Debug, Clone)]
pub struct SizingResult {
    /// Recommended number of contracts.
    pub contracts: i32,
    /// Maximum allowed by per-trade risk.
    pub max_by_trade_risk: i32,
    /// Maximum allowed by portfolio risk.
    pub max_by_portfolio_risk: i32,
    /// Maximum allowed by single position limit.
    pub max_by_position_limit: i32,
    /// Whether sizing is constrained by any limit.
    pub is_constrained: bool,
    /// Constraint reason (if any).
    pub constraint_reason: Option<String>,
}

impl SizingResult {
    /// Check if the trade is allowed (at least 1 contract).
    pub fn is_allowed(&self) -> bool {
        self.contracts >= 1
    }
}

/// Current portfolio state for risk calculations.
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current equity.
    pub equity: Decimal,
    /// Total risk currently deployed (sum of max_loss for open positions).
    pub current_risk: Decimal,
    /// Risk deployed in equity-correlated assets (SPY, QQQ, IWM).
    pub equity_correlated_risk: Decimal,
    /// Number of open positions.
    pub open_positions: usize,
}

/// Position sizer for determining contract counts.
pub struct PositionSizer {
    config: PositionSizerConfig,
}

impl PositionSizer {
    pub fn new(config: PositionSizerConfig) -> Self {
        Self { config }
    }

    /// Calculate the number of contracts for a new trade.
    ///
    /// # Arguments
    /// * `portfolio` - Current portfolio state
    /// * `max_loss_per_contract` - Maximum loss per contract for the trade
    /// * `ticker` - Underlying ticker (for correlation checks)
    ///
    /// # Returns
    /// * `SizingResult` - Recommended contracts and constraint info
    pub fn calculate(
        &self,
        portfolio: &PortfolioState,
        max_loss_per_contract: Decimal,
        ticker: &str,
    ) -> SizingResult {
        let equity_f64: f64 = portfolio.equity.try_into().unwrap_or(100_000.0);
        let max_loss_f64: f64 = max_loss_per_contract.try_into().unwrap_or(1.0);

        if max_loss_f64 <= 0.0 {
            return SizingResult {
                contracts: 0,
                max_by_trade_risk: 0,
                max_by_portfolio_risk: 0,
                max_by_position_limit: 0,
                is_constrained: true,
                constraint_reason: Some("Invalid max loss per contract".to_string()),
            };
        }

        // 1. Per-trade risk limit
        let trade_risk_budget = equity_f64 * (self.config.max_risk_per_trade_pct / 100.0);
        let max_by_trade_risk = (trade_risk_budget / max_loss_f64).floor() as i32;

        // 2. Portfolio risk limit
        let current_risk_f64: f64 = portfolio.current_risk.try_into().unwrap_or(0.0);
        let portfolio_risk_budget =
            equity_f64 * (self.config.max_portfolio_risk_pct / 100.0) - current_risk_f64;
        let max_by_portfolio_risk = if portfolio_risk_budget > 0.0 {
            (portfolio_risk_budget / max_loss_f64).floor() as i32
        } else {
            0
        };

        // 3. Single position limit
        let position_budget = equity_f64 * (self.config.max_single_position_pct / 100.0);
        let max_by_position_limit = (position_budget / max_loss_f64).floor() as i32;

        // 4. Equity correlation limit (for SPY, QQQ, IWM)
        let mut max_by_correlation = i32::MAX;
        let is_equity_correlated = matches!(ticker.to_uppercase().as_str(), "SPY" | "QQQ" | "IWM");

        if is_equity_correlated {
            let current_correlated: f64 = portfolio.equity_correlated_risk.try_into().unwrap_or(0.0);
            let correlation_budget =
                equity_f64 * (self.config.max_equity_correlation_pct / 100.0) - current_correlated;
            max_by_correlation = if correlation_budget > 0.0 {
                (correlation_budget / max_loss_f64).floor() as i32
            } else {
                0
            };
        }

        // Find minimum of all constraints
        let raw_contracts = max_by_trade_risk
            .min(max_by_portfolio_risk)
            .min(max_by_position_limit)
            .min(max_by_correlation);

        // Apply min/max bounds
        let contracts = raw_contracts
            .max(0)
            .min(self.config.max_contracts)
            .max(if raw_contracts > 0 {
                self.config.min_contracts
            } else {
                0
            });

        // Determine constraint reason
        let (is_constrained, constraint_reason) = if raw_contracts < max_by_trade_risk {
            if raw_contracts == max_by_portfolio_risk {
                (
                    true,
                    Some(format!(
                        "Portfolio risk limit ({:.1}% of equity)",
                        self.config.max_portfolio_risk_pct
                    )),
                )
            } else if raw_contracts == max_by_position_limit {
                (
                    true,
                    Some(format!(
                        "Single position limit ({:.1}% of equity)",
                        self.config.max_single_position_pct
                    )),
                )
            } else if raw_contracts == max_by_correlation && is_equity_correlated {
                (
                    true,
                    Some(format!(
                        "Equity correlation limit ({:.1}%)",
                        self.config.max_equity_correlation_pct
                    )),
                )
            } else {
                (false, None)
            }
        } else {
            (false, None)
        };

        SizingResult {
            contracts,
            max_by_trade_risk,
            max_by_portfolio_risk,
            max_by_position_limit: max_by_position_limit.min(max_by_correlation),
            is_constrained,
            constraint_reason,
        }
    }

    /// Calculate position size for a specific account size and risk parameters.
    /// Convenience method for quick calculations.
    pub fn quick_size(equity: f64, max_loss_per_contract: f64, risk_pct: f64) -> i32 {
        if max_loss_per_contract <= 0.0 {
            return 0;
        }
        let risk_budget = equity * (risk_pct / 100.0);
        (risk_budget / max_loss_per_contract).floor() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_config() {
        let config = PositionSizerConfig::default();
        assert_eq!(config.max_risk_per_trade_pct, 2.0);
        assert_eq!(config.max_portfolio_risk_pct, 10.0);
        assert_eq!(config.max_single_position_pct, 5.0);
    }

    #[test]
    fn test_basic_sizing() {
        let sizer = PositionSizer::new(PositionSizerConfig::default());
        let portfolio = PortfolioState {
            equity: dec!(100_000),
            current_risk: dec!(0),
            equity_correlated_risk: dec!(0),
            open_positions: 0,
        };

        // $500 max loss per contract, 2% risk = $2000 budget = 4 contracts
        let result = sizer.calculate(&portfolio, dec!(500), "SPY");
        assert_eq!(result.max_by_trade_risk, 4);
        assert_eq!(result.contracts, 4);
    }

    #[test]
    fn test_portfolio_risk_limit() {
        let sizer = PositionSizer::new(PositionSizerConfig::default());
        let portfolio = PortfolioState {
            equity: dec!(100_000),
            current_risk: dec!(9_000), // Already have $9000 risk deployed
            equity_correlated_risk: dec!(0),
            open_positions: 5,
        };

        // $500 max loss, only $1000 portfolio budget left = 2 contracts
        let result = sizer.calculate(&portfolio, dec!(500), "SPY");
        assert_eq!(result.max_by_portfolio_risk, 2);
        assert!(result.is_constrained);
    }

    #[test]
    fn test_correlation_limit() {
        let sizer = PositionSizer::new(PositionSizerConfig::default());
        let portfolio = PortfolioState {
            equity: dec!(100_000),
            current_risk: dec!(0),
            equity_correlated_risk: dec!(49_000), // 49% in equity-correlated
            open_positions: 3,
        };

        // Only $1000 left for equity-correlated assets = 2 contracts
        // Trade risk would allow 4 contracts (2% of 100K / 500)
        // Single position would allow 10 (5% of 100K / 500)
        // So correlation is the most restrictive at 2
        let result = sizer.calculate(&portfolio, dec!(500), "SPY");
        assert_eq!(result.max_by_trade_risk, 4);
        assert_eq!(result.max_by_position_limit, 2); // min(10, 2) = 2
        assert_eq!(result.contracts, 2);
        assert!(result.is_constrained);
    }

    #[test]
    fn test_non_equity_correlated() {
        let sizer = PositionSizer::new(PositionSizerConfig::default());
        let portfolio = PortfolioState {
            equity: dec!(100_000),
            current_risk: dec!(0),
            equity_correlated_risk: dec!(50_000), // At limit for equity-correlated
            open_positions: 5,
        };

        // Non-equity ticker should not be limited by correlation
        let result = sizer.calculate(&portfolio, dec!(500), "GLD");
        assert_eq!(result.max_by_trade_risk, 4);
        assert!(!result.is_constrained || result.constraint_reason.as_ref().map(|s| !s.contains("correlation")).unwrap_or(true));
    }

    #[test]
    fn test_quick_size() {
        // $100K account, $500 max loss, 2% risk
        let contracts = PositionSizer::quick_size(100_000.0, 500.0, 2.0);
        assert_eq!(contracts, 4);

        // $50K account, $200 max loss, 2% risk
        let contracts = PositionSizer::quick_size(50_000.0, 200.0, 2.0);
        assert_eq!(contracts, 5);
    }
}
