//! Risk gate module.
//!
//! Combines circuit breakers and position sizing to approve or reject orders.

use rust_decimal::Decimal;

use crate::engine_core::events::{OrderAction, OrderIntent, PortfolioView, RiskDecision};
use crate::data::OptionsSnapshot;

use super::circuit_breakers::{CircuitBreaker, CircuitBreakerConfig};
use super::position_sizer::{PortfolioState, PositionSizer, PositionSizerConfig};

pub trait RiskGate {
    fn check(&self, order: &OrderIntent, portfolio: &PortfolioView) -> RiskDecision;
    fn update(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView);
}

pub struct DefaultRiskGate {
    circuit_breaker: CircuitBreaker,
    position_sizer: PositionSizer,
}

impl DefaultRiskGate {
    pub fn new(cb_config: CircuitBreakerConfig, initial_equity: Decimal) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(cb_config, initial_equity),
            position_sizer: PositionSizer::new(PositionSizerConfig::default()),
        }
    }
}

impl RiskGate for DefaultRiskGate {
    fn check(&self, order: &OrderIntent, portfolio: &PortfolioView) -> RiskDecision {
        // Always allow exits
        if matches!(order.action, OrderAction::Close { .. }) {
            return RiskDecision::Allowed;
        }

        // Check circuit breaker
        if !self.circuit_breaker.allows_new_positions() {
            return RiskDecision::Rejected(
                self.circuit_breaker.status().reason().to_string()
            );
        }

        // Check position sizing -- estimate max_loss_per_contract from spread width
        let max_loss_per_contract = estimate_max_loss(order);
        let portfolio_state = PortfolioState {
            equity: portfolio.equity,
            current_risk: portfolio.open_positions.iter().map(|p| p.max_loss).sum(),
            equity_correlated_risk: portfolio.open_positions.iter()
                .filter(|p| matches!(p.ticker.to_uppercase().as_str(), "SPY" | "QQQ" | "IWM"))
                .map(|p| p.max_loss)
                .sum(),
            open_positions: portfolio.open_positions.len(),
        };

        let sizing = self.position_sizer.calculate(&portfolio_state, max_loss_per_contract, &order.ticker);
        if !sizing.is_allowed() {
            return RiskDecision::Rejected(
                sizing.constraint_reason.unwrap_or_else(|| "Position sizing limit reached".to_string())
            );
        }

        RiskDecision::Allowed
    }

    fn update(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) {
        let estimated_vix = None;
        self.circuit_breaker.update(snapshot.date, portfolio.equity, estimated_vix);
    }
}

/// Estimate max loss per contract from order legs (spread width * 100).
fn estimate_max_loss(order: &OrderIntent) -> Decimal {
    if order.legs.len() < 2 {
        return Decimal::from(500); // Default fallback
    }
    let strikes: Vec<Decimal> = order.legs.iter().map(|l| l.strike).collect();
    let max_strike = strikes.iter().copied().max().unwrap_or(Decimal::ZERO);
    let min_strike = strikes.iter().copied().min().unwrap_or(Decimal::ZERO);
    let width = max_strike - min_strike;
    width * Decimal::from(100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use crate::engine_core::events::{OrderIntent, PortfolioView, LegIntent, OrderSide};
    use crate::backtest::SpreadType;
    use crate::data::OptionType;
    use crate::risk::PortfolioGreeks;
    use rust_decimal::prelude::FromPrimitive;

    fn make_view(equity: Decimal, drawdown_pct: f64) -> PortfolioView {
        PortfolioView {
            equity,
            cash: equity,
            open_positions: vec![],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: drawdown_pct,
            peak_equity: equity / (Decimal::ONE - Decimal::from_f64(drawdown_pct / 100.0).unwrap_or_default()),
        }
    }

    fn make_entry_order() -> OrderIntent {
        OrderIntent::enter_spread(
            "SPY".to_string(),
            SpreadType::PutCreditSpread,
            vec![
                LegIntent {
                    option_type: OptionType::Put,
                    strike: dec!(470),
                    expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
                    contracts: -1,
                    side: OrderSide::Sell,
                },
                LegIntent {
                    option_type: OptionType::Put,
                    strike: dec!(465),
                    expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
                    contracts: 1,
                    side: OrderSide::Buy,
                },
            ],
        )
    }

    #[test]
    fn test_normal_conditions_allow_order() {
        let gate = DefaultRiskGate::new(
            CircuitBreakerConfig::default(),
            dec!(100_000),
        );
        let view = make_view(dec!(100_000), 0.0);
        let order = make_entry_order();

        let decision = gate.check(&order, &view);
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_circuit_breaker_rejects_after_daily_loss() {
        let mut gate = DefaultRiskGate::new(
            CircuitBreakerConfig::default(),
            dec!(100_000),
        );

        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let snapshot = crate::data::OptionsSnapshot::new(date, "SPY".to_string(), dec!(480));
        let mut view = make_view(dec!(100_000), 0.0);

        gate.update(&snapshot, &view);

        // Simulate 3% daily loss
        view.equity = dec!(97_000);
        gate.update(&snapshot, &view);

        let order = make_entry_order();
        let decision = gate.check(&order, &view);
        assert!(!decision.is_allowed());
    }
}
