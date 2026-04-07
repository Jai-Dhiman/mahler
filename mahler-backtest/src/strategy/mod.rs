pub mod put_spread;

use crate::engine_core::events::{FillEvent, OrderIntent, PortfolioView};
use crate::data::OptionsSnapshot;

/// Strategy trait: determines when to enter and exit positions.
///
/// Implementations own all signal logic. The engine owns safety (circuit breakers,
/// position sizing).
pub trait Strategy {
    fn on_snapshot(
        &mut self,
        snapshot: &OptionsSnapshot,
        portfolio: &PortfolioView,
    ) -> Vec<OrderIntent>;

    fn on_fill(&mut self, fill: &FillEvent);
}

/// A no-op strategy that never trades. Useful for testing the engine in isolation.
pub struct NoOpStrategy;

impl Strategy for NoOpStrategy {
    fn on_snapshot(&mut self, _snapshot: &OptionsSnapshot, _portfolio: &PortfolioView) -> Vec<OrderIntent> {
        vec![]
    }

    fn on_fill(&mut self, _fill: &FillEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::data::OptionsSnapshot;
    use crate::engine_core::events::PortfolioView;
    use crate::risk::PortfolioGreeks;

    #[test]
    fn test_noop_strategy_returns_no_orders() {
        let mut strategy = NoOpStrategy;
        let snapshot = OptionsSnapshot::new(
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            "SPY".to_string(),
            dec!(480),
        );
        let portfolio = PortfolioView {
            equity: dec!(100_000),
            cash: dec!(100_000),
            open_positions: vec![],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: 0.0,
            peak_equity: dec!(100_000),
        };
        let orders = strategy.on_snapshot(&snapshot, &portfolio);
        assert!(orders.is_empty());
    }
}
