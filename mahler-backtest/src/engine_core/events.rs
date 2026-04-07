use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::data::{OptionType, OptionsSnapshot};
use crate::backtest::{ExitReason, Position, SpreadType};
use crate::risk::PortfolioGreeks;

/// A timestamped market data event.
#[derive(Debug, Clone)]
pub struct MarketEvent {
    pub timestamp: NaiveDateTime,
    pub snapshot: OptionsSnapshot,
}

impl MarketEvent {
    /// Create a MarketEvent from a snapshot.
    /// Uses 15:46 ET as the default time (ORATS snapshot time, ~14 min before close).
    pub fn new(snapshot: OptionsSnapshot) -> Self {
        let time = NaiveTime::from_hms_opt(15, 46, 0).unwrap();
        let timestamp = NaiveDateTime::new(snapshot.date, time);
        Self { timestamp, snapshot }
    }

    pub fn with_timestamp(snapshot: OptionsSnapshot, timestamp: NaiveDateTime) -> Self {
        Self { timestamp, snapshot }
    }
}

/// Side of an order leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// A single leg of an order intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegIntent {
    pub option_type: OptionType,
    pub strike: Decimal,
    pub expiration: NaiveDate,
    pub contracts: i32,
    pub side: OrderSide,
}

/// What action the order performs.
#[derive(Debug, Clone)]
pub enum OrderAction {
    Open,
    Close {
        position_id: u64,
        reason: ExitReason,
    },
}

/// An intent to place an order (not yet executed).
#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub ticker: String,
    pub spread_type: SpreadType,
    pub legs: Vec<LegIntent>,
    pub action: OrderAction,
}

impl OrderIntent {
    pub fn enter_spread(ticker: String, spread_type: SpreadType, legs: Vec<LegIntent>) -> Self {
        Self { ticker, spread_type, legs, action: OrderAction::Open }
    }

    pub fn exit_position(position_id: u64, reason: ExitReason) -> Self {
        Self {
            ticker: String::new(),
            spread_type: SpreadType::Custom,
            legs: vec![],
            action: OrderAction::Close { position_id, reason },
        }
    }
}

/// Result of executing an order.
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub order: OrderIntent,
    pub fill_prices: Vec<Decimal>,
    pub commission: Decimal,
    pub timestamp: NaiveDateTime,
}

/// Risk gate decision.
#[derive(Debug, Clone)]
pub enum RiskDecision {
    Allowed,
    Rejected(String),
}

impl RiskDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Read-only view of portfolio state, passed to Strategy.
#[derive(Debug, Clone)]
pub struct PortfolioView {
    pub equity: Decimal,
    pub cash: Decimal,
    pub open_positions: Vec<Position>,
    pub aggregate_greeks: PortfolioGreeks,
    pub current_drawdown_pct: f64,
    pub peak_equity: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    #[test]
    fn test_market_event_creation() {
        let snapshot = OptionsSnapshot::new(
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            "SPY".to_string(),
            dec!(480),
        );
        let event = MarketEvent::new(snapshot);
        assert_eq!(event.snapshot.ticker, "SPY");
        assert_eq!(event.snapshot.date, NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
        assert!(event.timestamp.date() == NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_order_intent_entry() {
        let order = OrderIntent::enter_spread(
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
        );
        assert_eq!(order.ticker, "SPY");
        assert_eq!(order.legs.len(), 2);
        assert!(matches!(order.action, OrderAction::Open));
    }

    #[test]
    fn test_order_intent_exit() {
        let order = OrderIntent::exit_position(42, ExitReason::ProfitTarget);
        assert!(matches!(order.action, OrderAction::Close { position_id: 42, reason: ExitReason::ProfitTarget }));
    }

    #[test]
    fn test_fill_event_pnl() {
        let fill = FillEvent {
            order: OrderIntent::exit_position(1, ExitReason::ProfitTarget),
            fill_prices: vec![dec!(1.20), dec!(0.80)],
            commission: dec!(2),
            timestamp: NaiveDateTime::new(
                NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
                chrono::NaiveTime::from_hms_opt(15, 46, 0).unwrap(),
            ),
        };
        assert_eq!(fill.commission, dec!(2));
        assert_eq!(fill.fill_prices.len(), 2);
    }

    #[test]
    fn test_risk_decision_variants() {
        let allowed = RiskDecision::Allowed;
        assert!(allowed.is_allowed());

        let rejected = RiskDecision::Rejected("Daily loss limit exceeded".to_string());
        assert!(!rejected.is_allowed());
    }

    #[test]
    fn test_portfolio_view_construction() {
        let view = PortfolioView {
            equity: dec!(100_000),
            cash: dec!(95_000),
            open_positions: vec![],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: 0.0,
            peak_equity: dec!(100_000),
        };
        assert_eq!(view.equity, dec!(100_000));
        assert_eq!(view.open_positions.len(), 0);
    }
}
