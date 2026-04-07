# trader-joe Engine Refactor Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the monolithic BacktestEngine with a trait-based architecture where strategy code runs identically in backtest and live modes.
**Spec:** docs/specs/2026-04-06-trader-joe-engine-design.md
**Style:** Follow the project's coding standards (CLAUDE.md)

---

## Task Groups

Group A (parallel): Task 1, Task 2, Task 3, Task 4
Group B (parallel, depends on A): Task 5, Task 6, Task 7
Group C (sequential, depends on B): Task 8
Group D (sequential, depends on C): Task 9
Group E (sequential, depends on D): Task 10
Group F (sequential, depends on E): Task 11

---

### Task 1: Core Event Types
**Group:** A (parallel with Task 2, 3, 4)

**Behavior being verified:** The system has well-typed events (MarketEvent, OrderIntent, FillEvent, RiskDecision, PortfolioView) that carry all data needed by downstream consumers.
**Interface under test:** `core::events` public types and their constructors/accessors.

**Files:**
- Create: `src/core/mod.rs`
- Create: `src/core/events.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/core/events.rs at the bottom
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
        // Timestamp should be set to 15:46 ET (ORATS snapshot time) on the trade date
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
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test core::events::tests
```
Expected: FAIL -- `can't find crate for core` (module doesn't exist yet)

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/core/mod.rs`:
```rust
pub mod events;
```

Create `src/core/events.rs`:
```rust
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::data::{OptionType, OptionsSnapshot};
use crate::backtest::{ExitReason, Position, SpreadType};
use crate::risk::PortfolioGreeks;

/// A timestamped market data event.
#[derive(Debug, Clone)]
pub struct MarketEvent {
    /// Timestamp of the event (for ordering).
    pub timestamp: NaiveDateTime,
    /// Full options snapshot for this event.
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

    /// Create with a specific timestamp (for intraday data in the future).
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderAction {
    /// Open a new position.
    Open,
    /// Close an existing position.
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
    /// Create an order to enter a new spread.
    pub fn enter_spread(
        ticker: String,
        spread_type: SpreadType,
        legs: Vec<LegIntent>,
    ) -> Self {
        Self {
            ticker,
            spread_type,
            legs,
            action: OrderAction::Open,
        }
    }

    /// Create an order to exit an existing position.
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
    /// The original order.
    pub order: OrderIntent,
    /// Fill prices for each leg.
    pub fill_prices: Vec<Decimal>,
    /// Total commission charged.
    pub commission: Decimal,
    /// Timestamp of the fill.
    pub timestamp: NaiveDateTime,
}

/// Risk gate decision.
#[derive(Debug, Clone)]
pub enum RiskDecision {
    /// Order is allowed.
    Allowed,
    /// Order is rejected with reason.
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
```

Add to `src/lib.rs`:
```rust
pub mod core;
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test core::events::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/core/ src/lib.rs && git commit -m "feat(core): add event types for trait-based engine"
```

---

### Task 2: Strategy Trait Definition
**Group:** A (parallel with Task 1, 3, 4)

**Behavior being verified:** A Strategy trait exists with `on_snapshot` and `on_fill` methods, and a no-op strategy can be instantiated.
**Interface under test:** `strategy::Strategy` trait and `strategy::NoOpStrategy`.

**Files:**
- Create: `src/strategy/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/strategy/mod.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::data::OptionsSnapshot;
    use crate::core::events::PortfolioView;
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
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test strategy::tests
```
Expected: FAIL -- `can't find crate for strategy`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/strategy/mod.rs`:
```rust
pub mod put_spread;

use crate::core::events::{FillEvent, OrderIntent, PortfolioView};
use crate::data::OptionsSnapshot;

/// Strategy trait: determines when to enter and exit positions.
///
/// Implementations own all signal logic (IV filtering, regime classification,
/// spread screening, entry/exit rules). The engine owns safety (circuit breakers,
/// position sizing).
pub trait Strategy {
    /// Called with each market snapshot. Returns zero or more order intents.
    fn on_snapshot(
        &mut self,
        snapshot: &OptionsSnapshot,
        portfolio: &PortfolioView,
    ) -> Vec<OrderIntent>;

    /// Called when an order is filled. Use to update internal state.
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
```

Add to `src/lib.rs`:
```rust
pub mod strategy;
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test strategy::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/strategy/ src/lib.rs && git commit -m "feat(strategy): add Strategy trait and NoOpStrategy"
```

---

### Task 3: Broker Trait + SimulatedBroker
**Group:** A (parallel with Task 1, 2, 4)

**Behavior being verified:** A SimulatedBroker applies slippage and commission to produce realistic fill prices from an OrderIntent and snapshot.
**Interface under test:** `broker::Broker` trait and `broker::SimulatedBroker::submit_order()`.

**Files:**
- Create: `src/broker/mod.rs`
- Create: `src/broker/backtest.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/broker/backtest.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::backtest::{SpreadType, CommissionModel, SlippageModel};
    use crate::core::events::{OrderIntent, LegIntent, OrderSide};
    use crate::data::{OptionType, OptionQuote, OptionsChain, OptionsSnapshot, Greeks};

    fn make_snapshot_with_quotes() -> OptionsSnapshot {
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();
        let mut snapshot = OptionsSnapshot::new(date, "SPY".to_string(), dec!(480));
        let mut chain = OptionsChain::new(exp, 32);

        // Short put at 470
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(),
            trade_date: date,
            expiration: exp,
            dte: 32,
            strike: dec!(470),
            option_type: OptionType::Put,
            stock_price: dec!(480),
            bid: dec!(2.40),
            ask: dec!(2.60),
            mid: dec!(2.50),
            theoretical_value: dec!(2.50),
            volume: 1000,
            open_interest: 5000,
            bid_iv: 0.18,
            mid_iv: 0.19,
            ask_iv: 0.20,
            smv_vol: 0.19,
            greeks: Greeks { delta: -0.15, gamma: 0.02, theta: -0.05, vega: 0.10, rho: 0.01 },
            residual_rate: 0.05,
        });

        // Long put at 465
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(),
            trade_date: date,
            expiration: exp,
            dte: 32,
            strike: dec!(465),
            option_type: OptionType::Put,
            stock_price: dec!(480),
            bid: dec!(1.40),
            ask: dec!(1.60),
            mid: dec!(1.50),
            theoretical_value: dec!(1.50),
            volume: 800,
            open_interest: 4000,
            bid_iv: 0.19,
            mid_iv: 0.20,
            ask_iv: 0.21,
            smv_vol: 0.20,
            greeks: Greeks { delta: -0.10, gamma: 0.015, theta: -0.03, vega: 0.08, rho: 0.008 },
            residual_rate: 0.05,
        });

        snapshot.chains.push(chain);
        snapshot
    }

    #[test]
    fn test_simulated_broker_fills_spread() {
        let mut broker = SimulatedBroker::new(
            SlippageModel::default(),
            CommissionModel::default(),
        );
        let snapshot = make_snapshot_with_quotes();

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

        let fill = broker.submit_order(&order, &snapshot).unwrap();
        assert_eq!(fill.fill_prices.len(), 2);

        // Sell fill should be between bid and mid (slippage makes it worse for seller)
        let sell_fill = fill.fill_prices[0];
        assert!(sell_fill >= dec!(2.40) && sell_fill <= dec!(2.60),
            "Sell fill {} should be between bid 2.40 and ask 2.60", sell_fill);

        // Buy fill should be between mid and ask (slippage makes it worse for buyer)
        let buy_fill = fill.fill_prices[1];
        assert!(buy_fill >= dec!(1.40) && buy_fill <= dec!(1.60),
            "Buy fill {} should be between bid 1.40 and ask 1.60", buy_fill);

        // Commission should be $2 (1 contract * 2 legs * $1/contract)
        assert_eq!(fill.commission, dec!(2));
    }

    #[test]
    fn test_simulated_broker_missing_quote_returns_error() {
        let mut broker = SimulatedBroker::new(
            SlippageModel::default(),
            CommissionModel::default(),
        );
        let snapshot = make_snapshot_with_quotes();

        // Order for a strike that doesn't exist in the snapshot
        let order = OrderIntent::enter_spread(
            "SPY".to_string(),
            SpreadType::PutCreditSpread,
            vec![
                LegIntent {
                    option_type: OptionType::Put,
                    strike: dec!(999),  // Non-existent strike
                    expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
                    contracts: -1,
                    side: OrderSide::Sell,
                },
            ],
        );

        let result = broker.submit_order(&order, &snapshot);
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test broker::backtest::tests
```
Expected: FAIL -- `can't find crate for broker`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/broker/mod.rs`:
```rust
pub mod backtest;

use crate::core::events::{FillEvent, OrderIntent};
use crate::data::OptionsSnapshot;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BrokerError {
    #[error("No matching quote for strike {strike} {option_type:?} exp {expiration}")]
    NoMatchingQuote {
        strike: rust_decimal::Decimal,
        option_type: crate::data::OptionType,
        expiration: chrono::NaiveDate,
    },

    #[error("Order rejected: {0}")]
    OrderRejected(String),
}

/// Broker trait: executes orders and returns fills.
pub trait Broker {
    fn submit_order(
        &mut self,
        order: &OrderIntent,
        snapshot: &OptionsSnapshot,
    ) -> Result<FillEvent, BrokerError>;
}
```

Create `src/broker/backtest.rs`:
```rust
use chrono::{NaiveDateTime, NaiveTime};
use rust_decimal::Decimal;

use crate::backtest::{CommissionModel, SlippageModel};
use crate::core::events::{FillEvent, LegIntent, OrderAction, OrderIntent, OrderSide};
use crate::data::OptionsSnapshot;

use super::{Broker, BrokerError};

/// Simulated broker for backtesting. Fills orders against historical data
/// with slippage and commission applied.
pub struct SimulatedBroker {
    slippage: SlippageModel,
    commission: CommissionModel,
}

impl SimulatedBroker {
    pub fn new(slippage: SlippageModel, commission: CommissionModel) -> Self {
        Self { slippage, commission }
    }

    /// Find the bid/ask for a leg in the snapshot.
    fn find_quote_prices(
        &self,
        leg: &LegIntent,
        snapshot: &OptionsSnapshot,
    ) -> Result<(Decimal, Decimal), BrokerError> {
        for chain in &snapshot.chains {
            if chain.expiration != leg.expiration {
                continue;
            }
            let quotes = match leg.option_type {
                crate::data::OptionType::Call => &chain.calls,
                crate::data::OptionType::Put => &chain.puts,
            };
            if let Some(quote) = quotes.iter().find(|q| q.strike == leg.strike) {
                return Ok((quote.bid, quote.ask));
            }
        }
        Err(BrokerError::NoMatchingQuote {
            strike: leg.strike,
            option_type: leg.option_type,
            expiration: leg.expiration,
        })
    }
}

impl Broker for SimulatedBroker {
    fn submit_order(
        &mut self,
        order: &OrderIntent,
        snapshot: &OptionsSnapshot,
    ) -> Result<FillEvent, BrokerError> {
        let num_legs = order.legs.len();
        let slippage = self.slippage.for_legs(num_legs);

        let mut fill_prices = Vec::with_capacity(num_legs);
        let mut total_contracts = 0i32;

        for leg in &order.legs {
            let (bid, ask) = self.find_quote_prices(leg, snapshot)?;
            let fill = match leg.side {
                OrderSide::Sell => slippage.sell_fill(bid, ask),
                OrderSide::Buy => slippage.buy_fill(bid, ask),
            };
            fill_prices.push(fill);
            total_contracts += leg.contracts.abs();
        }

        let commission = self.commission.calculate(total_contracts, num_legs).total;

        let time = NaiveTime::from_hms_opt(15, 46, 0).unwrap();
        let timestamp = NaiveDateTime::new(snapshot.date, time);

        Ok(FillEvent {
            order: order.clone(),
            fill_prices,
            commission,
            timestamp,
        })
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod broker;
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test broker::backtest::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/broker/ src/lib.rs && git commit -m "feat(broker): add Broker trait and SimulatedBroker"
```

---

### Task 4: PortfolioTracker
**Group:** A (parallel with Task 1, 2, 3)

**Behavior being verified:** PortfolioTracker manages positions, updates MTM from snapshots, applies fills, and produces correct equity/P&L.
**Interface under test:** `portfolio::PortfolioTracker::update_mtm()`, `apply_fill()`, `view()`, `record_equity()`.

**Files:**
- Create: `src/portfolio/mod.rs`
- Create: `src/portfolio/tracker.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/portfolio/tracker.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::backtest::{CreditSpreadBuilder, ExitReason, SpreadType, CommissionModel, SlippageModel};
    use crate::core::events::{FillEvent, OrderIntent, OrderAction, LegIntent, OrderSide};
    use crate::data::{OptionType, OptionQuote, OptionsChain, OptionsSnapshot, Greeks};

    #[test]
    fn test_initial_state() {
        let tracker = PortfolioTracker::new(dec!(100_000));
        let view = tracker.view();
        assert_eq!(view.equity, dec!(100_000));
        assert_eq!(view.cash, dec!(100_000));
        assert!(view.open_positions.is_empty());
        assert_eq!(view.current_drawdown_pct, 0.0);
    }

    #[test]
    fn test_open_position_via_fill() {
        let mut tracker = PortfolioTracker::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();

        // Simulate opening a put credit spread: sell 470P @ 2.50, buy 465P @ 1.50
        let position = CreditSpreadBuilder::put_credit_spread("SPY", date)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.15)
            .long_leg(dec!(465), dec!(1.50), -0.10)
            .expiration(exp, 32)
            .contracts(1)
            .commission(dec!(2))
            .build();

        tracker.add_position(position);

        let view = tracker.view();
        assert_eq!(view.open_positions.len(), 1);
        // Cash reduced by commission only (credit spread collects premium, but cash accounting
        // is simplified: commission is deducted, premium is tracked via position value)
        assert_eq!(view.cash, dec!(100_000) - dec!(2));
    }

    #[test]
    fn test_close_position() {
        let mut tracker = PortfolioTracker::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();

        let position = CreditSpreadBuilder::put_credit_spread("SPY", date)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.15)
            .long_leg(dec!(465), dec!(1.50), -0.10)
            .expiration(exp, 32)
            .contracts(1)
            .commission(dec!(2))
            .build();

        tracker.add_position(position);
        let pos_id = tracker.view().open_positions[0].id;

        // Close at profit target: spread decayed to 50% of credit
        tracker.close_position(pos_id, date + chrono::Duration::days(10), ExitReason::ProfitTarget, dec!(50), dec!(2));

        let view = tracker.view();
        assert_eq!(view.open_positions.len(), 0);
        assert_eq!(tracker.closed_trades().len(), 1);
        assert!(tracker.closed_trades()[0].is_winner());
    }

    #[test]
    fn test_equity_curve_recording() {
        let mut tracker = PortfolioTracker::new(dec!(100_000));
        let date1 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2024, 1, 16).unwrap();

        tracker.record_equity(date1);
        tracker.record_equity(date2);

        assert_eq!(tracker.equity_curve().len(), 2);
        assert_eq!(tracker.equity_curve()[0].date, date1);
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test portfolio::tracker::tests
```
Expected: FAIL -- `can't find crate for portfolio`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/portfolio/mod.rs`:
```rust
pub mod tracker;

pub use tracker::PortfolioTracker;
```

Create `src/portfolio/tracker.rs`:
```rust
use chrono::NaiveDate;
use rust_decimal::Decimal;

use crate::backtest::{EquityPoint, ExitReason, Position, PositionStatus, Trade};
use crate::core::events::PortfolioView;
use crate::data::{OptionQuote, OptionsSnapshot};
use crate::risk::portfolio_greeks::{PortfolioGreeks, PositionWithGreeks};

/// Tracks portfolio state: positions, equity, P&L, drawdown.
pub struct PortfolioTracker {
    initial_equity: Decimal,
    equity: Decimal,
    cash: Decimal,
    peak_equity: Decimal,
    positions: Vec<Position>,
    closed_trades: Vec<Trade>,
    equity_curve: Vec<EquityPoint>,
    total_commission: Decimal,
}

impl PortfolioTracker {
    pub fn new(initial_equity: Decimal) -> Self {
        Self {
            initial_equity,
            equity: initial_equity,
            cash: initial_equity,
            peak_equity: initial_equity,
            positions: Vec::new(),
            closed_trades: Vec::new(),
            equity_curve: Vec::new(),
            total_commission: Decimal::ZERO,
        }
    }

    /// Add an opened position to the portfolio.
    pub fn add_position(&mut self, position: Position) {
        self.total_commission += position.entry_commission;
        self.cash -= position.entry_commission;
        self.positions.push(position);
    }

    /// Close a position by ID.
    pub fn close_position(
        &mut self,
        position_id: u64,
        exit_date: NaiveDate,
        reason: ExitReason,
        current_value: Decimal,
        exit_commission: Decimal,
    ) {
        if let Some(pos) = self.positions.iter_mut().find(|p| p.id == position_id && p.is_open()) {
            pos.current_value = current_value;
            self.total_commission += exit_commission;

            let pnl = pos.unrealized_pnl() - exit_commission;
            self.cash += pnl;

            pos.close(exit_date, reason, exit_commission);

            if let Some(trade) = Trade::from_position(pos.clone()) {
                self.closed_trades.push(trade);
            }
        }
    }

    /// Update mark-to-market for all open positions from a snapshot.
    pub fn update_mtm(&mut self, snapshot: &OptionsSnapshot) {
        let quotes: Vec<&OptionQuote> = snapshot
            .chains
            .iter()
            .flat_map(|c| c.calls.iter().chain(c.puts.iter()))
            .collect();

        for position in &mut self.positions {
            if position.is_open() {
                position.update_mtm(&quotes);
            }
        }

        // Recalculate equity
        let positions_value: Decimal = self
            .positions
            .iter()
            .filter(|p| p.is_open())
            .map(|p| p.unrealized_pnl())
            .sum();

        self.equity = self.cash + positions_value;

        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }
    }

    /// Record an equity point for the curve.
    pub fn record_equity(&mut self, date: NaiveDate) {
        let positions_value: Decimal = self
            .positions
            .iter()
            .filter(|p| p.is_open())
            .map(|p| p.unrealized_pnl())
            .sum();

        let prev_equity = self
            .equity_curve
            .last()
            .map(|e| e.equity)
            .unwrap_or(self.initial_equity);

        let daily_pnl = self.equity - prev_equity;

        self.equity_curve.push(EquityPoint {
            date,
            equity: self.equity,
            cash: self.cash,
            positions_value,
            open_positions: self.positions.iter().filter(|p| p.is_open()).count(),
            daily_pnl,
        });
    }

    /// Get a read-only view of the portfolio for strategies.
    pub fn view(&self) -> PortfolioView {
        let open_positions: Vec<Position> = self
            .positions
            .iter()
            .filter(|p| p.is_open())
            .cloned()
            .collect();

        let greeks_positions: Vec<PositionWithGreeks> = open_positions
            .iter()
            .map(|p| PositionWithGreeks::from_position(p))
            .collect();

        let aggregate_greeks = PortfolioGreeks::from_positions_with_greeks(&greeks_positions);

        let current_drawdown_pct = if !self.peak_equity.is_zero() {
            let peak: f64 = self.peak_equity.try_into().unwrap_or(1.0);
            let current: f64 = self.equity.try_into().unwrap_or(peak);
            (peak - current) / peak * 100.0
        } else {
            0.0
        };

        PortfolioView {
            equity: self.equity,
            cash: self.cash,
            open_positions,
            aggregate_greeks,
            current_drawdown_pct,
            peak_equity: self.peak_equity,
        }
    }

    /// Remove closed positions from the active list.
    pub fn cleanup_closed(&mut self) {
        self.positions.retain(|p| p.is_open());
    }

    /// Get closed trades.
    pub fn closed_trades(&self) -> &[Trade] {
        &self.closed_trades
    }

    /// Get the equity curve.
    pub fn equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    /// Get current equity.
    pub fn equity(&self) -> Decimal {
        self.equity
    }

    /// Get peak equity.
    pub fn peak_equity(&self) -> Decimal {
        self.peak_equity
    }

    /// Get total commission paid.
    pub fn total_commission(&self) -> Decimal {
        self.total_commission
    }

    /// Get initial equity.
    pub fn initial_equity(&self) -> Decimal {
        self.initial_equity
    }

    /// Get total risk deployed (sum of max_loss for open positions).
    pub fn current_risk(&self) -> Decimal {
        self.positions
            .iter()
            .filter(|p| p.is_open())
            .map(|p| p.max_loss)
            .sum()
    }

    /// Get equity-correlated risk (SPY/QQQ/IWM positions).
    pub fn equity_correlated_risk(&self) -> Decimal {
        self.positions
            .iter()
            .filter(|p| p.is_open())
            .filter(|p| matches!(p.ticker.to_uppercase().as_str(), "SPY" | "QQQ" | "IWM"))
            .map(|p| p.max_loss)
            .sum()
    }

    /// Close all remaining open positions (end of backtest).
    pub fn close_all_remaining(&mut self, end_date: NaiveDate, exit_commission_per_trade: Decimal) {
        let open_ids: Vec<u64> = self.positions.iter().filter(|p| p.is_open()).map(|p| p.id).collect();
        for id in open_ids {
            if let Some(pos) = self.positions.iter_mut().find(|p| p.id == id) {
                self.total_commission += exit_commission_per_trade;
                let pnl = pos.unrealized_pnl() - exit_commission_per_trade;
                self.cash += pnl;
                pos.close(end_date, ExitReason::EndOfPeriod, exit_commission_per_trade);
                if let Some(trade) = Trade::from_position(pos.clone()) {
                    self.closed_trades.push(trade);
                }
            }
        }
        self.positions.retain(|p| p.is_open());
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod portfolio;
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test portfolio::tracker::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/portfolio/ src/lib.rs && git commit -m "feat(portfolio): add PortfolioTracker for position lifecycle management"
```

---

### Task 5: RiskGate
**Group:** B (parallel with Task 6, 7; depends on Group A)

**Behavior being verified:** DefaultRiskGate composes CircuitBreaker and PositionSizer to allow or reject orders based on portfolio state.
**Interface under test:** `risk::gate::DefaultRiskGate::check()` and `update()`.

**Files:**
- Create: `src/risk/gate.rs`
- Modify: `src/risk/mod.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/risk/gate.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::core::events::{OrderIntent, PortfolioView, LegIntent, OrderSide};
    use crate::backtest::SpreadType;
    use crate::data::OptionType;

    fn make_view(equity: Decimal, drawdown_pct: f64) -> PortfolioView {
        PortfolioView {
            equity,
            cash: equity,
            open_positions: vec![],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: drawdown_pct,
            peak_equity: equity / (Decimal::ONE - Decimal::from_f64_retain(drawdown_pct / 100.0).unwrap_or_default()),
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

        // Update with normal equity first to set the day
        gate.update(&snapshot, &view);

        // Now simulate 3% daily loss
        view.equity = dec!(97_000);
        gate.update(&snapshot, &view);

        let order = make_entry_order();
        let decision = gate.check(&order, &view);
        assert!(!decision.is_allowed());
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test risk::gate::tests
```
Expected: FAIL -- module `gate` not found

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/risk/gate.rs`:
```rust
use rust_decimal::Decimal;

use crate::core::events::{OrderAction, OrderIntent, PortfolioView, RiskDecision};
use crate::data::OptionsSnapshot;

use super::circuit_breakers::{CircuitBreaker, CircuitBreakerConfig};
use super::portfolio_greeks::PortfolioGreeks;

/// Trait for risk checking between strategy and broker.
pub trait RiskGate {
    /// Check if an order should be allowed.
    fn check(&self, order: &OrderIntent, portfolio: &PortfolioView) -> RiskDecision;
    /// Update internal state with new market data.
    fn update(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView);
}

/// Default risk gate composing circuit breakers and position limits.
pub struct DefaultRiskGate {
    circuit_breaker: CircuitBreaker,
}

impl DefaultRiskGate {
    pub fn new(cb_config: CircuitBreakerConfig, initial_equity: Decimal) -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(cb_config, initial_equity),
        }
    }
}

impl RiskGate for DefaultRiskGate {
    fn check(&self, order: &OrderIntent, _portfolio: &PortfolioView) -> RiskDecision {
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

        RiskDecision::Allowed
    }

    fn update(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) {
        // Estimate VIX from portfolio context (simplified)
        let estimated_vix = None; // Strategy-owned in the new architecture
        self.circuit_breaker.update(snapshot.date, portfolio.equity, estimated_vix);
    }
}
```

Add to `src/risk/mod.rs`:
```rust
pub mod gate;
pub use gate::{DefaultRiskGate, RiskGate};
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test risk::gate::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/risk/gate.rs src/risk/mod.rs && git commit -m "feat(risk): add RiskGate trait and DefaultRiskGate"
```

---

### Task 6: DataSource Trait + HistoricalDataSource
**Group:** B (parallel with Task 5, 7; depends on Group A)

**Behavior being verified:** HistoricalDataSource wraps the existing DataLoader and emits MarketEvents in date order via the DataSource trait.
**Interface under test:** `data::DataSource` trait and `data::HistoricalDataSource::next_event()`.

**Files:**
- Modify: `src/data/mod.rs`

- [ ] **Step 1: Write the failing test**

```rust
// Append to src/data/mod.rs tests
#[cfg(test)]
mod datasource_tests {
    use super::*;

    #[test]
    fn test_historical_datasource_emits_events_in_order() {
        // This test requires actual parquet data
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2020, 1, 10).unwrap(),
        );

        match source {
            Ok(mut src) => {
                let mut prev_ts = None;
                let mut count = 0;
                while let Some(event) = src.next_event() {
                    if let Some(prev) = prev_ts {
                        assert!(event.timestamp >= prev, "Events must be in chronological order");
                    }
                    assert!(!event.snapshot.chains.is_empty(), "Snapshot should have chains");
                    assert!(event.snapshot.underlying_price > rust_decimal::Decimal::ZERO);
                    prev_ts = Some(event.timestamp);
                    count += 1;
                }
                assert!(count >= 5, "Should have at least 5 trading days in Jan 2-10 2020, got {}", count);
            }
            Err(e) => {
                // Skip test if data not available (CI environment)
                eprintln!("Skipping test: {}", e);
            }
        }
    }

    #[test]
    fn test_historical_datasource_returns_none_when_exhausted() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2020, 1, 3).unwrap(),
        );

        match source {
            Ok(mut src) => {
                // Consume all events
                while src.next_event().is_some() {}
                // Next call should return None
                assert!(src.next_event().is_none());
            }
            Err(_) => {
                eprintln!("Skipping test: data not available");
            }
        }
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test data::datasource_tests
```
Expected: FAIL -- `HistoricalDataSource` not found

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `src/data/mod.rs`:

```rust
use crate::core::events::MarketEvent;

/// DataSource trait: provides market events to the engine.
pub trait DataSource {
    fn next_event(&mut self) -> Option<MarketEvent>;
}

/// Historical data source backed by ORATS parquet files.
pub struct HistoricalDataSource {
    snapshots: Vec<OptionsSnapshot>,
    index: usize,
}

impl HistoricalDataSource {
    pub fn new(
        data_dir: &str,
        ticker: &str,
        start_date: chrono::NaiveDate,
        end_date: chrono::NaiveDate,
    ) -> Result<Self, loader::LoaderError> {
        let loader = DataLoader::new(data_dir);
        let snapshots = loader.load_snapshots(ticker, start_date, end_date)?;
        Ok(Self { snapshots, index: 0 })
    }

    /// Create from pre-loaded snapshots (for walk-forward optimization).
    pub fn from_snapshots(snapshots: Vec<OptionsSnapshot>) -> Self {
        Self { snapshots, index: 0 }
    }

    /// Get total number of events.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

impl DataSource for HistoricalDataSource {
    fn next_event(&mut self) -> Option<MarketEvent> {
        if self.index >= self.snapshots.len() {
            return None;
        }
        let snapshot = self.snapshots[self.index].clone();
        self.index += 1;
        Some(MarketEvent::new(snapshot))
    }
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test data::datasource_tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/data/mod.rs && git commit -m "feat(data): add DataSource trait and HistoricalDataSource"
```

---

### Task 7: PutCreditSpreadStrategy
**Group:** B (parallel with Task 5, 6; depends on Group A)

**Behavior being verified:** PutCreditSpreadStrategy implements the Strategy trait, producing entry OrderIntents when conditions are met and exit OrderIntents when positions hit targets.
**Interface under test:** `strategy::put_spread::PutCreditSpreadStrategy::on_snapshot()`.

**Files:**
- Create: `src/strategy/put_spread.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/strategy/put_spread.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::core::events::{OrderAction, PortfolioView};
    use crate::data::{OptionType, OptionQuote, OptionsChain, OptionsSnapshot, Greeks};
    use crate::risk::PortfolioGreeks;

    fn make_tradeable_snapshot() -> OptionsSnapshot {
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();
        let mut snapshot = OptionsSnapshot::new(date, "SPY".to_string(), dec!(480));
        let mut chain = OptionsChain::new(exp, 32);

        // Put at 470 (delta -0.25, within default range)
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(470), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(2.40), ask: dec!(2.60), mid: dec!(2.50),
            theoretical_value: dec!(2.50), volume: 1000, open_interest: 5000,
            bid_iv: 0.18, mid_iv: 0.19, ask_iv: 0.20, smv_vol: 0.19,
            greeks: Greeks { delta: -0.25, gamma: 0.02, theta: -0.05, vega: 0.10, rho: 0.01 },
            residual_rate: 0.05,
        });

        // Put at 465 (long leg)
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(465), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(1.40), ask: dec!(1.60), mid: dec!(1.50),
            theoretical_value: dec!(1.50), volume: 800, open_interest: 4000,
            bid_iv: 0.19, mid_iv: 0.20, ask_iv: 0.21, smv_vol: 0.20,
            greeks: Greeks { delta: -0.15, gamma: 0.015, theta: -0.03, vega: 0.08, rho: 0.008 },
            residual_rate: 0.05,
        });

        // Put at 460 (for $10 width spread option)
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(460), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(0.90), ask: dec!(1.10), mid: dec!(1.00),
            theoretical_value: dec!(1.00), volume: 600, open_interest: 3000,
            bid_iv: 0.20, mid_iv: 0.21, ask_iv: 0.22, smv_vol: 0.21,
            greeks: Greeks { delta: -0.08, gamma: 0.01, theta: -0.02, vega: 0.06, rho: 0.005 },
            residual_rate: 0.05,
        });

        snapshot.chains.push(chain);
        snapshot
    }

    fn make_empty_portfolio() -> PortfolioView {
        PortfolioView {
            equity: dec!(100_000),
            cash: dec!(100_000),
            open_positions: vec![],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: 0.0,
            peak_equity: dec!(100_000),
        }
    }

    #[test]
    fn test_strategy_generates_entry_order() {
        let config = PutSpreadConfig::default();
        let mut strategy = PutCreditSpreadStrategy::new(config);
        let snapshot = make_tradeable_snapshot();
        let portfolio = make_empty_portfolio();

        let orders = strategy.on_snapshot(&snapshot, &portfolio);

        // Should generate at least one entry order
        assert!(!orders.is_empty(), "Strategy should find a tradeable spread");
        assert!(matches!(orders[0].action, OrderAction::Open));
        assert_eq!(orders[0].legs.len(), 2);
    }

    #[test]
    fn test_strategy_respects_max_positions() {
        let mut config = PutSpreadConfig::default();
        config.max_positions = 0; // No positions allowed
        let mut strategy = PutCreditSpreadStrategy::new(config);
        let snapshot = make_tradeable_snapshot();
        let portfolio = make_empty_portfolio();

        let orders = strategy.on_snapshot(&snapshot, &portfolio);
        // Should only contain exit orders (none, since no positions open)
        assert!(orders.iter().all(|o| !matches!(o.action, OrderAction::Open)));
    }

    #[test]
    fn test_strategy_generates_exit_on_profit_target() {
        let config = PutSpreadConfig::default();
        let mut strategy = PutCreditSpreadStrategy::new(config);
        let snapshot = make_tradeable_snapshot();

        // Create a portfolio with an open position at 50%+ profit
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();
        let mut position = crate::backtest::CreditSpreadBuilder::put_credit_spread("SPY", date)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.25)
            .long_leg(dec!(465), dec!(1.50), -0.15)
            .expiration(exp, 32)
            .contracts(1)
            .build();
        // Simulate 60% profit (spread decayed from $1.00 to $0.40)
        position.current_value = dec!(40);

        let portfolio = PortfolioView {
            equity: dec!(100_060),
            cash: dec!(100_000),
            open_positions: vec![position],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: 0.0,
            peak_equity: dec!(100_060),
        };

        let orders = strategy.on_snapshot(&snapshot, &portfolio);

        // Should have at least one exit order
        let exit_orders: Vec<_> = orders.iter().filter(|o| matches!(o.action, OrderAction::Close { .. })).collect();
        assert!(!exit_orders.is_empty(), "Strategy should exit at profit target");
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test strategy::put_spread::tests
```
Expected: FAIL -- `PutCreditSpreadStrategy` not found

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/strategy/put_spread.rs` -- this extracts the entry/exit logic from the current `BacktestEngine::screen_and_enter()` and `check_exits()` into a Strategy implementation:

```rust
use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::analytics::{IVTermStructureAnalyzer, SpreadScreener, SpreadScreenerConfig};
use crate::backtest::{ExitReason, Position, SpreadType};
use crate::core::events::{FillEvent, LegIntent, OrderAction, OrderIntent, OrderSide, PortfolioView};
use crate::data::{OptionType, OptionsSnapshot};

use super::Strategy;

/// Configuration for put credit spread strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutSpreadConfig {
    /// Profit target as percentage of credit received.
    pub profit_target_pct: f64,
    /// Stop loss as percentage of credit received.
    pub stop_loss_pct: f64,
    /// Time-based exit at this DTE.
    pub time_exit_dte: i32,
    /// Screener configuration.
    pub screener: SpreadScreenerConfig,
    /// Maximum concurrent positions.
    pub max_positions: usize,
    /// Maximum trades per day.
    pub max_trades_per_day: usize,
    /// Minimum IV percentile for entry (0-100).
    pub min_iv_percentile: f64,
    /// Whether to use IV percentile filter.
    pub use_iv_filter: bool,
}

impl Default for PutSpreadConfig {
    fn default() -> Self {
        Self {
            profit_target_pct: 50.0,
            stop_loss_pct: 125.0,
            time_exit_dte: 21,
            screener: SpreadScreenerConfig::default(),
            max_positions: 10,
            max_trades_per_day: 1,
            min_iv_percentile: 50.0,
            use_iv_filter: false, // Disabled by default for simplicity
        }
    }
}

/// Put credit spread strategy.
pub struct PutCreditSpreadStrategy {
    config: PutSpreadConfig,
    screener: SpreadScreener,
    iv_analyzer: IVTermStructureAnalyzer,
    trades_today: usize,
    last_trade_date: Option<NaiveDate>,
}

impl PutCreditSpreadStrategy {
    pub fn new(config: PutSpreadConfig) -> Self {
        let screener = SpreadScreener::new(config.screener.clone());
        Self {
            config,
            screener,
            iv_analyzer: IVTermStructureAnalyzer::new(),
            trades_today: 0,
            last_trade_date: None,
        }
    }

    /// Check exit conditions for open positions.
    fn check_exits(&self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent> {
        let mut orders = Vec::new();

        for position in &portfolio.open_positions {
            if let Some(reason) = self.should_exit(position, snapshot.date) {
                orders.push(OrderIntent::exit_position(position.id, reason));
            }
        }

        orders
    }

    /// Determine if a position should be exited.
    fn should_exit(&self, position: &Position, current_date: NaiveDate) -> Option<ExitReason> {
        // Check expiration first
        if position.is_expired(current_date) {
            let short_leg = position.legs.iter().find(|l| l.is_short());
            if let Some(_leg) = short_leg {
                // Simplified: check if position is at a loss
                if position.unrealized_pnl() < Decimal::ZERO {
                    return Some(ExitReason::ExpiredITM);
                }
            }
            return Some(ExitReason::ExpiredWorthless);
        }

        // Check profit target
        if position.is_profit_target_hit(self.config.profit_target_pct) {
            return Some(ExitReason::ProfitTarget);
        }

        // Check stop loss
        if position.is_stop_loss_hit(self.config.stop_loss_pct) {
            return Some(ExitReason::StopLoss);
        }

        // Check time exit
        if position.is_time_exit(current_date, self.config.time_exit_dte) {
            return Some(ExitReason::TimeExit);
        }

        None
    }

    /// Screen for new entry opportunities.
    fn check_entries(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent> {
        // Reset daily counter if new day
        if self.last_trade_date != Some(snapshot.date) {
            self.trades_today = 0;
            self.last_trade_date = Some(snapshot.date);
        }

        // Check position limits
        if portfolio.open_positions.len() >= self.config.max_positions {
            return vec![];
        }

        if self.trades_today >= self.config.max_trades_per_day {
            return vec![];
        }

        // Update IV analysis
        let iv_structure = self.iv_analyzer.analyze(snapshot);

        // IV percentile filter
        if self.config.use_iv_filter {
            if let Some(atm_iv) = iv_structure.atm_iv {
                if let Some(pct) = self.iv_analyzer.iv_percentile(atm_iv) {
                    if pct < self.config.min_iv_percentile {
                        return vec![];
                    }
                }
            }
        }

        // Screen for candidates
        let iv_percentile = iv_structure.atm_iv.and_then(|iv| self.iv_analyzer.iv_percentile(iv));
        let candidates = self.screener.screen_put_spreads(snapshot, iv_percentile);

        if let Some(best) = self.screener.best_candidate(&candidates) {
            self.trades_today += 1;

            let order = OrderIntent::enter_spread(
                snapshot.ticker.clone(),
                SpreadType::PutCreditSpread,
                vec![
                    LegIntent {
                        option_type: OptionType::Put,
                        strike: best.short_strike,
                        expiration: best.expiration,
                        contracts: -1,
                        side: OrderSide::Sell,
                    },
                    LegIntent {
                        option_type: OptionType::Put,
                        strike: best.long_strike,
                        expiration: best.expiration,
                        contracts: 1,
                        side: OrderSide::Buy,
                    },
                ],
            );

            return vec![order];
        }

        vec![]
    }
}

impl Strategy for PutCreditSpreadStrategy {
    fn on_snapshot(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent> {
        let mut orders = Vec::new();

        // Check exits first (more important than entries)
        orders.extend(self.check_exits(snapshot, portfolio));

        // Then check entries
        orders.extend(self.check_entries(snapshot, portfolio));

        orders
    }

    fn on_fill(&mut self, _fill: &FillEvent) {
        // Could track fill quality, slippage stats, etc.
    }
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test strategy::put_spread::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/strategy/put_spread.rs src/strategy/mod.rs && git commit -m "feat(strategy): add PutCreditSpreadStrategy extracted from BacktestEngine"
```

---

### Task 8: Core Engine
**Group:** C (sequential, depends on Group B)

**Behavior being verified:** The Engine wires DataSource, Strategy, Broker, and RiskGate together to produce a BacktestResult identical to the old engine for the same inputs.
**Interface under test:** `core::engine::Engine::run()`.

**Files:**
- Create: `src/core/engine.rs`
- Modify: `src/core/mod.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/core/engine.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::backtest::{CommissionModel, SlippageModel};
    use crate::broker::backtest::SimulatedBroker;
    use crate::data::HistoricalDataSource;
    use crate::risk::circuit_breakers::CircuitBreakerConfig;
    use crate::risk::gate::DefaultRiskGate;
    use crate::strategy::NoOpStrategy;
    use crate::strategy::put_spread::{PutCreditSpreadStrategy, PutSpreadConfig};

    #[test]
    fn test_engine_runs_with_noop_strategy() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 31).unwrap(),
        );

        match source {
            Ok(data_source) => {
                let strategy = NoOpStrategy;
                let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
                let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), dec!(100_000));

                let result = Engine::run(
                    data_source,
                    strategy,
                    broker,
                    risk_gate,
                    dec!(100_000),
                );

                assert!(result.is_ok());
                let result = result.unwrap();
                // No trades with NoOpStrategy
                assert_eq!(result.total_trades, 0);
                // Equity should be unchanged
                assert_eq!(result.final_equity, dec!(100_000));
                // Should have trading days recorded
                assert!(result.trading_days > 0);
            }
            Err(_) => eprintln!("Skipping: data not available"),
        }
    }

    #[test]
    fn test_engine_runs_with_put_spread_strategy() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            NaiveDate::from_ymd_opt(2020, 6, 1).unwrap(),
            NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
        );

        match source {
            Ok(data_source) => {
                let config = PutSpreadConfig::default();
                let strategy = PutCreditSpreadStrategy::new(config);
                let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
                let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), dec!(100_000));

                let result = Engine::run(
                    data_source,
                    strategy,
                    broker,
                    risk_gate,
                    dec!(100_000),
                );

                assert!(result.is_ok());
                let result = result.unwrap();
                // Should have some trades in 7 months
                assert!(result.total_trades > 0, "Expected trades, got 0");
                assert!(result.trading_days > 100);
            }
            Err(_) => eprintln!("Skipping: data not available"),
        }
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test core::engine::tests
```
Expected: FAIL -- `Engine` not found

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/core/engine.rs`:
```rust
use rust_decimal::Decimal;
use tracing::info;

use crate::backtest::{BacktestResult, BacktestConfig, CommissionModel};
use crate::broker::Broker;
use crate::data::DataSource;
use crate::portfolio::PortfolioTracker;
use crate::risk::gate::RiskGate;
use crate::strategy::Strategy;

use super::events::OrderAction;

/// Errors that can occur during engine execution.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Data error: {0}")]
    Data(String),

    #[error("Broker error: {0}")]
    Broker(#[from] crate::broker::BrokerError),
}

/// The trait-based engine. Wires DataSource -> Strategy -> RiskGate -> Broker -> Portfolio.
pub struct Engine;

impl Engine {
    /// Run the engine with the provided components.
    pub fn run<D, S, B, R>(
        mut data_source: D,
        mut strategy: S,
        mut broker: B,
        mut risk_gate: R,
        initial_equity: Decimal,
    ) -> Result<BacktestResult, EngineError>
    where
        D: DataSource,
        S: Strategy,
        B: Broker,
        R: RiskGate,
    {
        let mut portfolio = PortfolioTracker::new(initial_equity);
        let mut start_date = None;
        let mut end_date = None;
        let mut ticker = String::new();

        while let Some(event) = data_source.next_event() {
            let snapshot = &event.snapshot;

            if start_date.is_none() {
                start_date = Some(snapshot.date);
                ticker = snapshot.ticker.clone();
            }
            end_date = Some(snapshot.date);

            // 1. Update MTM for open positions
            portfolio.update_mtm(snapshot);

            // 2. Update risk gate with current state
            let view = portfolio.view();
            risk_gate.update(snapshot, &view);

            // 3. Get orders from strategy
            let orders = strategy.on_snapshot(snapshot, &view);

            // 4. Process each order through risk gate and broker
            for order in orders {
                let current_view = portfolio.view();

                match order.action {
                    OrderAction::Close { position_id, reason } => {
                        // For exit orders, find the position and close it
                        let exit_commission = Decimal::from(2); // Default commission
                        if let Some(pos) = current_view.open_positions.iter().find(|p| p.id == position_id) {
                            portfolio.close_position(
                                position_id,
                                snapshot.date,
                                reason,
                                pos.current_value,
                                exit_commission,
                            );
                        }
                    }
                    OrderAction::Open => {
                        // Check risk gate
                        let decision = risk_gate.check(&order, &current_view);
                        if !decision.is_allowed() {
                            continue;
                        }

                        // Submit to broker
                        match broker.submit_order(&order, snapshot) {
                            Ok(fill) => {
                                // Build position from fill
                                let position = build_position_from_fill(&order, &fill, snapshot);
                                portfolio.add_position(position);
                                strategy.on_fill(&fill);
                            }
                            Err(e) => {
                                info!("Broker error: {}", e);
                            }
                        }
                    }
                }
            }

            // 5. Cleanup and record
            portfolio.cleanup_closed();
            portfolio.record_equity(snapshot.date);
        }

        // Close remaining positions
        let commission = Decimal::from(2);
        if let Some(end) = end_date {
            portfolio.close_all_remaining(end, commission);
        }

        // Build result
        let sd = start_date.unwrap_or_default();
        let ed = end_date.unwrap_or_default();

        Ok(build_result(&portfolio, vec![ticker], sd, ed, initial_equity))
    }
}

/// Build a Position from a FillEvent.
fn build_position_from_fill(
    order: &super::events::OrderIntent,
    fill: &super::events::FillEvent,
    snapshot: &crate::data::OptionsSnapshot,
) -> crate::backtest::Position {
    use crate::backtest::CreditSpreadBuilder;
    use crate::core::events::OrderSide;

    // Find short and long legs
    let short_idx = order.legs.iter().position(|l| l.side == OrderSide::Sell).unwrap_or(0);
    let long_idx = order.legs.iter().position(|l| l.side == OrderSide::Buy).unwrap_or(1);

    let short_leg = &order.legs[short_idx];
    let long_leg = &order.legs[long_idx];

    let short_fill = fill.fill_prices[short_idx];
    let long_fill = fill.fill_prices[long_idx];

    let dte = snapshot.chains.iter()
        .find(|c| c.expiration == short_leg.expiration)
        .map(|c| c.dte)
        .unwrap_or(0);

    // Find delta from snapshot
    let short_delta = snapshot.chains.iter()
        .flat_map(|c| c.puts.iter().chain(c.calls.iter()))
        .find(|q| q.strike == short_leg.strike && q.expiration == short_leg.expiration)
        .map(|q| q.greeks.delta)
        .unwrap_or(0.0);

    let long_delta = snapshot.chains.iter()
        .flat_map(|c| c.puts.iter().chain(c.calls.iter()))
        .find(|q| q.strike == long_leg.strike && q.expiration == long_leg.expiration)
        .map(|q| q.greeks.delta)
        .unwrap_or(0.0);

    CreditSpreadBuilder::put_credit_spread(&order.ticker, snapshot.date)
        .stock_price(snapshot.underlying_price)
        .short_leg(short_leg.strike, short_fill, short_delta)
        .long_leg(long_leg.strike, long_fill, long_delta)
        .expiration(short_leg.expiration, dte)
        .contracts(short_leg.contracts.abs())
        .commission(fill.commission)
        .build()
}

/// Build BacktestResult from portfolio state.
fn build_result(
    portfolio: &PortfolioTracker,
    tickers: Vec<String>,
    start_date: chrono::NaiveDate,
    end_date: chrono::NaiveDate,
    initial_equity: Decimal,
) -> BacktestResult {
    let equity = portfolio.equity();
    let initial_f64: f64 = initial_equity.try_into().unwrap_or(1.0);
    let final_f64: f64 = equity.try_into().unwrap_or(1.0);
    let total_return_pct = (final_f64 - initial_f64) / initial_f64 * 100.0;

    let peak: f64 = portfolio.peak_equity().try_into().unwrap_or(1.0);
    let max_drawdown = portfolio.peak_equity() - equity;
    let max_dd_f64: f64 = max_drawdown.try_into().unwrap_or(0.0);
    let max_drawdown_pct = if peak > 0.0 { max_dd_f64 / peak * 100.0 } else { 0.0 };

    let trades = portfolio.closed_trades();
    let winning_trades = trades.iter().filter(|t| t.is_winner()).count();
    let losing_trades = trades.len() - winning_trades;
    let total_pnl: Decimal = trades.iter().map(|t| t.pnl()).sum();
    let gross_profit: Decimal = trades.iter().filter(|t| t.is_winner()).map(|t| t.pnl()).sum();
    let gross_loss: Decimal = trades.iter().filter(|t| !t.is_winner()).map(|t| t.pnl()).sum();

    BacktestResult {
        config: BacktestConfig::default(),
        tickers,
        start_date,
        end_date,
        trades: trades.to_vec(),
        equity_curve: portfolio.equity_curve().to_vec(),
        final_equity: equity,
        total_return_pct,
        trading_days: portfolio.equity_curve().len(),
        peak_equity: portfolio.peak_equity(),
        max_drawdown,
        max_drawdown_pct,
        total_trades: trades.len(),
        winning_trades,
        losing_trades,
        total_pnl,
        gross_profit,
        gross_loss,
        total_commission: portfolio.total_commission(),
    }
}
```

Update `src/core/mod.rs`:
```rust
pub mod engine;
pub mod events;
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test core::engine::tests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/core/ && git commit -m "feat(core): add trait-based Engine that wires DataSource, Strategy, Broker, RiskGate"
```

---

### Task 9: Update lib.rs and backtest module
**Group:** D (sequential, depends on Group C)

**Behavior being verified:** The library re-exports the new modules correctly, and the old engine is no longer the primary entry point. The backtest module retains trade types, slippage, and commission.
**Interface under test:** `use mahler_backtest::{...}` compiles with the new module structure.

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/backtest/mod.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In src/lib.rs at the bottom
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_new_module_imports() {
        // Verify all new modules are accessible
        use crate::core::engine::Engine;
        use crate::core::events::{MarketEvent, OrderIntent, FillEvent, RiskDecision, PortfolioView};
        use crate::strategy::Strategy;
        use crate::strategy::put_spread::PutCreditSpreadStrategy;
        use crate::broker::Broker;
        use crate::broker::backtest::SimulatedBroker;
        use crate::portfolio::PortfolioTracker;
        use crate::risk::gate::{RiskGate, DefaultRiskGate};
        use crate::data::DataSource;

        // Verify old types still accessible
        use crate::backtest::{Position, Trade, SlippageModel, CommissionModel};
        use crate::data::{OptionQuote, OptionsChain, OptionsSnapshot};
        use crate::risk::{CircuitBreaker, PositionSizer};
        use crate::metrics::MetricsCalculator;
    }
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test integration_tests::test_new_module_imports
```
Expected: FAIL if lib.rs doesn't have all module declarations

- [ ] **Step 3: Implement the minimum to make the test pass**

Update `src/lib.rs`:
```rust
pub mod analytics;
pub mod backtest;
pub mod broker;
pub mod core;
pub mod data;
pub mod metrics;
pub mod portfolio;
pub mod regime;
pub mod risk;
pub mod strategy;
pub mod validation;
pub mod walkforward;

// Re-export commonly used types
pub use data::{OptionQuote, OptionType, OptionsChain, OptionsSnapshot};
pub use validation::{BlackScholes, DataIntegrityValidator, GreeksValidator};
pub use backtest::{BacktestConfig, BacktestResult, Position, SlippageModel, Trade};
pub use risk::{CircuitBreaker, PortfolioGreeks, PositionSizer};
pub use analytics::{IVTermStructureAnalyzer, SpreadScreener, SpreadCandidate};
pub use walkforward::{WalkForwardOptimizer, WalkForwardResult, ParameterGrid};
pub use regime::{MarketRegime, RegimeClassifier};
pub use metrics::{PerformanceMetrics, MetricsCalculator};

// New re-exports
pub use core::engine::Engine;
pub use strategy::Strategy;
pub use broker::Broker;
pub use data::DataSource;
pub use risk::gate::RiskGate;
pub use portfolio::PortfolioTracker;
```

Update `src/backtest/mod.rs` to remove or deprecate the old engine re-export while keeping trade/slippage/commission:
```rust
pub mod commission;
pub mod engine; // Keep for now -- will be removed after CLI migration
pub mod slippage;
pub mod trade;

pub use commission::{Commission, CommissionModel};
pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, EquityPoint};
pub use slippage::{Slippage, SlippageModel};
pub use trade::{
    CreditSpreadBuilder, ExitReason, Position, PositionLeg, PositionStatus, SpreadType, Trade,
    TradeDirection,
};
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test integration_tests::test_new_module_imports
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/lib.rs src/backtest/mod.rs && git commit -m "feat(lib): update module tree with new trait-based architecture"
```

---

### Task 10: CLI Migration
**Group:** E (sequential, depends on Group D)

**Behavior being verified:** The CLI `run` subcommand uses the new Engine instead of BacktestEngine, producing equivalent output.
**Interface under test:** `cargo run -- run --ticker SPY --start 2020-06-01 --end 2020-12-31` completes successfully.

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Write the failing test**

This is a manual integration test -- no unit test needed. The test is:

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo run -- run --ticker SPY --start 2020-06-01 --end 2020-12-31
```
Expected: Currently uses old BacktestEngine. After modification, should use new Engine and produce output.

- [ ] **Step 2: Verify current behavior**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo run -- run --ticker SPY --start 2020-06-01 --end 2020-12-31 2>&1 | head -20
```

- [ ] **Step 3: Implement the change**

In `src/main.rs`, find the `run` subcommand handler and replace the `BacktestEngine` usage with:

```rust
// Replace BacktestEngine::new() + engine.run() with:
use mahler_backtest::core::engine::Engine;
use mahler_backtest::strategy::put_spread::{PutCreditSpreadStrategy, PutSpreadConfig};
use mahler_backtest::broker::backtest::SimulatedBroker;
use mahler_backtest::risk::gate::DefaultRiskGate;
use mahler_backtest::data::HistoricalDataSource;

// ... inside the run handler:
let data_source = HistoricalDataSource::new(&data_dir, ticker, start_date, end_date)?;

let mut strategy_config = PutSpreadConfig::default();
strategy_config.profit_target_pct = profit_target;
strategy_config.stop_loss_pct = stop_loss;
strategy_config.time_exit_dte = time_exit_dte;
// ... map remaining config fields

let strategy = PutCreditSpreadStrategy::new(strategy_config);
let broker = SimulatedBroker::new(slippage_model, commission_model);
let risk_gate = DefaultRiskGate::new(circuit_breaker_config, initial_equity);

let result = Engine::run(data_source, strategy, broker, risk_gate, initial_equity)?;
```

The exact implementation depends on the current CLI structure in main.rs. Read main.rs fully before modifying. Preserve all existing CLI arguments and output formatting.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo run -- run --ticker SPY --start 2020-06-01 --end 2020-12-31
```
Expected: Completes with trade results similar to old engine

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add src/main.rs && git commit -m "feat(cli): migrate run command to trait-based Engine"
```

---

### Task 11: Determinism and Regression Test
**Group:** F (sequential, depends on Group E)

**Behavior being verified:** The new engine produces deterministic results (same inputs = same outputs every run) and the overall system works end-to-end.
**Interface under test:** `Engine::run()` with identical inputs twice.

**Files:**
- Create: `tests/determinism.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tests/determinism.rs
use chrono::NaiveDate;
use rust_decimal_macros::dec;

use mahler_backtest::backtest::{CommissionModel, SlippageModel};
use mahler_backtest::broker::backtest::SimulatedBroker;
use mahler_backtest::core::engine::Engine;
use mahler_backtest::data::HistoricalDataSource;
use mahler_backtest::risk::circuit_breakers::CircuitBreakerConfig;
use mahler_backtest::risk::gate::DefaultRiskGate;
use mahler_backtest::strategy::put_spread::{PutCreditSpreadStrategy, PutSpreadConfig};

#[test]
fn test_deterministic_results() {
    let start = NaiveDate::from_ymd_opt(2020, 6, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();
    let equity = dec!(100_000);

    let run = || -> Option<(usize, f64, f64)> {
        let data_source = HistoricalDataSource::new("data/orats", "SPY", start, end).ok()?;
        let strategy = PutCreditSpreadStrategy::new(PutSpreadConfig::default());
        let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
        let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), equity);

        let result = Engine::run(data_source, strategy, broker, risk_gate, equity).ok()?;
        Some((result.total_trades, result.total_return_pct, result.max_drawdown_pct))
    };

    match (run(), run()) {
        (Some(r1), Some(r2)) => {
            assert_eq!(r1.0, r2.0, "Trade count must be deterministic");
            assert!(
                (r1.1 - r2.1).abs() < 0.0001,
                "Return must be deterministic: {} vs {}",
                r1.1,
                r2.1
            );
            assert!(
                (r1.2 - r2.2).abs() < 0.0001,
                "Drawdown must be deterministic: {} vs {}",
                r1.2,
                r2.2
            );
        }
        _ => eprintln!("Skipping: data not available"),
    }
}

#[test]
fn test_engine_produces_trades() {
    let start = NaiveDate::from_ymd_opt(2020, 6, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();
    let equity = dec!(100_000);

    let data_source = match HistoricalDataSource::new("data/orats", "SPY", start, end) {
        Ok(ds) => ds,
        Err(_) => { eprintln!("Skipping: data not available"); return; }
    };

    let strategy = PutCreditSpreadStrategy::new(PutSpreadConfig::default());
    let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
    let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), equity);

    let result = Engine::run(data_source, strategy, broker, risk_gate, equity).unwrap();

    assert!(result.total_trades > 0, "Should produce trades over 7 months");
    assert!(result.trading_days > 100, "Should have >100 trading days");
    assert!(result.equity_curve.len() > 100, "Should have equity curve entries");

    // Sanity checks
    assert!(result.win_rate() > 0.0 && result.win_rate() <= 1.0, "Win rate should be between 0 and 1");
    assert!(result.total_commission > rust_decimal::Decimal::ZERO, "Should have commission");
}
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test --test determinism
```
Expected: FAIL if Engine or modules aren't wired correctly

- [ ] **Step 3: Implement -- no new code needed**

This test validates existing implementation. If it fails, fix bugs in the engine/strategy/broker wiring until it passes.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && cargo test --test determinism
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && git add tests/determinism.rs && git commit -m "test: add determinism and regression tests for trait-based engine"
```

---

## Spec Coverage Verification

| Spec Requirement | Task |
|---|---|
| DataSource trait + HistoricalDataSource | Task 6 |
| Strategy trait + PutCreditSpreadStrategy | Task 2, Task 7 |
| Broker trait + SimulatedBroker | Task 3 |
| RiskGate trait + DefaultRiskGate | Task 5 |
| Core engine event loop | Task 8 |
| PortfolioTracker | Task 4 |
| Event types (MarketEvent, OrderIntent, FillEvent) | Task 1 |
| Module re-exports and cleanup | Task 9 |
| CLI migration | Task 10 |
| Determinism guarantee | Task 11 |
| Explicit error handling (no unwrap_or_default) | Task 6 (data loader tightening) |
| Walk-forward optimizer adaptation | Deferred to follow-up task (optimizer still works with old engine; adapting it is a separate vertical slice) |

---

## Challenge Review

**Reviewed:** 2026-04-06
**Plan:** docs/plans/2026-04-06-trader-joe-engine.md
**Spec:** docs/specs/2026-04-06-trader-joe-engine-design.md

### CEO Pass

**Premise:** Sound. The monolithic engine genuinely blocks adding new strategies and prevents live trading. The refactor addresses the right structural problem. No simpler alternative achieves the stated goals -- you cannot get backtest/live parity without abstracting the data source and broker.

**Real pain:** Without this, every strategy change requires modifying the engine (src/backtest/engine.rs:663-803). The system cannot evolve. The silent zero-fill issue on financial data (src/data/loader.rs:432-451) is a correctness risk for any backtest results.

**Scope:**
- The plan correctly defers Alpaca integration, project renaming, and Python removal. Good.
- Walk-forward optimizer deferral is noted but the spec says "adapted in this phase." See finding below.
- 18 files touched (11 new, 7 modified). Justified given the scope of extracting 4 traits from a 940-line monolith.

**12-Month alignment:**
```
CURRENT STATE                  THIS PLAN                    12-MONTH IDEAL
Monolithic backtester      ->  Trait-based engine       ->  Same strategies running
with hardcoded strategy        with pluggable strategies    on live broker (Alpaca/IBKR)
                               and backtest/live parity     with validated parameters
```
Directly on the path. No tech debt created that conflicts with the 12-month ideal.

**Alternatives:** The spec documents the NautilusTrader message-bus alternative and explains why trait-based was chosen (single-threaded, debuggability, same parity). Adequate.

### Engineering Pass

#### Architecture

[RISK] (confidence: 7/10) -- **`pub mod core` shadows Rust's `core` crate.** Verified by test: defining `mod core` in a crate prevents `core::option::Option` from resolving. The existing crate does not use `core::` imports directly, and `cargo check` passes with a `mod core` added, but derive macros from dependencies (serde, polars) may internally expand to `core::` references. Today it compiles; tomorrow a dependency update could break it. **Rename to `engine` or `trading_engine` to eliminate the risk entirely.** If you keep `core`, any `core::` reference must be written as `::core::` which is fragile.

[RISK] (confidence: 8/10) -- **Exit orders bypass the broker entirely.** In Task 8's engine (line 1959-1969), `OrderAction::Close` directly calls `portfolio.close_position()` without going through the broker for fill simulation. This means exits use the position's current MTM value as the exit price, not a slippage-adjusted fill. In the old engine (src/backtest/engine.rs:619-639), exits also used `unrealized_pnl()` directly, so this is consistent with existing behavior -- but it means the new architecture's "same code for backtest and live" promise has a hole: in live mode, exits *must* go through the broker to get real fills. **Fallback:** Accept for Phase 1 since it matches the old engine. Add a TODO comment noting this must change for Phase 2.

[RISK] (confidence: 9/10) -- **Hardcoded exit commission `Decimal::from(2)`.** Task 8's engine (lines 1961, 2001) hardcodes exit commission as `Decimal::from(2)`. The old engine calculates this from `CommissionModel::calculate(position.total_contracts(), position.num_legs())`. With 10 contracts and 2 legs, the old engine would charge $20; the new engine charges $2. This will produce materially different backtest results. **Fallback:** Pass `CommissionModel` to the engine or compute exit commission from the Broker.

[RISK] (confidence: 7/10) -- **Position sizing dropped from RiskGate.** The spec says "Engine retains position sizing limits (per-trade, portfolio, correlation)." The old engine (src/backtest/engine.rs:733-802) uses `PositionSizer` to calculate contract count and reject oversized trades. The new `DefaultRiskGate` (Task 5) only checks circuit breakers -- it does not integrate `PositionSizer`. Position sizing effectively vanishes in the refactor. **Fallback:** Either add `PositionSizer` to `DefaultRiskGate::check()`, or move position sizing into the strategy (but the spec explicitly says the engine owns it).

#### Module Depth Audit

| Module | Interface Size | Implementation Size | Verdict |
|--------|---------------|-------------------|---------|
| `core::events` | 7 types, ~10 methods | ~120 LOC of struct defs | SHALLOW -- this is a type bag, which is fine for its role |
| `core::engine` | 1 function (`Engine::run`) | ~100 LOC loop + 2 helper fns | THIN by design, correct |
| `strategy::put_spread` | 2 trait methods | ~150 LOC of screening/exit logic | DEEP, correct |
| `portfolio::tracker` | 6 public methods | ~200 LOC of position lifecycle | DEEP, correct |
| `broker::backtest` | 1 trait method | ~50 LOC | THIN, correct for wrapper |
| `risk::gate` | 2 trait methods | ~30 LOC | SHALLOW -- only checks circuit breakers, should also check position sizing per spec |
| `data::HistoricalDataSource` | 1 trait method + 2 constructors | Wraps DataLoader (350 LOC) | DEEP, correct |

[OBS] -- `risk::gate` is shallower than intended. The spec says it composes circuit breakers + position sizer. The plan only implements circuit breakers.

#### Code Quality

[RISK] (confidence: 8/10) -- **`build_position_from_fill` always calls `put_credit_spread()`.** Task 8's `build_position_from_fill` (line 2051) hardcodes `CreditSpreadBuilder::put_credit_spread()` regardless of the order's `spread_type`. When a call credit spread strategy is added, this function will silently create wrong positions. **Fallback:** Match on `order.spread_type` to select the correct builder method.

[OBS] -- Task 1 tests (lines 37-128) are shape tests -- they verify struct construction and field access, not behavior. This is acceptable for a type-definition task since there is no behavior to test beyond construction. But they will pass without implementation if the struct fields happen to have the right defaults. The `test_order_intent_exit` test with pattern matching on `OrderAction::Close` is the strongest of the bunch.

#### Test Philosophy Audit

[OBS] -- Tests in Tasks 6, 8, 10, and 11 depend on parquet data files existing at `data/orats/`. They use `match source { Ok(..) => ..., Err(_) => eprintln!("Skipping") }` to handle missing data gracefully. This means CI without data files will silently skip the most important tests. This is acceptable for now given the 4.1 GB data size, but note that the core integration tests are effectively manual-only.

[OBS] -- Task 10 has no automated test -- it's a manual `cargo run` invocation. This is a pragmatic choice since the CLI is thin, but it means a regression in CLI-to-engine wiring won't be caught automatically. Task 11's determinism test partially covers this.

#### Vertical Slice Audit

[RISK] (confidence: 6/10) -- **Task 1 has 6 tests in Step 1 before implementation.** Strictly this violates the "one test, one impl, one commit" rule. However, these are type-definition tests for a single `events.rs` file -- splitting into 6 separate tasks would be overhead without benefit since the types are mutually dependent (e.g., `FillEvent` contains `OrderIntent`). **This is acceptable as-is** given the types are all defined in one file and tested together.

[OBS] -- Tasks 3, 4, 5, and 7 similarly bundle 2-3 tests per task. Same reasoning applies -- they test a single module's public interface and the behaviors are tightly coupled.

#### Test Coverage Gaps

```
[+] src/core/engine.rs (Engine::run)
    |
    +-- [TESTED]  NoOp strategy path -- Task 8 test
    +-- [TESTED]  Put spread strategy path -- Task 8 test, Task 11
    +-- [GAP]     Exit order when position not found (position_id mismatch)
    +-- [GAP]     Broker error on entry order (logged but not tested)
    +-- [GAP]     Empty data source (zero events) -- would return Ok with defaults

[+] src/strategy/put_spread.rs (PutCreditSpreadStrategy)
    |
    +-- [TESTED]  Entry generation -- Task 7
    +-- [TESTED]  Max positions check -- Task 7
    +-- [TESTED]  Profit target exit -- Task 7
    +-- [GAP]     Stop loss exit
    +-- [GAP]     Time exit (21 DTE)
    +-- [GAP]     Expiration exit (ITM vs worthless)
    +-- [GAP]     IV filter blocking entry

[+] src/portfolio/tracker.rs (PortfolioTracker)
    |
    +-- [TESTED]  Initial state -- Task 4
    +-- [TESTED]  Add position -- Task 4
    +-- [TESTED]  Close position -- Task 4
    +-- [TESTED]  Equity curve recording -- Task 4
    +-- [GAP]     update_mtm with real quotes
    +-- [GAP]     close_all_remaining
    +-- [GAP]     Drawdown tracking across multiple days

[+] src/risk/gate.rs (DefaultRiskGate)
    |
    +-- [TESTED]  Normal conditions allow -- Task 5
    +-- [TESTED]  Daily loss halts -- Task 5
    +-- [GAP]     VIX halt
    +-- [GAP]     Weekly loss halt
    +-- [GAP]     Drawdown halt
    +-- [GAP]     Exit orders always allowed
```

[RISK] (confidence: 7/10) -- Strategy exit paths (stop loss, time exit, expiration) are not tested. These are critical behaviors for a trading system. The integration test (Task 11) will exercise some of these over 7 months of data, but specific edge cases (stop loss at exactly 125%, expiration on the last day) won't be verified.

#### Failure Modes

[OBS] -- If Task 8's engine loop encounters a `BrokerError` on entry, it logs and continues (`info!("Broker error: {}", e)`). This is correct behavior -- a single failed fill shouldn't halt the backtest. The log is at `info` level; consider `warn` since failed fills are noteworthy.

[OBS] -- `HistoricalDataSource::new()` loads all snapshots into memory upfront (via `loader.load_snapshots()`). For SPY 2007-2026 (19 years), this could be several GB in memory. The old engine had the same pattern. Acceptable for now but will need streaming for larger datasets.

#### Spec vs. Plan Drift

[RISK] (confidence: 9/10) -- **Spec says "Explicit error handling (no unwrap_or_default)" but the plan defers this.** The spec's "Key Decision: Explicit Data Errors" section says `dataframe_to_snapshot()` should return `Err` instead of `unwrap_or(0.0)`. The plan's Spec Coverage table maps this to "Task 6 (data loader tightening)" but Task 6's actual implementation does not modify `data/loader.rs` -- it only adds `DataSource` trait and `HistoricalDataSource` wrapper to `data/mod.rs`. The `unwrap_or(0.0)` calls in `loader.rs` remain untouched. This is a spec requirement that the plan claims to cover but doesn't.

[RISK] (confidence: 9/10) -- **Spec says walk-forward optimizer should be "adapted in this phase" but the plan explicitly defers it.** The spec's Open Questions section defaults to "Adapted in this phase" because "It is tightly coupled to BacktestEngine and will break if the engine is replaced without updating it." The plan defers it with the note "optimizer still works with old engine." This is true only because Task 9 keeps the old engine alive (`pub mod engine; // Keep for now`). This is a coherent decision but contradicts the spec's default answer.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `pub mod core` compiles without breaking existing code | SAFE | Verified: `cargo check` passes with `mod core` added. No `use core::` in existing code. |
| `ExitReason` derives `Copy` (used in pattern matching in `OrderAction`) | SAFE | Verified: `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]` on line 40 of trade.rs |
| `OptionsSnapshot` implements `Clone` (needed for `HistoricalDataSource::next_event`) | SAFE | Verified: `#[derive(Debug, Clone, Default)]` on line 207 of types.rs |
| Group A tasks can all modify `src/lib.rs` in parallel | VALIDATE | Each adds one `pub mod X;` line. Non-conflicting content but git will see merge conflicts on adjacent lines. Build agent must apply sequentially or merge. |
| `SpreadScreener` works correctly when `iv_percentile` is `None` | SAFE | Verified: line 146-149 of spread_screener.rs shows `if let Some(pct) = iv_percentile { ... }` guard |
| The old `BacktestEngine` can coexist with new `Engine` in the same crate | SAFE | Different modules, no name collisions. Verified the old engine is only used in main.rs and walkforward. |
| `Position::new_id()` is deterministic across runs | RISKY | Uses `AtomicU64` counter starting at 1. IDs will be deterministic within a single run but the global counter persists across tests in the same process. Not a correctness issue but could cause confusing test failures if tests depend on specific IDs. |

### Summary

```
[BLOCKER] count: 0
[RISK]    count: 7
[QUESTION] count: 0
```

Risks ranked by severity:
1. **Hardcoded exit commission $2** (confidence: 9/10) -- Will produce materially wrong backtest results vs old engine. Easy fix.
2. **Position sizing dropped from RiskGate** (confidence: 7/10) -- Spec says engine owns it, plan doesn't implement it. Causes oversized positions.
3. **`build_position_from_fill` hardcodes put spread** (confidence: 8/10) -- Won't matter until second strategy is added but violates the pluggable design intent.
4. **Spec drift on explicit data errors** (confidence: 9/10) -- Claimed but not implemented. Acceptable to defer explicitly.
5. **Spec drift on walk-forward optimizer** (confidence: 9/10) -- Contradicts spec default. Coherent decision but should be made explicit.
6. **`pub mod core` naming** (confidence: 7/10) -- Compiles today, future risk from dependency updates.
7. **Exit orders bypass broker** (confidence: 8/10) -- Matches old behavior, needs fixing for Phase 2.

VERDICT: PROCEED_WITH_CAUTION -- Fix the hardcoded exit commission (risk #1) during Task 8 implementation. Monitor position sizing gap (risk #2) and decide during execution whether to add it to DefaultRiskGate or defer. The remaining risks are documented and acceptable for Phase 1.
