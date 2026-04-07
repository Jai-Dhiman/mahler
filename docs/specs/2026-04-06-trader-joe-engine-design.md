# trader-joe: Trait-Based Options Trading Engine

**Goal:** Replace the monolithic `BacktestEngine` with a trait-based architecture where strategy code runs identically in backtest and live modes, enabling validated strategies to go from ORATS historical data to Alpaca paper trading with zero code changes.

**Not in scope:**
- Alpaca broker/data source implementation (stubs only -- built in Phase 2)
- New strategies beyond put credit spreads (pluggable trait enables them later)
- Project renaming from mahler to trader-joe (separate task)
- Removal of Python/CF Workers code (separate task)
- UI, dashboard, or monitoring
- Intraday data support (events are timestamped for future compatibility, but only daily resolution is implemented)

---

## System Vision (3-Phase Roadmap)

**Phase 1 (this spec):** Refactor the existing `mahler-backtest` into a trait-based engine. Extract strategy logic from the engine. Validate against ORATS historical data. Produce a working backtester where strategies are pluggable and the engine is unaware of its data source or broker.

**Phase 2 (future spec):** Implement `AlpacaDataSource` and `AlpacaBroker` behind the existing traits. The same `PutCreditSpreadStrategy` runs live on Alpaca paper trading with no changes.

**Phase 3 (future spec):** Evaluate real broker (IBKR or Alpaca), implement production risk controls, add new strategies validated through the backtester.

---

## Problem

The current `BacktestEngine` (src/backtest/engine.rs) is a 940-line monolith that:

1. **Hardcodes the put credit spread strategy** into `screen_and_enter()` and `check_exits()`. Adding any new strategy requires modifying the engine.
2. **Mixes signal logic with safety logic.** IV percentile filtering, regime classification, and circuit breakers all live in the same methods. Signal filters (strategy-owned) cannot be separated from safety checks (engine-owned).
3. **Cannot run live.** The engine directly calls `DataLoader` and drives the simulation loop. There is no abstraction for a live data source or real broker.
4. **Silently swallows bad data.** The data loader uses `unwrap_or(0.0)` and `unwrap_or_default()` on financial fields (bid, ask, delta, IV). A missing bid becomes zero, which can trigger false trade entries.

Despite these problems, the existing code has substantial working infrastructure: ORATS parquet loading, position lifecycle management, slippage/commission models, circuit breakers, position sizing, walk-forward optimization, and performance metrics. The refactor preserves and reorganizes this code rather than rewriting it.

---

## Solution (from the user's perspective)

After this refactor, the user can:

1. Run backtests with `cargo run -- run --ticker SPY --start 2020-01-01 --end 2024-12-31` and get the same results as before (regression validated).
2. Swap strategies by changing a config flag: `--strategy put-spread` vs future `--strategy iron-condor`.
3. Trust that a strategy validated in backtest will run identically in live mode (when the live data source and broker are implemented in Phase 2).
4. See explicit errors when data is malformed instead of silent zero-fills.

---

## Design

### Architecture: Four Traits

The engine is decomposed into four traits that form the boundary between components:

```rust
trait DataSource {
    fn next_event(&mut self) -> Option<MarketEvent>;
}

trait Strategy {
    fn on_snapshot(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent>;
    fn on_fill(&mut self, fill: &FillEvent);
}

trait Broker {
    fn submit_order(&mut self, order: &OrderIntent, snapshot: &OptionsSnapshot) -> Result<FillEvent, BrokerError>;
}

trait RiskGate {
    fn check(&self, order: &OrderIntent, portfolio: &PortfolioView) -> RiskDecision;
    fn update(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView);
}
```

### Key Decision: Engine Drives the Loop

The engine owns the event loop and calls traits in a fixed order:

```
while let Some(event) = data_source.next_event() {
    portfolio.update_mtm(&event.snapshot);
    risk_gate.update(&event.snapshot, &portfolio.view());
    let orders = strategy.on_snapshot(&event.snapshot, &portfolio.view());
    for order in orders {
        match risk_gate.check(&order, &portfolio.view()) {
            RiskDecision::Allowed => {
                match broker.submit_order(&order, &event.snapshot) {
                    Ok(fill) => {
                        portfolio.apply_fill(&fill);
                        strategy.on_fill(&fill);
                    }
                    Err(e) => log_broker_error(e),
                }
            }
            RiskDecision::Rejected(reason) => log_rejection(reason),
        }
    }
    portfolio.record_equity(event.timestamp);
}
```

This design was chosen over a message-bus architecture (like NautilusTrader) because:
- The codebase is single-threaded and single-process
- Message buses add complexity that pays off at scale but hurts debuggability at this size
- The trait-based approach achieves the same backtest/live parity without the indirection

### Key Decision: Strategy Owns Signal Logic

Everything that was in `BacktestEngine::screen_and_enter()` moves to `PutCreditSpreadStrategy`:
- IV percentile calculation and filtering
- Regime classification
- Spread screening (DTE, delta, liquidity filters)
- Entry decision logic
- Exit condition checking (profit target, stop loss, time exit)

The engine retains only safety infrastructure:
- Circuit breakers (daily/weekly loss, drawdown, VIX halt)
- Position sizing limits (per-trade, portfolio, correlation)

### Key Decision: Explicit Data Errors

The current `dataframe_to_snapshot()` uses `unwrap_or(0.0)` on all numeric fields. In the refactor:
- Missing or unparseable bid/ask/delta/IV returns `Err`, and the quote is skipped with a warning log
- A snapshot with zero valid quotes returns `DataError`
- The engine propagates data errors as `EngineError::Data` and halts the backtest

### What Changes vs. What Stays

**Stays (moved but unchanged):**
- `data/types.rs` -- OptionQuote, OptionsChain, OptionsSnapshot, Greeks
- `backtest/slippage.rs` -- SlippageModel, Slippage
- `backtest/commission.rs` -- CommissionModel, Commission
- `backtest/trade.rs` -- Position, PositionLeg, Trade, CreditSpreadBuilder, ExitReason
- `risk/circuit_breakers.rs` -- CircuitBreaker, CircuitBreakerConfig
- `risk/position_sizer.rs` -- PositionSizer, PositionSizerConfig, SizingResult
- `risk/portfolio_greeks.rs` -- PortfolioGreeks
- `metrics/calculator.rs` -- MetricsCalculator, PerformanceMetrics
- `walkforward/optimizer.rs` -- WalkForwardOptimizer (adapted to use Strategy trait)
- `walkforward/periods.rs` -- WalkForwardPeriods
- `validation/` -- DataIntegrityValidator, BlackScholes (unchanged)
- `analytics/iv_term_structure.rs` -- IVTermStructureAnalyzer (moved to strategy-owned)
- `analytics/spread_screener.rs` -- SpreadScreener (moved to strategy-owned)
- `regime/classifier.rs` -- RegimeClassifier (moved to strategy-owned)

**Changes:**
- `backtest/engine.rs` -- gutted and replaced with trait-based `Engine` in `core/engine.rs`
- `data/loader.rs` -- wrapped behind `DataSource` trait; error handling tightened
- New: `core/events.rs` -- MarketEvent, OrderIntent, FillEvent, RiskDecision types
- New: `core/engine.rs` -- generic Engine<D: DataSource, S: Strategy, B: Broker, R: RiskGate>
- New: `strategy/mod.rs` -- Strategy trait definition
- New: `strategy/put_spread.rs` -- PutCreditSpreadStrategy extracted from engine
- New: `broker/mod.rs` -- Broker trait definition
- New: `broker/backtest.rs` -- SimulatedBroker wrapping slippage + commission
- New: `portfolio/tracker.rs` -- PortfolioTracker extracted from engine equity tracking

---

## Modules

### core::engine (THIN)
- **Interface:** `Engine::run(data_source, strategy, broker, risk_gate) -> Result<BacktestResult, EngineError>`
- **Hides:** Event loop ordering, portfolio-strategy-broker wiring
- **Tested through:** Integration tests that run a full backtest with known data and verify results

### data::orats (DEEP)
- **Interface:** `impl DataSource for HistoricalDataSource` with `next_event() -> Option<MarketEvent>`
- **Hides:** Polars parquet loading, DataFrame-to-OptionsSnapshot conversion, year-spanning date ranges, lazy loading, quote validation and error filtering
- **Tested through:** Load a known date from SPY 2020 parquet, verify snapshot has expected chains/quotes/Greeks

### strategy::put_spread (DEEP)
- **Interface:** `impl Strategy for PutCreditSpreadStrategy` with `on_snapshot()` and `on_fill()`
- **Hides:** IV percentile calculation, regime classification, spread screening, entry/exit state machine, DTE/delta/liquidity filtering
- **Tested through:** Given hand-crafted snapshots and portfolio states, verify correct OrderIntents are emitted

### portfolio::tracker (DEEP)
- **Interface:** `PortfolioTracker::update_mtm()`, `apply_fill()`, `view()`, `record_equity()`, `result()`
- **Hides:** Position lifecycle management, MTM calculation, equity curve tracking, drawdown calculation, aggregate Greeks
- **Tested through:** Sequence of fills and market updates, verify equity/P&L/positions are correct

### broker::backtest (THIN)
- **Interface:** `impl Broker for SimulatedBroker` with `submit_order()`
- **Hides:** Slippage model application, commission calculation, fill price computation from snapshot bid/ask
- **Tested through:** Submit an OrderIntent with known bid/ask, verify fill price matches slippage model

### risk::gate (THIN)
- **Interface:** `impl RiskGate for DefaultRiskGate` with `check()` and `update()`
- **Hides:** Composition of CircuitBreaker + PositionSizer checks
- **Tested through:** Given portfolio states at various loss levels, verify orders are allowed/rejected correctly

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `src/core/mod.rs` | Module declaration for core | New |
| `src/core/engine.rs` | Trait-based engine loop | New |
| `src/core/events.rs` | MarketEvent, OrderIntent, FillEvent, RiskDecision | New |
| `src/strategy/mod.rs` | Strategy trait + re-exports | New |
| `src/strategy/put_spread.rs` | PutCreditSpreadStrategy extracted from engine | New |
| `src/broker/mod.rs` | Broker trait + re-exports | New |
| `src/broker/backtest.rs` | SimulatedBroker wrapping slippage/commission | New |
| `src/portfolio/mod.rs` | Module declaration | New |
| `src/portfolio/tracker.rs` | PortfolioTracker extracted from engine | New |
| `src/risk/gate.rs` | DefaultRiskGate composing breakers + sizer | New |
| `src/risk/mod.rs` | Add gate re-export | Modify |
| `src/data/mod.rs` | Add DataSource trait, HistoricalDataSource | Modify |
| `src/data/loader.rs` | Tighten error handling (remove unwrap_or_default) | Modify |
| `src/lib.rs` | Update module declarations and re-exports | Modify |
| `src/main.rs` | Adapt CLI to use new Engine | Modify |
| `src/backtest/engine.rs` | Remove (replaced by core/engine.rs) | Delete |
| `src/backtest/mod.rs` | Remove engine re-export, keep trade/slippage/commission | Modify |
| `src/walkforward/optimizer.rs` | Adapt to use Strategy trait instead of BacktestEngine | Modify |

---

## Open Questions

- **Q:** Should `PortfolioView` (the read-only view passed to Strategy) include individual position details, or just aggregates (equity, Greeks, open count)?
  **Default:** Include individual positions. The strategy needs to know what it holds to make exit decisions (e.g., "this specific position hit its profit target").

- **Q:** Should the walk-forward optimizer be adapted in this phase, or deferred?
  **Default:** Adapted in this phase. It is tightly coupled to `BacktestEngine` and will break if the engine is replaced without updating it.
