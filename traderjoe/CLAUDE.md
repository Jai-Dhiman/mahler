See docs/ARCHITECTURE_V2.md for full system design.

## Project

TraderJoe is an autonomous options credit-spread trading system. Two distinct codebases:

- **`traderjoe-backtest/`** — Rust crate: backtesting engine, walk-forward optimizer, CLI
- **`trader-joe/`** — Rust: Cloudflare Workers handlers (WASM), broker integration, risk management

## Rust Crate (`traderjoe-backtest/`)

Trait-based engine. Key types:

- `Engine::run(data_source, strategy, broker, risk_gate, initial_equity, commission)` — main entry point in `engine_core/engine.rs`
- `PutCreditSpreadStrategy` + `PutSpreadConfig` — the only active strategy (`strategy/put_spread.rs`)
- `HistoricalDataSource` / `HistoricalDataSource::from_snapshots()` — data ingestion (`data/mod.rs`)
- `SimulatedBroker` — fills with ORATS slippage model (`broker/`)
- `DefaultRiskGate` — circuit breakers + position sizing (`risk/gate.rs`)
- `WalkForwardOptimizer` — grid search over `PutSpreadConfig` params (`walkforward/optimizer.rs`)
- `BacktestConfig`, `BacktestResult`, `EquityPoint` — result types (`backtest/engine.rs`)

CLI commands: `traderjoe-backtest run` and `traderjoe-backtest optimize`

Data lives in `data/orats/` (gitignored). Tests skip gracefully when data is absent.

## Rust Worker (`trader-joe/`)

Cloudflare Worker (WASM via worker-rs 0.4). Three cron handlers in `src/handlers/`:

- `morning_scan.rs` — 10:00 AM ET (`0 14 * * MON-FRI`), primary scan + order placement
- `position_monitor.rs` — every 5 min during market hours (`*/5 14-20 * * MON-FRI`), profit/stop/gamma exits
- `eod_summary.rs` — 9:00 PM ET (`0 1 * * TUE-SAT`), daily P&L Discord report

All trade decisions are **algorithmic only** — IV rank/percentile, delta filters, spread scoring, position sizing. No LLM involvement in trade logic.

Key modules:
- `analysis/`: `greeks.rs` (Black-Scholes delta), `iv_rank.rs` (IVMetrics), `screener.rs` (OptionsScreener)
- `broker/`: `alpaca.rs` (AlpacaClient HTTP), `types.rs` (OptionsChain, Order, VixData)
- `db/`: `d1.rs` (D1Client — trades table CRUD), `kv.rs` (KvClient — circuit breaker + daily stats)
- `risk/`: `circuit_breaker.rs` (CircuitBreaker with 6 RiskLevels), `position_sizer.rs` (PositionSizer)
- `notifications/`: `discord.rs` (DiscordClient embed builders)
- `config.rs`: `SpreadConfig` — autoresearch-validated params (profit_target=0.25, stop_loss=1.25)
- `types.rs`: `CreditSpread`, `Trade`, `TradeStatus`, `SpreadType`, `ExitReason`

Storage: Cloudflare D1 (`trades`, `iv_history`, `scan_log` tables — migration `0017`) + KV (circuit breaker state, daily stats).

Build: `wrangler deploy` (WASM target). Local dev: `wrangler dev`.

Note: VIX is proxied via VIXY ETF (Alpaca has no spot VIX endpoint). Circuit breaker and position sizer thresholds were calibrated against spot VIX — VIXY typically trades at a discount so thresholds err cautious.

## Key Constraints

- `BacktestConfig::default()` is still used in `engine_core/engine.rs:build_result()` — do not remove it
- Walk-forward optimizer uses `PutSpreadConfig` (not `BacktestConfig`) for parameter sweeps
- 154 Rust tests must pass: `cargo test` (101 backtest + 53 trader-joe)
- `trader-joe/` builds to WASM: `cargo build --target wasm32-unknown-unknown`
