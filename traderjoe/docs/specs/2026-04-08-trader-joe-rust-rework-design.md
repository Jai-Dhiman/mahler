# TraderJoe Rust Rework Design

**Goal:** Replace the Python Cloudflare Worker with a lean Rust Worker (`trader-joe` crate) that mechanically executes backtest-validated parameters, eliminating all unvalidated complexity.

**Not in scope:**
- ML regime detection (GaussianMixture)
- Multi-agent debate / FinMem memory system
- Dynamic beta calculation
- Three-perspective risk framework
- Dynamic exit thresholds
- IV term structure spline / OU mean reversion
- Midday and afternoon scan windows
- R2 model storage
- Vectorize / Workers AI bindings
- Changes to `mahler-backtest/` Rust CLI (untouched)

---

## Problem

The Python Worker has ~8,000 lines across 5 handlers. Autoresearch shows that simple, fixed parameters (profit_target=25%, stop_loss=125%, delta 0.05–0.15, DTE 30–45) outperform every ML-enhanced configuration. The regime detector runs stale frozen weights from R2, the three-perspective risk framework adds code paths with no measured edge, and the agents/memory systems are completely dead. The system is unreliable because it is complex without reason.

---

## Solution

A new `trader-joe/` Rust crate compiled as a Cloudflare Worker. Three cron handlers. Three D1 tables. Business logic expressed as pure Rust functions. Python `src/` deleted entirely.

The architecture has two layers:
1. **Business logic** — pure functions with no side effects: IV rank, spread scoring, circuit breaker evaluation, position sizing. All unit-tested.
2. **Handler orchestration** — thin glue that calls external APIs (Alpaca, D1, KV, Discord), feeds data into business logic, and acts on results. Verified via deployment.

---

## Design

### Chosen Approach: Rust Cloudflare Worker (worker-rs)

TraderJoe's Python Workers were already compiled to Pyodide WASM — the runtime model is identical for Rust. The `worker` crate (worker-rs) provides direct bindings to D1, KV, Fetch, and scheduled events. The existing `mahler-backtest/` uses Rust for backtesting with full tokio/polars; the Worker uses a separate, WASM-compatible crate.

The two crates are independent. A workspace `Cargo.toml` at `traderjoe/` ties them together for local `cargo test`.

### Parameters (from autoresearch)

| Parameter | Value | Source |
|-----------|-------|--------|
| profit_target | 25% of credit | autoresearch iter 1 |
| stop_loss | 125% of credit | autoresearch baseline |
| min_dte / max_dte | 30 / 45 | screener.py |
| min_delta / max_delta | 0.05 / 0.15 | backtest validated |
| min_credit_pct | 10% of width | screener.py |
| min_spread_width | $2.00 | screener.py |
| max_risk_per_trade | 2% equity | position_sizer.py |
| max_portfolio_heat | 20% equity | position_sizer.py |
| daily_halt_pct | 2% loss | circuit_breaker.py |
| weekly_halt_pct | 5% loss | circuit_breaker.py |
| vix_halt | 50 | circuit_breaker.py |

### Circuit Breaker State

Circuit breaker uses KV exclusively (not D1). KV keys: `circuit_breaker:status`, `cb_alert:{reason}` (with TTL for dedup).

### Cron Schedule (updated from 3 windows to 2)

| Cron | Handler | Purpose |
|------|---------|---------|
| `0 14 * * MON-FRI` | morning_scan | 10:00 AM ET — scan + place trades |
| `*/5 14-20 * * MON-FRI` | position_monitor | Every 5 min — check exits |
| `15 20 * * MON-FRI` | eod_summary | 4:15 PM ET — daily summary + IV history |

---

## Modules

### analysis::iv_rank

**Interface:** `calculate_iv_metrics(current_iv: f64, history: &[f64]) -> IVMetrics`

**Hides:** The distinction between IV Rank (range-based) and IV Percentile (count-based); neutral defaults when history is insufficient; clamping to 0–100.

**Tested through:** `calculate_iv_metrics` called with controlled history arrays.

**Depth verdict:** DEEP — simple interface, two distinct calculations hidden inside.

---

### analysis::greeks

**Interface:** `black_scholes_delta(option_type: &str, spot: f64, strike: f64, tte: f64, vol: f64, risk_free: f64) -> f64`, `days_to_expiry(expiration: &str) -> i64`

**Hides:** Normal CDF polynomial approximation (Hart 1968), date parsing, YYYY-MM-DD format assumption.

**Tested through:** Known delta values for deep OTM options (verifiable against published Black-Scholes tables).

**Depth verdict:** DEEP — pure math hidden behind a 6-parameter call.

---

### analysis::screener

**Interface:** `OptionsScreener::screen_chain(chain: &OptionsChain, iv_metrics: &IVMetrics) -> Vec<ScoredSpread>`

**Hides:** DTE filtering, liquidity filtering (OI, volume, bid-ask spread), delta range check, credit minimum, spread construction, score calculation (equal-weight: IV percentile + delta proximity + credit ratio + expected value).

**Tested through:** `screen_chain` with synthetic chains that violate one filter at a time.

**Depth verdict:** DEEP — complex multi-filter pipeline behind a 2-parameter interface.

---

### risk::circuit_breaker

**Interface:** `CircuitBreaker::evaluate(params: &EvaluateParams) -> RiskState`

**Hides:** Six graduated thresholds (daily/weekly/drawdown/VIX/rapid-loss/stale-data), size multiplier calculation, reason string selection.

**Tested through:** `evaluate` called with equity values at each threshold boundary.

**Depth verdict:** DEEP — six independent trigger paths behind a single struct method.

---

### risk::position_sizer

**Interface:** `PositionSizer::calculate(max_loss_per_contract: f64, equity: f64, existing_positions: &[ExistingPosition], vix: f64) -> SizeResult`

**Hides:** 2% per-trade limit, 20% portfolio heat limit, 20% per-underlying concentration limit, VIX reduction at 40/50, integer floor rounding.

**Tested through:** `calculate` with controlled equity and position arrays that trip each limit.

**Depth verdict:** DEEP — four independent sizing constraints hidden behind one method.

---

### broker::alpaca

**Interface:** `AlpacaClient::get_account`, `get_options_chain`, `place_spread_order`, `cancel_order`, `get_vix`, `is_market_open`

**Hides:** Alpaca API URL construction (paper vs live), OCC symbol format (`SPYYYMMDDCBBBBB`), authentication headers, response deserialization, Greek fallback calculation when broker omits them.

**Depth verdict:** DEEP — hides Alpaca's REST API and OCC format entirely.

---

### db::d1 and db::kv

**Interface:** `D1Client` — trades CRUD, iv_history upsert, scan_log insert. `KVClient` — circuit breaker get/put, daily stats get/put.

**Hides:** D1 execute/run pattern, KV serialization, retry logic (3 attempts with exponential backoff), JS↔Rust binding conversion.

**Depth verdict:** DEEP — hides the CF binding mechanics entirely.

---

### notifications::discord

**Interface:** `DiscordClient::send_scan_summary`, `send_trade_placed`, `send_position_exit`, `send_error`, `send_eod_summary`

**Hides:** Discord embed format, color codes, field limits, HTTP POST construction, channel ID routing.

**Depth verdict:** DEEP — hides Discord API format behind domain-specific methods.

---

## D1 Schema (additions to mahler-db)

Three new tables added alongside the assistant's `email_triage_log` and `triage_state`. No conflicts.

```sql
-- All placed trades, full lifecycle
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL CHECK(spread_type IN ('bull_put', 'bear_call')),
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    entry_credit REAL NOT NULL,
    max_loss REAL NOT NULL,
    broker_order_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending_fill'
        CHECK(status IN ('pending_fill', 'open', 'closed', 'cancelled')),
    fill_price REAL,
    fill_time TEXT,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    net_pnl REAL,
    iv_rank REAL,
    short_delta REAL,
    short_theta REAL
);

-- Per-symbol daily IV for iv_rank lookback
CREATE TABLE iv_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    iv REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(symbol, date)
);

-- One row per scan run for ops visibility
CREATE TABLE scan_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_time TEXT NOT NULL,
    scan_type TEXT NOT NULL CHECK(scan_type IN ('morning', 'position_monitor', 'eod')),
    underlyings_scanned INTEGER DEFAULT 0,
    opportunities_found INTEGER DEFAULT 0,
    trades_placed INTEGER DEFAULT 0,
    vix REAL,
    circuit_breaker_active INTEGER DEFAULT 0,
    duration_ms INTEGER,
    notes TEXT
);

CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_underlying ON trades(underlying);
CREATE INDEX idx_iv_history_symbol_date ON iv_history(symbol, date);
CREATE INDEX idx_scan_log_time ON scan_log(scan_time);
```

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `traderjoe/Cargo.toml` | New workspace root (`mahler-backtest`, `trader-joe`) | New |
| `traderjoe/trader-joe/Cargo.toml` | New Cloudflare Worker crate | New |
| `traderjoe/trader-joe/src/lib.rs` | Cron router + fetch handler | New |
| `traderjoe/trader-joe/src/types.rs` | Domain types | New |
| `traderjoe/trader-joe/src/config.rs` | SpreadConfig with validated params | New |
| `traderjoe/trader-joe/src/handlers/morning_scan.rs` | Morning scan handler | New |
| `traderjoe/trader-joe/src/handlers/position_monitor.rs` | Position monitor | New |
| `traderjoe/trader-joe/src/handlers/eod_summary.rs` | EOD summary | New |
| `traderjoe/trader-joe/src/broker/alpaca.rs` | Alpaca HTTP client | New |
| `traderjoe/trader-joe/src/broker/types.rs` | Broker-side types | New |
| `traderjoe/trader-joe/src/db/d1.rs` | D1 client | New |
| `traderjoe/trader-joe/src/db/kv.rs` | KV client | New |
| `traderjoe/trader-joe/src/analysis/greeks.rs` | BS delta, days_to_expiry | New |
| `traderjoe/trader-joe/src/analysis/iv_rank.rs` | IVMetrics, calculate_iv_metrics | New |
| `traderjoe/trader-joe/src/analysis/screener.rs` | OptionsScreener | New |
| `traderjoe/trader-joe/src/risk/circuit_breaker.rs` | CircuitBreaker, RiskState | New |
| `traderjoe/trader-joe/src/risk/position_sizer.rs` | PositionSizer | New |
| `traderjoe/trader-joe/src/notifications/discord.rs` | DiscordClient | New |
| `traderjoe/wrangler.toml` | Point at Rust WASM, remove dead bindings, rename to trader-joe | Modify |
| `traderjoe/src/migrations/0017_trader_joe_tables.sql` | New tables | New |
| `traderjoe/src/` (Python tree) | Delete all Python handlers and core modules | Delete |
| `traderjoe/tests/` | Delete Python test suite | Delete |
| `traderjoe/scripts/` | Delete Python training scripts | Delete |
| `traderjoe/pyproject.toml` | Delete | Delete |
| `traderjoe/uv.lock` | Delete | Delete |

---

## Open Questions

- Q: Should `OptionsScreener` score bear_call spreads or only bull_put?
  Default: Both — the Python screener found both, and the backtest validates put spreads primarily, but the screener logic is symmetric. Include both, filter by score.

- Q: What underlyings to scan?
  Default: SPY, QQQ, IWM only (TLT and GLD dropped — they had low OI, wide bid-ask, and paper trading artifacts).
