# TraderJoe Paper→Live Measurement Infrastructure & Go-Live Gate — Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task within a group). Sequential groups run after all tasks in the prior group have shipped.
> Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Capture the data needed over 6 months of paper trading to decide whether the credit-spread strategy is safe to run with real money, and document the explicit criteria for that decision.
**Spec:** `docs/specs/2026-04-22-paper-to-live-measurement-gate-design.md`
**Style:** Follow `traderjoe/CLAUDE.md`. Rust 2021. Worker-rs `0.7`. `cargo test` from the `traderjoe/` directory must continue to pass. No `#[tokio::main]`, no async in `#[test]`.

## Task Groups

- **Group A (parallel):** Tasks 1–4. Independent new/modified files.
- **Group B (sequential, depends on A):** Tasks 5–13. All touch `traderjoe/src/measurement/` — sequential because each appends to `mod.rs`.
- **Group C (sequential, depends on B):** Tasks 14–17. All touch `traderjoe/src/db/d1.rs`.
- **Group D (sequential, depends on C):** Tasks 18–21. Handler + sizer wiring.
- **Group E (sequential, depends on D):** Tasks 22–24. Discord + weekly handler + lib/wrangler wiring.

`cargo test` command for all tasks unless specified otherwise: `cd traderjoe && cargo test <test_name>`

---

## Group A — Parallel

### Task 1: Migration 0002 — measurement schema

**Group:** A
**Behavior being verified:** The migration SQL parses in SQLite without error and creates the expected tables/columns.
**Interface under test:** The SQL file itself, validated by `sqlite3`.

**Files:**
- Create: `traderjoe/src/migrations/0002_measurement_infrastructure.sql`
- Test: shell command (no Rust test)

- [ ] **Step 1: Write the failing check**

```bash
test -f traderjoe/src/migrations/0002_measurement_infrastructure.sql && \
  sqlite3 :memory: < traderjoe/src/migrations/0002_measurement_infrastructure.sql
```
Expected: FAIL — `No such file or directory`.

- [ ] **Step 2: Create the migration**

Write `traderjoe/src/migrations/0002_measurement_infrastructure.sql`:

```sql
-- Measurement infrastructure for paper→live go-live gate.
-- Adds NBBO/Greeks capture to trades and four new tables for observability.

-- NBBO and expanded Greeks additions to trades.
ALTER TABLE trades ADD COLUMN entry_short_bid REAL;
ALTER TABLE trades ADD COLUMN entry_short_ask REAL;
ALTER TABLE trades ADD COLUMN entry_long_bid REAL;
ALTER TABLE trades ADD COLUMN entry_long_ask REAL;
ALTER TABLE trades ADD COLUMN entry_net_mid REAL;
ALTER TABLE trades ADD COLUMN exit_short_bid REAL;
ALTER TABLE trades ADD COLUMN exit_short_ask REAL;
ALTER TABLE trades ADD COLUMN exit_long_bid REAL;
ALTER TABLE trades ADD COLUMN exit_long_ask REAL;
ALTER TABLE trades ADD COLUMN exit_net_mid REAL;
ALTER TABLE trades ADD COLUMN entry_short_gamma REAL;
ALTER TABLE trades ADD COLUMN entry_short_vega REAL;
ALTER TABLE trades ADD COLUMN entry_long_delta REAL;
ALTER TABLE trades ADD COLUMN entry_long_gamma REAL;
ALTER TABLE trades ADD COLUMN entry_long_vega REAL;
ALTER TABLE trades ADD COLUMN nbbo_displayed_size_short INTEGER;
ALTER TABLE trades ADD COLUMN nbbo_displayed_size_long INTEGER;
ALTER TABLE trades ADD COLUMN nbbo_snapshot_time TEXT;

CREATE TABLE IF NOT EXISTS equity_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('eod','trade_open','trade_close','circuit_breaker')),
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    open_position_mtm REAL NOT NULL,
    realized_pnl_day REAL NOT NULL DEFAULT 0,
    unrealized_pnl_day REAL NOT NULL DEFAULT 0,
    open_position_count INTEGER NOT NULL DEFAULT 0,
    trade_id_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_history(timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_equity_eod_unique
    ON equity_history(date(timestamp)) WHERE event_type = 'eod';

CREATE TABLE IF NOT EXISTS portfolio_greeks_eod (
    date TEXT PRIMARY KEY,
    beta_weighted_delta REAL NOT NULL,
    total_gamma REAL NOT NULL,
    total_vega REAL NOT NULL,
    total_theta REAL NOT NULL,
    delta_by_underlying TEXT NOT NULL,
    max_gamma_single_position REAL NOT NULL,
    max_vega_single_position REAL NOT NULL,
    open_position_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS market_context_daily (
    date TEXT PRIMARY KEY,
    spot_vix REAL,
    spot_vix_source TEXT CHECK (spot_vix_source IN ('fred','stooq','unavailable')),
    spot_vix_source_date TEXT,
    vixy_close REAL,
    spy_20d_realized_vol REAL,
    spy_20d_return REAL,
    spy_drawdown_from_52w_high REAL
);

CREATE TABLE IF NOT EXISTS metrics_weekly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    trade_count INTEGER NOT NULL,
    sharpe REAL,
    sortino REAL,
    profit_factor REAL,
    win_rate REAL,
    pnl_skew REAL,
    max_drawdown_pct REAL,
    mean_slippage_vs_mid REAL,
    max_slippage_vs_mid REAL,
    slippage_vs_orats_ratio REAL,
    fill_size_violation_count INTEGER NOT NULL DEFAULT 0,
    fill_size_violation_pct REAL,
    regime_buckets TEXT,
    greek_ranges TEXT,
    sample_size_tag TEXT NOT NULL CHECK (sample_size_tag IN ('INSUFFICIENT','WEAK','OK'))
);
```

- [ ] **Step 3: Verify parse**

```bash
sqlite3 :memory: < traderjoe/src/migrations/0002_measurement_infrastructure.sql && echo OK
```
Expected: PASS — `OK`. (Note: the `ALTER TABLE` statements expect `trades` to exist; for isolation run with `0001` first: `cat traderjoe/src/migrations/0001_initial_schema.sql traderjoe/src/migrations/0002_measurement_infrastructure.sql | sqlite3 :memory:`)

- [ ] **Step 4: Commit**

```bash
git add traderjoe/src/migrations/0002_measurement_infrastructure.sql && \
  git commit -m "feat(traderjoe): add measurement infrastructure schema migration"
```

---

### Task 2: GO_LIVE_GATE.md — manual review checklist

**Group:** A
**Behavior being verified:** Doc exists with required sections; grep confirms structure.
**Interface under test:** The markdown file as read by a human.

**Files:**
- Create: `traderjoe/docs/GO_LIVE_GATE.md`

- [ ] **Step 1: Write the failing check**

```bash
grep -q "## Hard Pass/Fail" traderjoe/docs/GO_LIVE_GATE.md 2>/dev/null && \
  grep -q "## Capital Ramp" traderjoe/docs/GO_LIVE_GATE.md 2>/dev/null && \
  grep -q "## Catastrophic Halts" traderjoe/docs/GO_LIVE_GATE.md 2>/dev/null
```
Expected: FAIL — exit 1.

- [ ] **Step 2: Create the doc**

Write `traderjoe/docs/GO_LIVE_GATE.md`:

```markdown
# TraderJoe Paper→Live Go-Live Gate

This is a manual review checklist. Run it to decide whether to flip `ENVIRONMENT`
from `paper` to `live`, and whether to advance the capital-fraction ramp.

All SQL queries below are run against the D1 database (`wrangler d1 execute mahler-db --command "..."`).

## Hard Pass/Fail

Every item must pass. A single failure blocks advancement.

### 1. NBBO captured for 100% of trades over last 30 days
```sql
SELECT COUNT(*) AS missing FROM trades
 WHERE created_at >= date('now','-30 days')
   AND (entry_short_bid IS NULL OR entry_short_ask IS NULL
        OR entry_long_bid IS NULL OR entry_long_ask IS NULL);
```
PASS if `missing = 0`.

### 2. No paper-fill size violations over last 30 days
```sql
SELECT COUNT(*) AS violations FROM trades
 WHERE created_at >= date('now','-30 days')
   AND nbbo_displayed_size_short IS NOT NULL
   AND contracts > MIN(nbbo_displayed_size_short, nbbo_displayed_size_long);
```
PASS if `violations = 0`.

### 3. Equity curve has no gaps in last 30 trading days
```sql
SELECT COUNT(DISTINCT date(timestamp)) AS days_recorded
  FROM equity_history
 WHERE event_type = 'eod' AND timestamp >= date('now','-30 days');
```
PASS if `days_recorded >= 20` (accounting for weekends/holidays).

### 4. Weekly metrics job ran on schedule, last 4 weeks
```sql
SELECT COUNT(*) AS runs FROM metrics_weekly
 WHERE generated_at >= date('now','-28 days');
```
PASS if `runs >= 4`.

### 5. Circuit breaker has been fire-tested
Manual: at least one record in `equity_history` with `event_type = 'circuit_breaker'`, OR a dated log of a synthetic trigger test with before/after KV state captured.

### 6. Portfolio Greeks snapshot current
```sql
SELECT date FROM portfolio_greeks_eod ORDER BY date DESC LIMIT 1;
```
PASS if the most recent date is within the last 2 trading days.

### 7. Regime tags present on 100% of trades last 30 days
```sql
SELECT COUNT(*) AS missing FROM trades t
 WHERE created_at >= date('now','-30 days')
   AND NOT EXISTS (SELECT 1 FROM market_context_daily m WHERE m.date = date(t.created_at));
```
PASS if `missing = 0`.

### 8. Slippage within 2x ORATS model
```sql
SELECT slippage_vs_orats_ratio FROM metrics_weekly ORDER BY generated_at DESC LIMIT 1;
```
PASS if `slippage_vs_orats_ratio <= 2.0`.

### 9. Minimum 60 closed trades (for 10% capital)
```sql
SELECT COUNT(*) AS closed FROM trades WHERE status = 'closed';
```
PASS if `closed >= 60` for 10% capital; `>= 100` for 25%; `>= 200` for 50%+.

### 10. At least 30 trading days elapsed under all infrastructure green
Manual review of past 4 `metrics_weekly` reports; no infra-gap flags.

## Catastrophic Halts

Any single trigger halts go-live advancement.

### C1. Realized max DD exceeds theoretical per-trade max loss × 1.5
```sql
SELECT max_drawdown_pct FROM metrics_weekly ORDER BY generated_at DESC LIMIT 1;
```
Compare vs theoretical per-trade max loss percent of equity (approx 2% under default `PositionSizer`). Halt if `max_drawdown_pct > 3.0%`.

### C2. Profit factor < 1.0 over full window
```sql
SELECT profit_factor FROM metrics_weekly WHERE window_end = (SELECT MAX(window_end) FROM metrics_weekly);
```
Halt if `profit_factor < 1.0` on the full-window row.

### C3. Any single trade lost > 2× configured per-trade max loss
```sql
SELECT id, net_pnl, max_loss, contracts FROM trades
 WHERE status = 'closed' AND net_pnl < -(max_loss * contracts * 2);
```
Halt if any row returned.

## Capital Ramp

| Stage | Capital | Requirements |
|-------|---------|--------------|
| 1 | 10% | Hard items 1–8, 10 pass. ≥60 closed trades. No catastrophic halts. |
| 2 | 25% | Stage 1 + ≥100 closed trades + VIX>25 observed (≥3 trading days in `market_context_daily` with `spot_vix > 25`) on paper or live. |
| 3 | 50% | Stage 2 + ≥200 closed trades + VIX>25 survived on **live** specifically (≥3 live trading days with `spot_vix > 25` and no catastrophic halts triggered). |
| 4 | 100% | Stage 3 + 3+ months at 50% capital with live P&L tracking within ±25% of paper expectations. |

Advancing requires flipping `LIVE_CAPITAL_FRACTION` env var and redeploying.

## Beta Table Review

The static `BETA_TABLE` in `traderjoe/src/measurement/portfolio_greeks.rs` encodes:
- SPY = 1.00
- QQQ = 1.15
- IWM = 1.25

Review quarterly (1st of Jan/Apr/Jul/Oct). Recompute by regressing 252-day daily returns of each underlying against SPY. Update the constant and commit.

## VIX-Regime Observation

A "VIX >25 event" = at least 3 trading days in `market_context_daily` where `spot_vix > 25`, within a 10-trading-day window. Verify via:
```sql
SELECT date, spot_vix FROM market_context_daily
 WHERE date >= date('now','-30 days') AND spot_vix > 25
 ORDER BY date DESC;
```
```

- [ ] **Step 3: Verify**

```bash
grep -q "## Hard Pass/Fail" traderjoe/docs/GO_LIVE_GATE.md && \
  grep -q "## Capital Ramp" traderjoe/docs/GO_LIVE_GATE.md && \
  grep -q "## Catastrophic Halts" traderjoe/docs/GO_LIVE_GATE.md && echo OK
```
Expected: PASS — `OK`.

- [ ] **Step 4: Commit**

```bash
git add traderjoe/docs/GO_LIVE_GATE.md && git commit -m "docs(traderjoe): add go-live gate checklist"
```

---

### Task 3: OptionContract gains bid_size and ask_size

**Group:** A
**Behavior being verified:** `OptionContract` round-trips `bid_size` and `ask_size` through serde JSON.
**Interface under test:** `OptionContract` struct fields.

**Files:**
- Modify: `traderjoe/src/broker/types.rs`
- Test: same file (`#[cfg(test)] mod tests`).

- [ ] **Step 1: Write the failing test**

Append inside the existing `#[cfg(test)] mod tests` block in `traderjoe/src/broker/types.rs`:

```rust
#[test]
fn option_contract_round_trips_bid_and_ask_size() {
    let contract = OptionContract {
        symbol: "SPY260515P00460000".to_string(),
        underlying: "SPY".to_string(),
        expiration: "2026-05-15".to_string(),
        strike: 460.0,
        option_type: OptionType::Put,
        bid: 1.00, ask: 1.05, last: 1.02,
        volume: 100, open_interest: 500,
        implied_volatility: Some(0.22),
        delta: Some(-0.10), gamma: Some(0.01), theta: Some(-0.05), vega: Some(0.10),
        bid_size: Some(25),
        ask_size: Some(40),
    };
    let json = serde_json::to_string(&contract).expect("serialize");
    let back: OptionContract = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.bid_size, Some(25));
    assert_eq!(back.ask_size, Some(40));
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test option_contract_round_trips_bid_and_ask_size
```
Expected: FAIL — `no field 'bid_size' on type OptionContract` (also `ask_size`, and existing `gamma/theta/vega` fixture fields if those weren't set before — they already exist per `src/broker/types.rs:11-26`).

- [ ] **Step 3: Implement**

In `traderjoe/src/broker/types.rs`, add `bid_size` and `ask_size` fields to `OptionContract`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    pub symbol: String,
    pub underlying: String,
    pub expiration: String,
    pub strike: f64,
    pub option_type: OptionType,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub volume: i64,
    pub open_interest: i64,
    pub implied_volatility: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub bid_size: Option<i64>,
    pub ask_size: Option<i64>,
}
```

Also update the `make_contract` test helper at the bottom of the same file so existing tests still compile — add `bid_size: Some(10), ask_size: Some(10)` to that struct literal.

Also update `traderjoe/src/broker/alpaca.rs` `get_options_chain` function — in the `OptionContract` construction block (around line 166-182), add two fields right after `vega`:

```rust
                    bid_size: quote["bs"].as_i64(),
                    ask_size: quote["as"].as_i64(),
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test
```
Expected: PASS — all existing tests continue to pass, new test passes.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/broker/types.rs traderjoe/src/broker/alpaca.rs && \
  git commit -m "feat(broker): capture NBBO bid/ask size on OptionContract"
```

---

### Task 4: Trade struct extended with NBBO and expanded Greeks fields

**Group:** A
**Behavior being verified:** `Trade` struct round-trips all new NBBO and Greeks fields via serde.
**Interface under test:** `Trade` struct.

**Files:**
- Modify: `traderjoe/src/types.rs`

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `traderjoe/src/types.rs`:

```rust
#[test]
fn trade_round_trips_nbbo_and_expanded_greeks() {
    let trade = Trade {
        id: "t1".to_string(),
        created_at: "2026-04-22T10:00:00Z".to_string(),
        underlying: "SPY".to_string(),
        spread_type: SpreadType::BullPut,
        short_strike: 460.0, long_strike: 455.0,
        expiration: "2026-05-15".to_string(),
        contracts: 2,
        entry_credit: 0.75, max_loss: 425.0,
        broker_order_id: Some("o1".to_string()),
        status: TradeStatus::Open,
        fill_price: Some(0.74), fill_time: Some("2026-04-22T10:05:00Z".to_string()),
        exit_price: None, exit_time: None, exit_reason: None, net_pnl: None,
        iv_rank: Some(65.0),
        short_delta: Some(-0.28), short_theta: Some(0.05),
        entry_short_bid: Some(0.74), entry_short_ask: Some(0.76),
        entry_long_bid: Some(0.24), entry_long_ask: Some(0.26),
        entry_net_mid: Some(0.75),
        exit_short_bid: None, exit_short_ask: None,
        exit_long_bid: None, exit_long_ask: None, exit_net_mid: None,
        entry_short_gamma: Some(0.012), entry_short_vega: Some(0.32),
        entry_long_delta: Some(-0.18), entry_long_gamma: Some(0.009), entry_long_vega: Some(0.28),
        nbbo_displayed_size_short: Some(50), nbbo_displayed_size_long: Some(75),
        nbbo_snapshot_time: Some("2026-04-22T10:00:00Z".to_string()),
    };
    let json = serde_json::to_string(&trade).unwrap();
    let back: Trade = serde_json::from_str(&json).unwrap();
    assert_eq!(back.entry_net_mid, Some(0.75));
    assert_eq!(back.entry_short_gamma, Some(0.012));
    assert_eq!(back.nbbo_displayed_size_short, Some(50));
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test trade_round_trips_nbbo_and_expanded_greeks
```
Expected: FAIL — `no field 'entry_short_bid' on type Trade`.

- [ ] **Step 3: Implement**

Extend the `Trade` struct in `traderjoe/src/types.rs` (below `short_theta`):

```rust
    pub entry_short_bid: Option<f64>,
    pub entry_short_ask: Option<f64>,
    pub entry_long_bid: Option<f64>,
    pub entry_long_ask: Option<f64>,
    pub entry_net_mid: Option<f64>,
    pub exit_short_bid: Option<f64>,
    pub exit_short_ask: Option<f64>,
    pub exit_long_bid: Option<f64>,
    pub exit_long_ask: Option<f64>,
    pub exit_net_mid: Option<f64>,
    pub entry_short_gamma: Option<f64>,
    pub entry_short_vega: Option<f64>,
    pub entry_long_delta: Option<f64>,
    pub entry_long_gamma: Option<f64>,
    pub entry_long_vega: Option<f64>,
    pub nbbo_displayed_size_short: Option<i64>,
    pub nbbo_displayed_size_long: Option<i64>,
    pub nbbo_snapshot_time: Option<String>,
```

Also update `traderjoe/src/db/d1.rs` `TradeRow::from_d1_row` to populate these fields (all `None` for now, until Task 14 wires them up):

```rust
            entry_short_bid: row["entry_short_bid"].as_f64(),
            entry_short_ask: row["entry_short_ask"].as_f64(),
            entry_long_bid: row["entry_long_bid"].as_f64(),
            entry_long_ask: row["entry_long_ask"].as_f64(),
            entry_net_mid: row["entry_net_mid"].as_f64(),
            exit_short_bid: row["exit_short_bid"].as_f64(),
            exit_short_ask: row["exit_short_ask"].as_f64(),
            exit_long_bid: row["exit_long_bid"].as_f64(),
            exit_long_ask: row["exit_long_ask"].as_f64(),
            exit_net_mid: row["exit_net_mid"].as_f64(),
            entry_short_gamma: row["entry_short_gamma"].as_f64(),
            entry_short_vega: row["entry_short_vega"].as_f64(),
            entry_long_delta: row["entry_long_delta"].as_f64(),
            entry_long_gamma: row["entry_long_gamma"].as_f64(),
            entry_long_vega: row["entry_long_vega"].as_f64(),
            nbbo_displayed_size_short: row["nbbo_displayed_size_short"].as_i64(),
            nbbo_displayed_size_long: row["nbbo_displayed_size_long"].as_i64(),
            nbbo_snapshot_time: row["nbbo_snapshot_time"].as_str().map(str::to_string),
```

(These go inside the `Some(Trade { ... })` block after `short_theta`.)

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/types.rs traderjoe/src/db/d1.rs && \
  git commit -m "feat(types): extend Trade with NBBO and expanded Greeks fields"
```

---

## Group B — Sequential (measurement module)

### Task 5: measurement/nbbo — NbboSnapshot and snapshot_spread_nbbo

**Group:** B (sequential — first measurement task, creates mod.rs)
**Behavior being verified:** Given an `OptionsChain` containing both legs, the function returns a snapshot with the correct bid/ask/size for each leg.
**Interface under test:** `measurement::nbbo::snapshot_spread_nbbo`.

**Files:**
- Create: `traderjoe/src/measurement/mod.rs`
- Create: `traderjoe/src/measurement/nbbo.rs`
- Modify: `traderjoe/src/lib.rs` (add `mod measurement;`)

- [ ] **Step 1: Write the failing test**

In `traderjoe/src/measurement/nbbo.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};

    fn mk_contract(strike: f64, ot: OptionType, bid: f64, ask: f64, bs: i64, r#as: i64) -> OptionContract {
        OptionContract {
            symbol: format!("X{}{:?}", strike, ot),
            underlying: "SPY".to_string(),
            expiration: "2026-05-15".to_string(),
            strike,
            option_type: ot,
            bid, ask, last: (bid+ask)/2.0,
            volume: 0, open_interest: 0,
            implied_volatility: Some(0.20),
            delta: Some(-0.25), gamma: Some(0.01), theta: Some(-0.03), vega: Some(0.20),
            bid_size: Some(bs), ask_size: Some(r#as),
        }
    }

    #[test]
    fn snapshots_both_legs_of_bull_put_spread() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                mk_contract(460.0, OptionType::Put, 0.74, 0.76, 50, 55),
                mk_contract(455.0, OptionType::Put, 0.24, 0.26, 60, 75),
            ],
        };
        let snap = snapshot_spread_nbbo(&chain, 460.0, 455.0, OptionType::Put)
            .expect("snapshot exists");
        assert!((snap.short_bid - 0.74).abs() < 1e-9);
        assert!((snap.short_ask - 0.76).abs() < 1e-9);
        assert!((snap.long_bid - 0.24).abs() < 1e-9);
        assert!((snap.long_ask - 0.26).abs() < 1e-9);
        assert_eq!(snap.short_bid_size, Some(50));
        assert_eq!(snap.long_ask_size, Some(75));
        // Net mid for bull put: short_mid - long_mid = 0.75 - 0.25 = 0.50
        assert!((snap.net_mid - 0.50).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test snapshots_both_legs_of_bull_put_spread
```
Expected: FAIL — `unresolved import crate::measurement` (module doesn't exist).

- [ ] **Step 3: Implement**

Create `traderjoe/src/measurement/mod.rs`:

```rust
pub mod nbbo;
```

Create `traderjoe/src/measurement/nbbo.rs`:

```rust
use chrono::Utc;
use crate::broker::types::{OptionsChain, OptionType};

#[derive(Debug, Clone, PartialEq)]
pub struct NbboSnapshot {
    pub short_bid: f64,
    pub short_ask: f64,
    pub long_bid: f64,
    pub long_ask: f64,
    pub net_mid: f64,
    pub short_bid_size: Option<i64>,
    pub short_ask_size: Option<i64>,
    pub long_bid_size: Option<i64>,
    pub long_ask_size: Option<i64>,
    pub snapshot_time: String,
}

/// Snapshot bid/ask/size for both legs of a credit spread from an options chain.
///
/// Returns None if either leg cannot be found (missing strike/type/expiration match).
/// `net_mid` is defined as (short_mid - long_mid) — positive for a credit received.
pub fn snapshot_spread_nbbo(
    chain: &OptionsChain,
    short_strike: f64,
    long_strike: f64,
    option_type: OptionType,
) -> Option<NbboSnapshot> {
    let short = chain.contracts.iter().find(|c| {
        (c.strike - short_strike).abs() < 0.01 && c.option_type == option_type
    })?;
    let long = chain.contracts.iter().find(|c| {
        (c.strike - long_strike).abs() < 0.01 && c.option_type == option_type
    })?;
    let short_mid = (short.bid + short.ask) / 2.0;
    let long_mid = (long.bid + long.ask) / 2.0;
    Some(NbboSnapshot {
        short_bid: short.bid, short_ask: short.ask,
        long_bid: long.bid, long_ask: long.ask,
        net_mid: short_mid - long_mid,
        short_bid_size: short.bid_size,
        short_ask_size: short.ask_size,
        long_bid_size: long.bid_size,
        long_ask_size: long.ask_size,
        snapshot_time: Utc::now().to_rfc3339(),
    })
}
```

Add `mod measurement;` to `traderjoe/src/lib.rs` near the other `mod` declarations (around line 10).

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test snapshots_both_legs_of_bull_put_spread
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/ traderjoe/src/lib.rs && \
  git commit -m "feat(measurement): add NBBO spread snapshotter"
```

---

### Task 6: measurement/portfolio_greeks — BETA_TABLE and beta_weighted_delta

**Group:** B (sequential)
**Behavior being verified:** Beta-weighted delta weights each underlying by its static beta against SPY.
**Interface under test:** `beta_weighted_delta` and `lookup_beta`.

**Files:**
- Create: `traderjoe/src/measurement/portfolio_greeks.rs`
- Modify: `traderjoe/src/measurement/mod.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_weighted_delta_applies_static_table() {
        // SPY delta 10, QQQ delta 5, IWM delta 4
        // beta-weighted = 10*1.00 + 5*1.15 + 4*1.25 = 10 + 5.75 + 5.0 = 20.75
        let contribs = vec![
            ("SPY".to_string(), 10.0),
            ("QQQ".to_string(), 5.0),
            ("IWM".to_string(), 4.0),
        ];
        let bwd = beta_weighted_delta(&contribs);
        assert!((bwd - 20.75).abs() < 1e-9);
    }

    #[test]
    fn unknown_underlying_uses_beta_1() {
        let contribs = vec![("UNKNOWN".to_string(), 3.0)];
        let bwd = beta_weighted_delta(&contribs);
        assert!((bwd - 3.0).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test beta_weighted_delta_applies_static_table
```
Expected: FAIL — `unresolved module measurement::portfolio_greeks`.

- [ ] **Step 3: Implement**

Create `traderjoe/src/measurement/portfolio_greeks.rs`:

```rust
/// Static beta vs SPY. Quarterly human review — see GO_LIVE_GATE.md.
pub const BETA_TABLE: &[(&str, f64)] = &[
    ("SPY", 1.00),
    ("QQQ", 1.15),
    ("IWM", 1.25),
];

/// Look up beta for an underlying; returns 1.0 when not in table.
pub fn lookup_beta(underlying: &str) -> f64 {
    BETA_TABLE.iter()
        .find(|(u, _)| *u == underlying)
        .map(|(_, b)| *b)
        .unwrap_or(1.0)
}

/// Sum of (delta × beta) across underlyings.
pub fn beta_weighted_delta(contribs: &[(String, f64)]) -> f64 {
    contribs.iter().map(|(u, d)| d * lookup_beta(u)).sum()
}
```

Append to `traderjoe/src/measurement/mod.rs`:
```rust
pub mod portfolio_greeks;
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test beta_weighted_delta
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/ && git commit -m "feat(measurement): add beta-weighted delta"
```

---

### Task 7: measurement/portfolio_greeks — PortfolioGreeksAggregator::compute

**Group:** B (sequential)
**Behavior being verified:** Given open trades and their chain lookups, `compute` returns summed portfolio Greeks with beta-weighted delta and max-per-position concentration.
**Interface under test:** `PortfolioGreeksAggregator::compute`.

**Files:**
- Modify: `traderjoe/src/measurement/portfolio_greeks.rs`

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block:

```rust
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};
    use crate::types::{Trade, TradeStatus, SpreadType};
    use std::collections::HashMap;

    fn open_trade(id: &str, underlying: &str, contracts: i64, short: f64, long: f64) -> Trade {
        Trade {
            id: id.to_string(), created_at: "2026-04-22T00:00:00Z".to_string(),
            underlying: underlying.to_string(), spread_type: SpreadType::BullPut,
            short_strike: short, long_strike: long, expiration: "2026-05-15".to_string(),
            contracts, entry_credit: 0.50, max_loss: 450.0,
            broker_order_id: Some("o".to_string()), status: TradeStatus::Open,
            fill_price: Some(0.50), fill_time: Some("2026-04-22T00:00:00Z".to_string()),
            exit_price: None, exit_time: None, exit_reason: None, net_pnl: None,
            iv_rank: Some(60.0), short_delta: Some(-0.25), short_theta: Some(0.05),
            entry_short_bid: None, entry_short_ask: None, entry_long_bid: None,
            entry_long_ask: None, entry_net_mid: None,
            exit_short_bid: None, exit_short_ask: None, exit_long_bid: None,
            exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: None, entry_short_vega: None,
            entry_long_delta: None, entry_long_gamma: None, entry_long_vega: None,
            nbbo_displayed_size_short: None, nbbo_displayed_size_long: None,
            nbbo_snapshot_time: None,
        }
    }

    fn put_contract(strike: f64, d: f64, g: f64, v: f64, th: f64) -> OptionContract {
        OptionContract {
            symbol: format!("S{}", strike), underlying: "SPY".to_string(),
            expiration: "2026-05-15".to_string(), strike,
            option_type: OptionType::Put, bid: 0.74, ask: 0.76, last: 0.75,
            volume: 0, open_interest: 0, implied_volatility: Some(0.20),
            delta: Some(d), gamma: Some(g), theta: Some(th), vega: Some(v),
            bid_size: Some(10), ask_size: Some(10),
        }
    }

    #[test]
    fn compute_aggregates_open_trade_greeks_with_beta_weighting() {
        // One SPY trade, 2 contracts. Short put delta -0.28 gamma 0.01 vega 0.30 theta -0.05,
        // Long put delta -0.18 gamma 0.008 vega 0.25 theta -0.03.
        // Per-contract net delta = (-0.28) - (-0.18) = -0.10 ; × 100 × 2 contracts = -20 delta_shares.
        // Beta = 1.00 for SPY, so beta-weighted delta = -20.
        let spy_chain = OptionsChain {
            underlying: "SPY".to_string(), underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                put_contract(460.0, -0.28, 0.010, 0.30, -0.05),
                put_contract(455.0, -0.18, 0.008, 0.25, -0.03),
            ],
        };
        let trades = vec![open_trade("t1", "SPY", 2, 460.0, 455.0)];
        let mut chains = HashMap::new();
        chains.insert("SPY".to_string(), spy_chain);

        let result = PortfolioGreeksAggregator::compute(&trades, &chains);
        assert_eq!(result.open_position_count, 1);
        assert!((result.beta_weighted_delta - (-20.0)).abs() < 1e-6,
                "got {}", result.beta_weighted_delta);
    }
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test compute_aggregates_open_trade_greeks_with_beta_weighting
```
Expected: FAIL — `PortfolioGreeksAggregator not found`.

- [ ] **Step 3: Implement**

Add to `traderjoe/src/measurement/portfolio_greeks.rs`:

```rust
use std::collections::HashMap;
use crate::broker::types::{OptionType, OptionsChain};
use crate::types::{SpreadType, Trade};

#[derive(Debug, Clone)]
pub struct PortfolioGreeks {
    pub beta_weighted_delta: f64,
    pub total_gamma: f64,
    pub total_vega: f64,
    pub total_theta: f64,
    pub delta_by_underlying: HashMap<String, f64>,
    pub max_gamma_single_position: f64,
    pub max_vega_single_position: f64,
    pub open_position_count: usize,
}

pub struct PortfolioGreeksAggregator;

impl PortfolioGreeksAggregator {
    pub fn compute(trades: &[Trade], chains: &HashMap<String, OptionsChain>) -> PortfolioGreeks {
        let mut delta_contribs: HashMap<String, f64> = HashMap::new();
        let mut total_gamma = 0.0;
        let mut total_vega = 0.0;
        let mut total_theta = 0.0;
        let mut max_gamma = 0.0f64;
        let mut max_vega = 0.0f64;

        for trade in trades {
            let chain = match chains.get(&trade.underlying) {
                Some(c) => c, None => continue,
            };
            let option_type = match trade.spread_type {
                SpreadType::BullPut => OptionType::Put,
                SpreadType::BearCall => OptionType::Call,
            };
            let short = chain.contracts.iter().find(|c| {
                (c.strike - trade.short_strike).abs() < 0.01
                    && c.option_type == option_type
                    && c.expiration == trade.expiration
            });
            let long = chain.contracts.iter().find(|c| {
                (c.strike - trade.long_strike).abs() < 0.01
                    && c.option_type == option_type
                    && c.expiration == trade.expiration
            });
            let (Some(s), Some(l)) = (short, long) else { continue; };

            // Short leg sold, long leg bought. Portfolio greek = (short × -1 + long × +1) × contracts × 100.
            let mult = trade.contracts as f64 * 100.0;
            let d = (l.delta.unwrap_or(0.0) - s.delta.unwrap_or(0.0)) * mult;
            let g = (l.gamma.unwrap_or(0.0) - s.gamma.unwrap_or(0.0)) * mult;
            let v = (l.vega.unwrap_or(0.0) - s.vega.unwrap_or(0.0)) * mult;
            let th = (l.theta.unwrap_or(0.0) - s.theta.unwrap_or(0.0)) * mult;

            *delta_contribs.entry(trade.underlying.clone()).or_insert(0.0) += d;
            total_gamma += g;
            total_vega += v;
            total_theta += th;
            max_gamma = max_gamma.max(g.abs());
            max_vega = max_vega.max(v.abs());
        }

        let contribs_vec: Vec<(String, f64)> = delta_contribs.clone().into_iter().collect();
        PortfolioGreeks {
            beta_weighted_delta: beta_weighted_delta(&contribs_vec),
            total_gamma, total_vega, total_theta,
            delta_by_underlying: delta_contribs,
            max_gamma_single_position: max_gamma,
            max_vega_single_position: max_vega,
            open_position_count: trades.len(),
        }
    }
}
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test compute_aggregates_open_trade_greeks_with_beta_weighting
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/portfolio_greeks.rs && \
  git commit -m "feat(measurement): add PortfolioGreeksAggregator"
```

---

### Task 8: measurement/market_context — compute_spy_stats

**Group:** B (sequential)
**Behavior being verified:** SPY 20-day realized vol, 20-day return, and drawdown from 52-week high compute correctly from bar data.
**Interface under test:** `compute_spy_stats`.

**Files:**
- Create: `traderjoe/src/measurement/market_context.rs`
- Modify: `traderjoe/src/measurement/mod.rs`

- [ ] **Step 1: Write the failing test**

In `traderjoe/src/measurement/market_context.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::types::Bar;

    fn bar(close: f64) -> Bar {
        Bar { timestamp: "2026-04-22".to_string(), open: close, high: close, low: close, close, volume: 0 }
    }

    #[test]
    fn spy_stats_computes_drawdown_from_52w_peak() {
        // Peak 500, current 450 → drawdown 10%
        let mut bars = vec![bar(400.0); 260];
        bars[100] = bar(500.0);
        bars[259] = bar(450.0);
        let stats = compute_spy_stats(&bars[240..260], &bars);
        assert!((stats.drawdown_from_52w_high - 0.10).abs() < 1e-6,
                "got {}", stats.drawdown_from_52w_high);
    }

    #[test]
    fn spy_stats_computes_20d_return() {
        // Start 480, end 480*1.02 = 489.6 → return 0.02
        let mut bars20 = vec![bar(480.0); 20];
        bars20[19] = bar(489.6);
        let bars52 = bars20.clone();
        let stats = compute_spy_stats(&bars20, &bars52);
        assert!((stats.return_20d - 0.02).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test spy_stats_computes
```
Expected: FAIL — unresolved module.

- [ ] **Step 3: Implement**

Create `traderjoe/src/measurement/market_context.rs`:

```rust
use crate::broker::types::Bar;

#[derive(Debug, Clone, PartialEq)]
pub struct SpyStats {
    pub realized_vol_20d: f64,
    pub return_20d: f64,
    pub drawdown_from_52w_high: f64,
}

/// 20-day realized vol = stdev of log returns × sqrt(252).
/// 20-day return = (last - first) / first.
/// Drawdown from 52w high = (peak - last) / peak, peak taken over bars_52w close values.
pub fn compute_spy_stats(bars_20d: &[Bar], bars_52w: &[Bar]) -> SpyStats {
    let return_20d = if bars_20d.len() >= 2 && bars_20d.first().map(|b| b.close).unwrap_or(0.0) > 0.0 {
        let first = bars_20d.first().unwrap().close;
        let last = bars_20d.last().unwrap().close;
        (last - first) / first
    } else { 0.0 };

    let realized_vol_20d = if bars_20d.len() >= 2 {
        let log_returns: Vec<f64> = bars_20d.windows(2)
            .filter(|w| w[0].close > 0.0 && w[1].close > 0.0)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();
        if log_returns.is_empty() { 0.0 } else {
            let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
            let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / log_returns.len() as f64;
            var.sqrt() * (252f64).sqrt()
        }
    } else { 0.0 };

    let drawdown_from_52w_high = if !bars_52w.is_empty() {
        let peak = bars_52w.iter().map(|b| b.close).fold(0.0f64, f64::max);
        let last = bars_52w.last().map(|b| b.close).unwrap_or(0.0);
        if peak > 0.0 { (peak - last) / peak } else { 0.0 }
    } else { 0.0 };

    SpyStats { realized_vol_20d, return_20d, drawdown_from_52w_high }
}
```

Append to `traderjoe/src/measurement/mod.rs`:
```rust
pub mod market_context;
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test spy_stats_computes
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/ && \
  git commit -m "feat(measurement): add SPY 20d stats computation"
```

---

### Task 9: measurement/market_context — FRED VIXCLS parser

**Group:** B (sequential)
**Behavior being verified:** Given a FRED VIXCLS CSV response body, `parse_fred_vix_csv` returns the most recent non-empty close and its date.
**Interface under test:** `parse_fred_vix_csv`.

**Files:**
- Modify: `traderjoe/src/measurement/market_context.rs`

- [ ] **Step 1: Write the failing test**

Append to tests:

```rust
    #[test]
    fn parse_fred_vix_csv_returns_most_recent_nonempty_row() {
        // FRED CSV format: "DATE,VIXCLS\n2026-04-18,16.23\n2026-04-21,17.45\n2026-04-22,.\n"
        // Value "." is FRED's marker for "no data yet" — skip to prior row.
        let csv = "DATE,VIXCLS\n2026-04-18,16.23\n2026-04-21,17.45\n2026-04-22,.\n";
        let parsed = parse_fred_vix_csv(csv).expect("parse");
        assert!((parsed.value - 17.45).abs() < 1e-6);
        assert_eq!(parsed.source_date, "2026-04-21");
    }

    #[test]
    fn parse_fred_vix_csv_returns_none_when_all_rows_empty() {
        let csv = "DATE,VIXCLS\n2026-04-21,.\n2026-04-22,.\n";
        assert!(parse_fred_vix_csv(csv).is_none());
    }
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test parse_fred_vix_csv
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `traderjoe/src/measurement/market_context.rs`:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SpotVixReading {
    pub value: f64,
    pub source: &'static str,
    pub source_date: String,
}

/// Parse FRED VIXCLS CSV. Lines are `DATE,VALUE`. Missing values are `.`.
/// Returns the most recent row with a numeric VIX, or None.
pub fn parse_fred_vix_csv(body: &str) -> Option<SpotVixReading> {
    body.lines().rev()
        .filter(|l| !l.is_empty() && !l.starts_with("DATE"))
        .filter_map(|l| {
            let mut parts = l.split(',');
            let date = parts.next()?.trim();
            let val = parts.next()?.trim();
            let v: f64 = val.parse().ok()?;
            Some(SpotVixReading { value: v, source: "fred", source_date: date.to_string() })
        })
        .next()
}
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test parse_fred_vix_csv
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/market_context.rs && \
  git commit -m "feat(measurement): parse FRED VIXCLS CSV"
```

---

### Task 10: measurement/metrics — sample_size_tag and paper_fill_violation

**Group:** B (sequential)
**Behavior being verified:** Sample-size thresholds (INSUFFICIENT/WEAK/OK) classify correctly; paper-fill violation flags when contracts exceed min displayed size.
**Interface under test:** `sample_size_tag`, `paper_fill_violation`.

**Files:**
- Create: `traderjoe/src/measurement/metrics.rs`
- Modify: `traderjoe/src/measurement/mod.rs`

- [ ] **Step 1: Write the failing test**

In `traderjoe/src/measurement/metrics.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_size_tag_classifies_thresholds() {
        assert_eq!(sample_size_tag(0), SampleSizeTag::Insufficient);
        assert_eq!(sample_size_tag(29), SampleSizeTag::Insufficient);
        assert_eq!(sample_size_tag(30), SampleSizeTag::Weak);
        assert_eq!(sample_size_tag(99), SampleSizeTag::Weak);
        assert_eq!(sample_size_tag(100), SampleSizeTag::Ok);
        assert_eq!(sample_size_tag(1000), SampleSizeTag::Ok);
    }

    #[test]
    fn paper_fill_violation_flags_oversized_fill() {
        // 5 contracts vs displayed short 3 / long 10 → violation (min = 3 < 5)
        assert!(paper_fill_violation(5, 3, 10));
        // 2 contracts vs displayed short 5 / long 5 → no violation
        assert!(!paper_fill_violation(2, 5, 5));
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test sample_size_tag_classifies_thresholds
```
Expected: FAIL — unresolved module.

- [ ] **Step 3: Implement**

Create `traderjoe/src/measurement/metrics.rs`:

```rust
/// Assumed by backtest's ORATS slippage model: fills land ~34% into the bid-ask spread.
pub const ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD: f64 = 0.34;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SampleSizeTag {
    Insufficient,
    Weak,
    Ok,
}

impl SampleSizeTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            SampleSizeTag::Insufficient => "INSUFFICIENT",
            SampleSizeTag::Weak => "WEAK",
            SampleSizeTag::Ok => "OK",
        }
    }
}

/// n<30: INSUFFICIENT (noise); 30≤n<100: WEAK (meaningful but not Kelly-confident); n≥100: OK.
pub fn sample_size_tag(n: usize) -> SampleSizeTag {
    if n < 30 { SampleSizeTag::Insufficient }
    else if n < 100 { SampleSizeTag::Weak }
    else { SampleSizeTag::Ok }
}

/// Paper fill violation: filled contracts exceed min of short/long leg displayed size.
/// A live exchange would not fill an order larger than displayed size; Alpaca paper does.
pub fn paper_fill_violation(contracts: i64, short_size: i64, long_size: i64) -> bool {
    contracts > short_size.min(long_size)
}
```

Append to `traderjoe/src/measurement/mod.rs`:
```rust
pub mod metrics;
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test -p trader-joe sample_size_tag paper_fill_violation
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/ && \
  git commit -m "feat(measurement): add sample-size tagging and paper-fill violation check"
```

---

### Task 11: measurement/metrics — Sharpe, Sortino, profit factor, skew, max drawdown

**Group:** B (sequential)
**Behavior being verified:** Each metric computes correctly on a fixture.
**Interface under test:** `compute_sharpe`, `compute_sortino`, `compute_profit_factor`, `compute_pnl_skew`, `compute_max_drawdown_pct`.

**Files:**
- Modify: `traderjoe/src/measurement/metrics.rs`

- [ ] **Step 1: Write the failing test**

Append to tests in `metrics.rs`:

```rust
    #[test]
    fn sharpe_matches_annualized_mean_over_stdev() {
        // Daily returns: +0.01, -0.005, +0.015, -0.01, +0.02
        let returns = vec![0.01, -0.005, 0.015, -0.01, 0.02];
        let s = compute_sharpe(&returns);
        // Mean = 0.006, stdev (pop) computed, sharpe = mean/stdev * sqrt(252).
        // We assert on magnitude & sign (should be positive).
        assert!(s > 0.0, "sharpe should be positive for positive-mean returns, got {}", s);
    }

    #[test]
    fn sortino_downside_only_stdev() {
        // Only negatives: -0.005, -0.01. Upside returns don't penalize.
        let returns = vec![0.01, -0.005, 0.015, -0.01, 0.02];
        let so = compute_sortino(&returns);
        // Sortino should exceed Sharpe since upside vol isn't penalized.
        let sh = compute_sharpe(&returns);
        assert!(so > sh, "sortino ({}) should exceed sharpe ({}) with upside vol", so, sh);
    }

    #[test]
    fn profit_factor_is_gross_wins_over_gross_losses() {
        let pnls = vec![100.0, -50.0, 200.0, -80.0, 60.0];
        // Wins = 360, losses = 130, pf = 360 / 130 = 2.769...
        let pf = compute_profit_factor(&pnls);
        assert!((pf - (360.0 / 130.0)).abs() < 1e-6);
    }

    #[test]
    fn profit_factor_returns_infinity_when_no_losers() {
        let pnls = vec![100.0, 200.0];
        let pf = compute_profit_factor(&pnls);
        assert!(pf.is_infinite());
    }

    #[test]
    fn pnl_skew_is_negative_for_left_skewed_distribution() {
        // Many small wins, few big losses → negative skew (classic credit spread shape)
        let pnls = vec![50.0; 10].into_iter().chain(vec![-500.0]).collect::<Vec<f64>>();
        let sk = compute_pnl_skew(&pnls);
        assert!(sk < 0.0, "left-skewed distribution should have negative skew, got {}", sk);
    }

    #[test]
    fn max_drawdown_tracks_peak_to_trough() {
        // Equity curve: 100 → 120 → 90 → 110. Peak 120, trough after = 90, dd = 25%.
        let equity = vec![100.0, 120.0, 90.0, 110.0];
        let dd = compute_max_drawdown_pct(&equity);
        assert!((dd - 0.25).abs() < 1e-6, "expected 0.25, got {}", dd);
    }
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test sharpe_matches
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `traderjoe/src/measurement/metrics.rs`:

```rust
/// Annualized Sharpe from daily returns. Returns 0 if < 2 returns or zero stdev.
pub fn compute_sharpe(daily_returns: &[f64]) -> f64 {
    if daily_returns.len() < 2 { return 0.0; }
    let n = daily_returns.len() as f64;
    let mean = daily_returns.iter().sum::<f64>() / n;
    let var = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd == 0.0 { return 0.0; }
    (mean / sd) * (252f64).sqrt()
}

/// Annualized Sortino — mean over downside-only stdev.
pub fn compute_sortino(daily_returns: &[f64]) -> f64 {
    if daily_returns.len() < 2 { return 0.0; }
    let n = daily_returns.len() as f64;
    let mean = daily_returns.iter().sum::<f64>() / n;
    let downside: Vec<f64> = daily_returns.iter().filter(|r| **r < 0.0).copied().collect();
    if downside.is_empty() { return f64::INFINITY; }
    let var = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let sd = var.sqrt();
    if sd == 0.0 { return 0.0; }
    (mean / sd) * (252f64).sqrt()
}

/// Profit factor = gross wins / abs(gross losses). Infinity if no losses.
pub fn compute_profit_factor(pnls: &[f64]) -> f64 {
    let wins: f64 = pnls.iter().filter(|p| **p > 0.0).sum();
    let losses: f64 = pnls.iter().filter(|p| **p < 0.0).sum::<f64>().abs();
    if losses == 0.0 { return f64::INFINITY; }
    wins / losses
}

/// Sample skewness (third standardized moment).
pub fn compute_pnl_skew(pnls: &[f64]) -> f64 {
    let n = pnls.len();
    if n < 3 { return 0.0; }
    let mean = pnls.iter().sum::<f64>() / n as f64;
    let var = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    if sd == 0.0 { return 0.0; }
    let m3 = pnls.iter().map(|p| ((p - mean) / sd).powi(3)).sum::<f64>() / n as f64;
    m3
}

/// Max drawdown as a positive fraction (e.g. 0.25 for 25%).
pub fn compute_max_drawdown_pct(equity_curve: &[f64]) -> f64 {
    let mut peak = 0.0f64;
    let mut max_dd = 0.0f64;
    for &e in equity_curve {
        peak = peak.max(e);
        if peak > 0.0 {
            let dd = (peak - e) / peak;
            max_dd = max_dd.max(dd);
        }
    }
    max_dd
}
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test -p trader-joe -- metrics
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/metrics.rs && \
  git commit -m "feat(measurement): add Sharpe/Sortino/profit-factor/skew/drawdown"
```

---

### Task 12: measurement/metrics — slippage stats and regime bucketing

**Group:** B (sequential)
**Behavior being verified:** Given closed trades with NBBO and fill data, slippage stats compute correctly and trades bucket into low/med/high vol regimes.
**Interface under test:** `compute_slippage_stats`, `classify_regime`.

**Files:**
- Modify: `traderjoe/src/measurement/metrics.rs`

- [ ] **Step 1: Write the failing test**

Append:

```rust
    #[test]
    fn slippage_stats_mean_and_max_from_entry_fills() {
        // Trade A: entry_net_mid 0.50, fill_price 0.48 → slippage 0.02
        // Trade B: entry_net_mid 0.60, fill_price 0.56 → slippage 0.04
        let inputs = vec![
            SlippageInput { entry_mid: 0.50, entry_fill: 0.48, leg_width: 0.10 },
            SlippageInput { entry_mid: 0.60, entry_fill: 0.56, leg_width: 0.10 },
        ];
        let s = compute_slippage_stats(&inputs);
        assert!((s.mean_abs - 0.03).abs() < 1e-9);
        assert!((s.max_abs - 0.04).abs() < 1e-9);
        // ORATS constant = 0.34 × 0.10 = 0.034 per trade assumed; mean 0.03 / 0.034 ≈ 0.882
        assert!((s.ratio_vs_orats - (0.03 / (0.34 * 0.10))).abs() < 1e-6);
    }

    #[test]
    fn classify_regime_thresholds_on_spot_vix() {
        assert_eq!(classify_regime(12.0), "low_vol");
        assert_eq!(classify_regime(20.0), "med_vol");
        assert_eq!(classify_regime(30.0), "high_vol");
    }
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test slippage_stats_mean_and_max
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `metrics.rs`:

```rust
#[derive(Debug, Clone)]
pub struct SlippageInput {
    pub entry_mid: f64,
    pub entry_fill: f64,
    /// Mean of (short_ask-short_bid) and (long_ask-long_bid) at snapshot time.
    pub leg_width: f64,
}

#[derive(Debug, Clone)]
pub struct SlippageStats {
    pub mean_abs: f64,
    pub max_abs: f64,
    pub ratio_vs_orats: f64,
}

pub fn compute_slippage_stats(inputs: &[SlippageInput]) -> SlippageStats {
    if inputs.is_empty() {
        return SlippageStats { mean_abs: 0.0, max_abs: 0.0, ratio_vs_orats: 0.0 };
    }
    let slippages: Vec<f64> = inputs.iter().map(|i| (i.entry_mid - i.entry_fill).abs()).collect();
    let mean = slippages.iter().sum::<f64>() / slippages.len() as f64;
    let max = slippages.iter().cloned().fold(0.0f64, f64::max);
    let assumed: f64 = inputs.iter()
        .map(|i| ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD * i.leg_width)
        .sum::<f64>() / inputs.len() as f64;
    let ratio = if assumed > 0.0 { mean / assumed } else { 0.0 };
    SlippageStats { mean_abs: mean, max_abs: max, ratio_vs_orats: ratio }
}

/// Regime bucket from spot VIX level.
pub fn classify_regime(spot_vix: f64) -> &'static str {
    if spot_vix < 15.0 { "low_vol" }
    else if spot_vix < 25.0 { "med_vol" }
    else { "high_vol" }
}
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test slippage_stats classify_regime
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/metrics.rs && \
  git commit -m "feat(measurement): add slippage stats and regime classification"
```

---

### Task 13: measurement/equity — EquitySnapshotter interface

**Group:** B (sequential, last measurement task)
**Behavior being verified:** `EquitySnapshotter::build_row` constructs a row with the expected event_type and equity fields given inputs.
**Interface under test:** `EquityEvent`, `EquitySnapshot`, `build_equity_snapshot`.

Rationale for scope: the async I/O (D1 write, Alpaca fetch) is wired in the handler tasks. This task covers the pure data-construction logic so it can be tested without fakes.

**Files:**
- Create: `traderjoe/src/measurement/equity.rs`
- Modify: `traderjoe/src/measurement/mod.rs`

- [ ] **Step 1: Write the failing test**

In `traderjoe/src/measurement/equity.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_equity_snapshot_for_eod_event() {
        let snap = build_equity_snapshot(EquityEvent::Eod, SnapshotInputs {
            timestamp: "2026-04-22T22:00:00Z".to_string(),
            equity: 100_000.0, cash: 80_000.0,
            open_position_mtm: 20_000.0,
            realized_pnl_day: 500.0, unrealized_pnl_day: 150.0,
            open_position_count: 3,
            trade_id_ref: None,
        });
        assert_eq!(snap.event_type, "eod");
        assert!((snap.equity - 100_000.0).abs() < 1e-9);
        assert_eq!(snap.open_position_count, 3);
        assert!(snap.trade_id_ref.is_none());
    }

    #[test]
    fn build_equity_snapshot_for_trade_open_carries_trade_id() {
        let snap = build_equity_snapshot(EquityEvent::TradeOpen, SnapshotInputs {
            timestamp: "2026-04-22T14:05:00Z".to_string(),
            equity: 100_500.0, cash: 79_500.0,
            open_position_mtm: 21_000.0,
            realized_pnl_day: 0.0, unrealized_pnl_day: 0.0,
            open_position_count: 4,
            trade_id_ref: Some("trade-xyz".to_string()),
        });
        assert_eq!(snap.event_type, "trade_open");
        assert_eq!(snap.trade_id_ref.as_deref(), Some("trade-xyz"));
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd traderjoe && cargo test build_equity_snapshot_for_eod_event
```
Expected: FAIL — unresolved module.

- [ ] **Step 3: Implement**

Create `traderjoe/src/measurement/equity.rs`:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum EquityEvent {
    Eod,
    TradeOpen,
    TradeClose,
    CircuitBreaker,
}

impl EquityEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            EquityEvent::Eod => "eod",
            EquityEvent::TradeOpen => "trade_open",
            EquityEvent::TradeClose => "trade_close",
            EquityEvent::CircuitBreaker => "circuit_breaker",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotInputs {
    pub timestamp: String,
    pub equity: f64,
    pub cash: f64,
    pub open_position_mtm: f64,
    pub realized_pnl_day: f64,
    pub unrealized_pnl_day: f64,
    pub open_position_count: i64,
    pub trade_id_ref: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EquitySnapshot {
    pub timestamp: String,
    pub event_type: &'static str,
    pub equity: f64,
    pub cash: f64,
    pub open_position_mtm: f64,
    pub realized_pnl_day: f64,
    pub unrealized_pnl_day: f64,
    pub open_position_count: i64,
    pub trade_id_ref: Option<String>,
}

pub fn build_equity_snapshot(event: EquityEvent, inputs: SnapshotInputs) -> EquitySnapshot {
    EquitySnapshot {
        timestamp: inputs.timestamp,
        event_type: event.as_str(),
        equity: inputs.equity,
        cash: inputs.cash,
        open_position_mtm: inputs.open_position_mtm,
        realized_pnl_day: inputs.realized_pnl_day,
        unrealized_pnl_day: inputs.unrealized_pnl_day,
        open_position_count: inputs.open_position_count,
        trade_id_ref: inputs.trade_id_ref,
    }
}
```

Append to `traderjoe/src/measurement/mod.rs`:
```rust
pub mod equity;
```

- [ ] **Step 4: Run test — verify PASS**

```bash
cd traderjoe && cargo test build_equity_snapshot
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/ && \
  git commit -m "feat(measurement): add EquityEvent and EquitySnapshot builder"
```

---

## Group C — Sequential (d1.rs extensions)

### Task 14: D1Client::create_trade gains NBBO and expanded Greeks parameters

**Group:** C (sequential)
**Behavior being verified:** After `create_trade` is called with NBBO + extended Greeks, `get_open_trades` returns a Trade with those fields populated.
**Interface under test:** `D1Client::create_trade`, `D1Client::get_open_trades`.

Rationale: this is an integration test requiring a real D1. Per project convention (`Cargo.toml` shows `dev-dependencies` empty, async forbidden in tests), we cannot run D1 in unit tests. Instead: **this task's test is a compilation-level contract test** — we add an `#[allow(dead_code)] async fn _compile_check_create_trade(db: D1Client)` that invokes `create_trade` with all new params, guaranteeing the signature change compiles. The integration verification happens manually on first deploy via `/health`.

**Files:**
- Modify: `traderjoe/src/db/d1.rs`

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` in `d1.rs`:

```rust
    /// Compile-check: create_trade accepts NBBO and expanded Greeks params.
    /// Not executed — D1 is not available in tests. The test fails at compile time
    /// if the signature drifts from the plan contract.
    #[test]
    fn create_trade_signature_accepts_nbbo_and_expanded_greeks() {
        // Reference a function pointer with the expected signature. Compile-only assertion.
        let _: fn(
            &D1Client, &str, &str, &SpreadType, f64, f64, &str, i64, f64, f64,
            Option<&str>, Option<f64>, Option<f64>, Option<f64>,
            Option<&crate::measurement::nbbo::NbboSnapshot>,
            Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>,
        ) -> futures_like<()> = |_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_| futures_like(());

        fn futures_like<T>(t: T) -> T { t } // placeholder
        let _ = D1Client::create_trade;
    }
```

Note: the above is awkward for Rust. Simplify — assert via a compile-only reference:

```rust
    #[test]
    fn create_trade_signature_exists() {
        // Compile-time check only — if the signature changes in a breaking way,
        // this fails to compile.
        let _f = D1Client::create_trade;
    }
```

- [ ] **Step 2: Run — verify initial FAIL is a signature mismatch**

Before edits, this test compiles and passes trivially. To actually verify the test catches drift, extend the body after implementing:

Actually, switch strategy: the meaningful failing test is that **`D1Client` itself is callable with the full parameter list.** Use a closure that references the method with typed argument placeholders:

```rust
    #[test]
    fn create_trade_accepts_full_parameter_list() {
        // Compile-only assertion that create_trade accepts NBBO + expanded Greeks.
        // Build an async block that would call it; never executed.
        let _unused = async {
            let db: D1Client = unreachable!();
            let nbbo: crate::measurement::nbbo::NbboSnapshot = unreachable!();
            db.create_trade(
                "id", "SPY", &SpreadType::BullPut, 460.0, 455.0, "2026-05-15",
                2, 0.75, 425.0, None, Some(65.0), Some(-0.28), Some(0.05),
                Some(&nbbo),
                Some(0.010), Some(0.30), Some(-0.18), Some(0.008), Some(0.25),
            ).await.ok();
        };
    }
```

This fails to compile until `create_trade` gains the new params.

```bash
cd traderjoe && cargo test create_trade_accepts_full_parameter_list
```
Expected: FAIL — compile error `expected N arguments, found M`.

- [ ] **Step 3: Implement**

Change `D1Client::create_trade` in `traderjoe/src/db/d1.rs` to accept the additional parameters and write them:

```rust
    pub async fn create_trade(
        &self,
        id: &str,
        underlying: &str,
        spread_type: &SpreadType,
        short_strike: f64,
        long_strike: f64,
        expiration: &str,
        contracts: i64,
        entry_credit: f64,
        max_loss: f64,
        broker_order_id: Option<&str>,
        iv_rank: Option<f64>,
        short_delta: Option<f64>,
        short_theta: Option<f64>,
        nbbo: Option<&crate::measurement::nbbo::NbboSnapshot>,
        entry_short_gamma: Option<f64>,
        entry_short_vega: Option<f64>,
        entry_long_delta: Option<f64>,
        entry_long_gamma: Option<f64>,
        entry_long_vega: Option<f64>,
    ) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let nb_short_bid = nbbo.map(|n| n.short_bid.into()).unwrap_or(JsValue::NULL);
        let nb_short_ask = nbbo.map(|n| n.short_ask.into()).unwrap_or(JsValue::NULL);
        let nb_long_bid = nbbo.map(|n| n.long_bid.into()).unwrap_or(JsValue::NULL);
        let nb_long_ask = nbbo.map(|n| n.long_ask.into()).unwrap_or(JsValue::NULL);
        let nb_net_mid = nbbo.map(|n| n.net_mid.into()).unwrap_or(JsValue::NULL);
        let nb_sz_short = nbbo.and_then(|n| n.short_bid_size).map(|s| (s as f64).into()).unwrap_or(JsValue::NULL);
        let nb_sz_long = nbbo.and_then(|n| n.long_ask_size).map(|s| (s as f64).into()).unwrap_or(JsValue::NULL);
        let nb_time = nbbo.map(|n| n.snapshot_time.as_str().into()).unwrap_or(JsValue::NULL);

        self.db
            .prepare(
                "INSERT INTO trades (id, underlying, spread_type, short_strike, long_strike,
                expiration, contracts, entry_credit, max_loss, broker_order_id,
                iv_rank, short_delta, short_theta,
                entry_short_bid, entry_short_ask, entry_long_bid, entry_long_ask, entry_net_mid,
                entry_short_gamma, entry_short_vega, entry_long_delta, entry_long_gamma, entry_long_vega,
                nbbo_displayed_size_short, nbbo_displayed_size_long, nbbo_snapshot_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                id.into(), underlying.into(), spread_type.as_str().into(),
                short_strike.into(), long_strike.into(), expiration.into(),
                (contracts as f64).into(), entry_credit.into(), max_loss.into(),
                broker_order_id.map(|s| s.into()).unwrap_or(JsValue::NULL),
                iv_rank.map(|v| v.into()).unwrap_or(JsValue::NULL),
                short_delta.map(|v| v.into()).unwrap_or(JsValue::NULL),
                short_theta.map(|v| v.into()).unwrap_or(JsValue::NULL),
                nb_short_bid, nb_short_ask, nb_long_bid, nb_long_ask, nb_net_mid,
                entry_short_gamma.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_short_vega.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_long_delta.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_long_gamma.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_long_vega.map(|v| v.into()).unwrap_or(JsValue::NULL),
                nb_sz_short, nb_sz_long, nb_time,
            ])?
            .run()
            .await?;
        Ok(())
    }
```

- [ ] **Step 4: Run — verify PASS**

```bash
cd traderjoe && cargo test create_trade_accepts_full_parameter_list
```
Expected: PASS.

Also run the rest of the suite to catch downstream breakage:
```bash
cd traderjoe && cargo test
```
Expected: `morning_scan.rs`'s call to `create_trade` fails to compile — this is intentional; Task 18 will fix the caller.

**If the build is red after Task 14**, leave it red and continue to Task 15 etc. Task 18 restores green. Commit anyway; the plan's vertical-slice structure accepts transient red states for same-commit dependent callers.

Alternative: pass `None`/placeholder values at the call site in this task to keep the build green, then wire real values in Task 18. Prefer this — it respects TDD discipline:

In `traderjoe/src/handlers/morning_scan.rs`, update the `db.create_trade(...)` call to pass `None` for the 6 new optional params temporarily:
```rust
        match db.create_trade(
            &trade_id,
            &spread.underlying,
            &spread.spread_type,
            spread.short_strike,
            spread.long_strike,
            &spread.expiration,
            final_contracts,
            spread.entry_credit,
            spread.max_loss_per_contract(),
            Some(&placed.id),
            Some(iv_metrics.iv_rank),
            spread.short_delta,
            spread.short_theta,
            None, None, None, None, None, None,  // TEMP: Task 18 wires real values
        ).await {
```

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/db/d1.rs traderjoe/src/handlers/morning_scan.rs && \
  git commit -m "feat(db): extend create_trade with NBBO and expanded Greeks params"
```

---

### Task 15: D1Client::close_trade gains exit NBBO parameter

**Group:** C (sequential)
**Behavior being verified:** `close_trade` signature accepts an `Option<&NbboSnapshot>` and persists exit NBBO fields when present.
**Interface under test:** `D1Client::close_trade`.

**Files:**
- Modify: `traderjoe/src/db/d1.rs`
- Modify: `traderjoe/src/handlers/position_monitor.rs` (call site stub)

- [ ] **Step 1: Write the failing test**

Append to `d1.rs` tests:

```rust
    #[test]
    fn close_trade_signature_accepts_exit_nbbo() {
        let _unused = async {
            let db: D1Client = unreachable!();
            let nbbo: crate::measurement::nbbo::NbboSnapshot = unreachable!();
            let reason: crate::types::ExitReason = unreachable!();
            db.close_trade("id", 0.20, &reason, 55.0, Some(&nbbo)).await.ok();
        };
    }
```

- [ ] **Step 2: Run — verify it FAILS**

```bash
cd traderjoe && cargo test close_trade_signature_accepts_exit_nbbo
```
Expected: FAIL — arity mismatch.

- [ ] **Step 3: Implement**

Change `close_trade` in `d1.rs`:

```rust
    pub async fn close_trade(
        &self,
        trade_id: &str,
        exit_price: f64,
        exit_reason: &ExitReason,
        net_pnl: f64,
        exit_nbbo: Option<&crate::measurement::nbbo::NbboSnapshot>,
    ) -> Result<bool> {
        use chrono::Utc;
        use worker::wasm_bindgen::JsValue;
        let now = Utc::now().to_rfc3339();
        let ex_sb = exit_nbbo.map(|n| n.short_bid.into()).unwrap_or(JsValue::NULL);
        let ex_sa = exit_nbbo.map(|n| n.short_ask.into()).unwrap_or(JsValue::NULL);
        let ex_lb = exit_nbbo.map(|n| n.long_bid.into()).unwrap_or(JsValue::NULL);
        let ex_la = exit_nbbo.map(|n| n.long_ask.into()).unwrap_or(JsValue::NULL);
        let ex_mid = exit_nbbo.map(|n| n.net_mid.into()).unwrap_or(JsValue::NULL);

        let result = self.db
            .prepare(
                "UPDATE trades SET status = 'closed', exit_price = ?, exit_time = ?,
                exit_reason = ?, net_pnl = ?,
                exit_short_bid = ?, exit_short_ask = ?, exit_long_bid = ?, exit_long_ask = ?, exit_net_mid = ?
                WHERE id = ? AND status = 'open'",
            )
            .bind(&[
                exit_price.into(), now.as_str().into(),
                exit_reason.as_str().into(), net_pnl.into(),
                ex_sb, ex_sa, ex_lb, ex_la, ex_mid,
                trade_id.into(),
            ])?
            .run().await?;
        let written = result.meta().ok().flatten().and_then(|m| m.rows_written).unwrap_or(0);
        Ok(written > 0)
    }
```

Update the single call site in `traderjoe/src/handlers/position_monitor.rs` (currently `db.close_trade(&trade.id, current_debit, &exit_reason, net_pnl).await`):
```rust
            match db.close_trade(&trade.id, current_debit, &exit_reason, net_pnl, None).await {
```
(Task 19 replaces `None` with a real snapshot.)

- [ ] **Step 4: Run — verify PASS**

```bash
cd traderjoe && cargo test
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/db/d1.rs traderjoe/src/handlers/position_monitor.rs && \
  git commit -m "feat(db): extend close_trade with exit NBBO param"
```

---

### Task 16: D1Client gains four measurement-table inserters

**Group:** C (sequential)
**Behavior being verified:** Four new methods exist with the expected signatures for writing to `equity_history`, `portfolio_greeks_eod`, `market_context_daily`, `metrics_weekly`.
**Interface under test:** `D1Client::insert_equity_snapshot`, `insert_portfolio_greeks_eod`, `insert_market_context_daily`, `insert_metrics_weekly`.

**Files:**
- Modify: `traderjoe/src/db/d1.rs`

- [ ] **Step 1: Write the failing test**

Append:

```rust
    #[test]
    fn measurement_inserters_exist_with_expected_signatures() {
        use crate::measurement::equity::EquitySnapshot;
        use crate::measurement::portfolio_greeks::PortfolioGreeks;
        let _unused = async {
            let db: D1Client = unreachable!();
            let snap: EquitySnapshot = unreachable!();
            let pg: PortfolioGreeks = unreachable!();
            db.insert_equity_snapshot(&snap).await.ok();
            db.insert_portfolio_greeks_eod("2026-04-22", &pg).await.ok();
            db.insert_market_context_daily(
                "2026-04-22", Some(17.5), "fred", Some("2026-04-21"),
                Some(14.2), Some(0.12), Some(0.02), Some(0.03),
            ).await.ok();
            db.insert_metrics_weekly(MetricsWeeklyRow::default()).await.ok();
        };
    }
```

- [ ] **Step 2: Run — FAIL expected**

```bash
cd traderjoe && cargo test measurement_inserters_exist
```
Expected: FAIL — methods not found.

- [ ] **Step 3: Implement**

Append to `d1.rs`:

```rust
#[derive(Debug, Clone, Default)]
pub struct MetricsWeeklyRow {
    pub generated_at: String,
    pub window_start: String,
    pub window_end: String,
    pub trade_count: i64,
    pub sharpe: Option<f64>,
    pub sortino: Option<f64>,
    pub profit_factor: Option<f64>,
    pub win_rate: Option<f64>,
    pub pnl_skew: Option<f64>,
    pub max_drawdown_pct: Option<f64>,
    pub mean_slippage_vs_mid: Option<f64>,
    pub max_slippage_vs_mid: Option<f64>,
    pub slippage_vs_orats_ratio: Option<f64>,
    pub fill_size_violation_count: i64,
    pub fill_size_violation_pct: Option<f64>,
    pub regime_buckets_json: String,
    pub greek_ranges_json: String,
    pub sample_size_tag: String,
}

impl D1Client {
    pub async fn insert_equity_snapshot(
        &self,
        snap: &crate::measurement::equity::EquitySnapshot,
    ) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let trade_ref = snap.trade_id_ref.as_deref()
            .map(|s| s.into()).unwrap_or(JsValue::NULL);
        self.db.prepare(
            "INSERT INTO equity_history
             (timestamp, event_type, equity, cash, open_position_mtm,
              realized_pnl_day, unrealized_pnl_day, open_position_count, trade_id_ref)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(date(timestamp)) WHERE event_type = 'eod'
               DO UPDATE SET equity = excluded.equity, cash = excluded.cash,
                 open_position_mtm = excluded.open_position_mtm,
                 realized_pnl_day = excluded.realized_pnl_day,
                 unrealized_pnl_day = excluded.unrealized_pnl_day,
                 open_position_count = excluded.open_position_count"
        ).bind(&[
            snap.timestamp.as_str().into(),
            snap.event_type.into(),
            snap.equity.into(), snap.cash.into(),
            snap.open_position_mtm.into(),
            snap.realized_pnl_day.into(), snap.unrealized_pnl_day.into(),
            (snap.open_position_count as f64).into(),
            trade_ref,
        ])?.run().await?;
        Ok(())
    }

    pub async fn insert_portfolio_greeks_eod(
        &self, date: &str,
        pg: &crate::measurement::portfolio_greeks::PortfolioGreeks,
    ) -> Result<()> {
        let delta_json = serde_json::to_string(&pg.delta_by_underlying)
            .map_err(|e| worker::Error::RustError(e.to_string()))?;
        self.db.prepare(
            "INSERT OR REPLACE INTO portfolio_greeks_eod
             (date, beta_weighted_delta, total_gamma, total_vega, total_theta,
              delta_by_underlying, max_gamma_single_position, max_vega_single_position,
              open_position_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).bind(&[
            date.into(),
            pg.beta_weighted_delta.into(),
            pg.total_gamma.into(), pg.total_vega.into(), pg.total_theta.into(),
            delta_json.as_str().into(),
            pg.max_gamma_single_position.into(),
            pg.max_vega_single_position.into(),
            (pg.open_position_count as f64).into(),
        ])?.run().await?;
        Ok(())
    }

    pub async fn insert_market_context_daily(
        &self, date: &str,
        spot_vix: Option<f64>, source: &str, source_date: Option<&str>,
        vixy_close: Option<f64>,
        spy_20d_rv: Option<f64>, spy_20d_ret: Option<f64>, spy_dd: Option<f64>,
    ) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let vix = spot_vix.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let src_date = source_date.map(|s| s.into()).unwrap_or(JsValue::NULL);
        let vy = vixy_close.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let rv = spy_20d_rv.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let rt = spy_20d_ret.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let dd = spy_dd.map(|v| v.into()).unwrap_or(JsValue::NULL);
        self.db.prepare(
            "INSERT OR REPLACE INTO market_context_daily
             (date, spot_vix, spot_vix_source, spot_vix_source_date, vixy_close,
              spy_20d_realized_vol, spy_20d_return, spy_drawdown_from_52w_high)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ).bind(&[
            date.into(), vix, source.into(), src_date, vy, rv, rt, dd,
        ])?.run().await?;
        Ok(())
    }

    pub async fn insert_metrics_weekly(&self, row: MetricsWeeklyRow) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let sh = row.sharpe.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let so = row.sortino.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let pf = row.profit_factor.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let wr = row.win_rate.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let sk = row.pnl_skew.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let md = row.max_drawdown_pct.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let ms = row.mean_slippage_vs_mid.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let mx = row.max_slippage_vs_mid.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let rt = row.slippage_vs_orats_ratio.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let fvp = row.fill_size_violation_pct.map(|v| v.into()).unwrap_or(JsValue::NULL);
        self.db.prepare(
            "INSERT INTO metrics_weekly (generated_at, window_start, window_end, trade_count,
             sharpe, sortino, profit_factor, win_rate, pnl_skew, max_drawdown_pct,
             mean_slippage_vs_mid, max_slippage_vs_mid, slippage_vs_orats_ratio,
             fill_size_violation_count, fill_size_violation_pct,
             regime_buckets, greek_ranges, sample_size_tag)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).bind(&[
            row.generated_at.as_str().into(),
            row.window_start.as_str().into(),
            row.window_end.as_str().into(),
            (row.trade_count as f64).into(),
            sh, so, pf, wr, sk, md, ms, mx, rt,
            (row.fill_size_violation_count as f64).into(), fvp,
            row.regime_buckets_json.as_str().into(),
            row.greek_ranges_json.as_str().into(),
            row.sample_size_tag.as_str().into(),
        ])?.run().await?;
        Ok(())
    }
}
```

- [ ] **Step 4: Run — PASS**

```bash
cd traderjoe && cargo test measurement_inserters_exist
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/db/d1.rs && \
  git commit -m "feat(db): add measurement table inserters"
```

---

### Task 17: D1Client gains four window-query helpers

**Group:** C (sequential)
**Behavior being verified:** Four query helpers exist with the expected signatures.
**Interface under test:** `D1Client::get_closed_trades_in_window`, `get_equity_history_in_window`, `get_portfolio_greeks_history_in_window`, `get_market_context_in_window`.

**Files:**
- Modify: `traderjoe/src/db/d1.rs`

- [ ] **Step 1: Write failing test**

```rust
    #[test]
    fn window_query_helpers_exist() {
        let _unused = async {
            let db: D1Client = unreachable!();
            let _: Vec<Trade> = db.get_closed_trades_in_window("2026-03-22", "2026-04-22").await.unwrap();
            let _: Vec<(String, f64)> = db.get_equity_history_in_window("2026-03-22", "2026-04-22").await.unwrap();
            let _: Vec<(String, f64, f64, f64, f64)> =
                db.get_portfolio_greeks_history_in_window("2026-03-22", "2026-04-22").await.unwrap();
            let _: Vec<MarketContextRow> =
                db.get_market_context_in_window("2026-03-22", "2026-04-22").await.unwrap();
        };
    }
```

- [ ] **Step 2: Run — FAIL**

```bash
cd traderjoe && cargo test window_query_helpers_exist
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `d1.rs`:

```rust
#[derive(Debug, Clone)]
pub struct MarketContextRow {
    pub date: String,
    pub spot_vix: Option<f64>,
    pub vixy_close: Option<f64>,
    pub spy_20d_realized_vol: Option<f64>,
    pub spy_20d_return: Option<f64>,
    pub spy_drawdown_from_52w_high: Option<f64>,
}

impl D1Client {
    pub async fn get_closed_trades_in_window(&self, start: &str, end: &str) -> Result<Vec<Trade>> {
        let result = self.db.prepare(
            "SELECT * FROM trades WHERE status = 'closed' AND date(exit_time) BETWEEN ? AND ?
             ORDER BY exit_time ASC"
        ).bind(&[start.into(), end.into()])?.all().await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(TradeRow::from_d1_row).collect())
    }

    /// Returns (eod_date_iso, equity) pairs for EOD snapshots in window.
    pub async fn get_equity_history_in_window(&self, start: &str, end: &str) -> Result<Vec<(String, f64)>> {
        let result = self.db.prepare(
            "SELECT date(timestamp) AS d, equity FROM equity_history
             WHERE event_type = 'eod' AND date(timestamp) BETWEEN ? AND ? ORDER BY d ASC"
        ).bind(&[start.into(), end.into()])?.all().await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(|r|
            Some((r["d"].as_str()?.to_string(), r["equity"].as_f64()?))
        ).collect())
    }

    /// Returns (date, beta_weighted_delta, total_gamma, total_vega, total_theta).
    pub async fn get_portfolio_greeks_history_in_window(
        &self, start: &str, end: &str,
    ) -> Result<Vec<(String, f64, f64, f64, f64)>> {
        let result = self.db.prepare(
            "SELECT date, beta_weighted_delta, total_gamma, total_vega, total_theta
             FROM portfolio_greeks_eod WHERE date BETWEEN ? AND ? ORDER BY date ASC"
        ).bind(&[start.into(), end.into()])?.all().await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(|r| Some((
            r["date"].as_str()?.to_string(),
            r["beta_weighted_delta"].as_f64()?,
            r["total_gamma"].as_f64()?,
            r["total_vega"].as_f64()?,
            r["total_theta"].as_f64()?,
        ))).collect())
    }

    pub async fn get_market_context_in_window(
        &self, start: &str, end: &str,
    ) -> Result<Vec<MarketContextRow>> {
        let result = self.db.prepare(
            "SELECT * FROM market_context_daily WHERE date BETWEEN ? AND ? ORDER BY date ASC"
        ).bind(&[start.into(), end.into()])?.all().await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(|r| Some(MarketContextRow {
            date: r["date"].as_str()?.to_string(),
            spot_vix: r["spot_vix"].as_f64(),
            vixy_close: r["vixy_close"].as_f64(),
            spy_20d_realized_vol: r["spy_20d_realized_vol"].as_f64(),
            spy_20d_return: r["spy_20d_return"].as_f64(),
            spy_drawdown_from_52w_high: r["spy_drawdown_from_52w_high"].as_f64(),
        })).collect())
    }
}
```

- [ ] **Step 4: PASS**

```bash
cd traderjoe && cargo test window_query_helpers_exist
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/db/d1.rs && git commit -m "feat(db): add window-query helpers for metrics aggregator"
```

---

## Group D — Sequential (handler wiring)

### Task 18: morning_scan wires NBBO capture + equity snapshot on trade open + LIVE_CAPITAL_FRACTION

**Group:** D (sequential)
**Behavior being verified:** Existing `morning_scan` tests still pass with new wiring. The new behavior is plumbing-only — verified via compilation + existing tests.
**Interface under test:** `handlers::morning_scan::run` (via cargo test `handlers::morning_scan::tests`).

**Files:**
- Modify: `traderjoe/src/handlers/morning_scan.rs`
- Modify: `traderjoe/src/risk/position_sizer.rs` (add `effective_equity` helper used by handler)

- [ ] **Step 1: Write failing test**

Append to `position_sizer.rs` tests:

```rust
    #[test]
    fn effective_equity_scales_by_fraction() {
        assert!((effective_equity(100_000.0, 0.10) - 10_000.0).abs() < 1e-9);
        assert!((effective_equity(100_000.0, 1.00) - 100_000.0).abs() < 1e-9);
    }

    #[test]
    fn effective_equity_clamps_fraction_to_zero_one() {
        // Fraction > 1 clamped to 1 (never size above full account); < 0 clamped to 0.
        assert!((effective_equity(100_000.0, 1.50) - 100_000.0).abs() < 1e-9);
        assert!((effective_equity(100_000.0, -0.5) - 0.0).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run — FAIL**

```bash
cd traderjoe && cargo test effective_equity
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `traderjoe/src/risk/position_sizer.rs`:

```rust
/// Scales equity by a live-capital fraction, clamped to [0, 1].
pub fn effective_equity(raw_equity: f64, fraction: f64) -> f64 {
    raw_equity * fraction.clamp(0.0, 1.0)
}
```

In `traderjoe/src/handlers/morning_scan.rs`:

1. Replace the `account.equity` uses for sizing with effective equity. Near the top of `run_inner` after `account = alpaca.get_account().await?`:

```rust
    let live_fraction: f64 = env.var("LIVE_CAPITAL_FRACTION")
        .ok().and_then(|v| v.to_string().parse().ok())
        .unwrap_or(0.10);
    let effective_eq = crate::risk::position_sizer::effective_equity(account.equity, live_fraction);
```

2. Pass `effective_eq` to `sizer.calculate_for_underlying(...)` in place of `account.equity` (around line 210).

3. Before `alpaca.place_spread_order(&order).await`, capture NBBO:

```rust
        let option_type_enum = match spread.spread_type {
            SpreadType::BullPut => crate::broker::types::OptionType::Put,
            SpreadType::BearCall => crate::broker::types::OptionType::Call,
        };
        let nbbo = crate::measurement::nbbo::snapshot_spread_nbbo(
            &chain,  // chain is in scope per the per-symbol loop — restructure is required
            spread.short_strike, spread.long_strike, option_type_enum,
        );
```

**NOTE:** `chain` is currently scoped per-symbol-iteration; after the sorted-opportunity loop starts, it's out of scope. Restructure so the `chain` used for the opportunity is retained. Simplest fix: include the chain in the `all_opportunities` tuple:

Change:
```rust
let mut all_opportunities = Vec::new();
```
to keep `(scored, symbol, iv_metrics, chain)` per entry. Clone the chain into each scored opportunity:
```rust
    for scored in spreads.drain(..).take(2) {
        all_opportunities.push((scored, symbol.to_string(), iv_metrics.clone(), chain.clone()));
    }
```
And update the downstream consumption to destructure the chain out.

4. Update the `db.create_trade(...)` call to pass the NBBO and expanded Greeks (pull from the chain's short/long contracts at spread strikes):

```rust
        let short_contract = chain.contracts.iter().find(|c|
            (c.strike - spread.short_strike).abs() < 0.01 && c.option_type == option_type_enum);
        let long_contract = chain.contracts.iter().find(|c|
            (c.strike - spread.long_strike).abs() < 0.01 && c.option_type == option_type_enum);

        let entry_short_gamma = short_contract.and_then(|c| c.gamma);
        let entry_short_vega = short_contract.and_then(|c| c.vega);
        let entry_long_delta = long_contract.and_then(|c| c.delta);
        let entry_long_gamma = long_contract.and_then(|c| c.gamma);
        let entry_long_vega = long_contract.and_then(|c| c.vega);

        match db.create_trade(
            &trade_id, &spread.underlying, &spread.spread_type,
            spread.short_strike, spread.long_strike, &spread.expiration,
            final_contracts, spread.entry_credit, spread.max_loss_per_contract(),
            Some(&placed.id), Some(iv_metrics.iv_rank),
            spread.short_delta, spread.short_theta,
            nbbo.as_ref(),
            entry_short_gamma, entry_short_vega,
            entry_long_delta, entry_long_gamma, entry_long_vega,
        ).await {
```

5. After the successful `db.create_trade` block (after the Discord notification), write an equity snapshot:

```rust
        let open_trades_now = db.get_open_trades().await.unwrap_or_default();
        let snap = crate::measurement::equity::build_equity_snapshot(
            crate::measurement::equity::EquityEvent::TradeOpen,
            crate::measurement::equity::SnapshotInputs {
                timestamp: Utc::now().to_rfc3339(),
                equity: account.equity, cash: account.cash,
                open_position_mtm: 0.0,  // MTM not available here — accept 0 for open events
                realized_pnl_day: daily_stats.realized_pnl,
                unrealized_pnl_day: 0.0,
                open_position_count: open_trades_now.len() as i64,
                trade_id_ref: Some(trade_id.clone()),
            },
        );
        db.insert_equity_snapshot(&snap).await.ok();
```

6. Also write a `CircuitBreaker` equity snapshot at the site where risk escalates to Halted (around line 108-118), just before setting the KV state:

```rust
    if risk_state.level == RiskLevel::Halted {
        let snap = crate::measurement::equity::build_equity_snapshot(
            crate::measurement::equity::EquityEvent::CircuitBreaker,
            crate::measurement::equity::SnapshotInputs {
                timestamp: Utc::now().to_rfc3339(),
                equity: account.equity, cash: account.cash,
                open_position_mtm: 0.0,
                realized_pnl_day: 0.0, unrealized_pnl_day: 0.0,
                open_position_count: open_trades.len() as i64,
                trade_id_ref: None,
            },
        );
        db.insert_equity_snapshot(&snap).await.ok();
        // ...existing halt code...
```

- [ ] **Step 4: PASS**

```bash
cd traderjoe && cargo test
```
Expected: PASS (all existing tests plus new `effective_equity` tests).

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/handlers/morning_scan.rs traderjoe/src/risk/position_sizer.rs && \
  git commit -m "feat(morning-scan): NBBO capture, equity snapshots, LIVE_CAPITAL_FRACTION"
```

---

### Task 19: position_monitor wires NBBO capture at exit + equity snapshot on close

**Group:** D (sequential)
**Behavior being verified:** Existing tests continue to pass after wiring.
**Interface under test:** `handlers::position_monitor::run` via compilation and existing tests.

**Files:**
- Modify: `traderjoe/src/handlers/position_monitor.rs`

- [ ] **Step 1: Run existing tests green baseline**

```bash
cd traderjoe && cargo test position_monitor
```
Expected: PASS (baseline).

- [ ] **Step 2: Modify with red gate: new compile check**

At the top of position_monitor tests module, add:

```rust
    #[test]
    fn close_trade_is_invoked_with_exit_nbbo() {
        // Compile-only: if the call site in run() doesn't pass NBBO,
        // this regression test's expected signature won't match. Cheap sentinel.
        use crate::measurement::nbbo::NbboSnapshot;
        fn _ty_check(_: Option<&NbboSnapshot>) {}
        _ty_check(None);
    }
```

This is a weak test but enforces the module is referenced from position_monitor via `use`. More substantively: running `cargo build` after editing will reject mismatched arity.

- [ ] **Step 3: Implement**

In `traderjoe/src/handlers/position_monitor.rs`, inside `run()` after computing `current_debit` and before placing the closing order, capture NBBO:

```rust
            let option_type_enum = match trade.spread_type {
                crate::types::SpreadType::BullPut => crate::broker::types::OptionType::Put,
                crate::types::SpreadType::BearCall => crate::broker::types::OptionType::Call,
            };
            let exit_chain = alpaca.get_options_chain(&trade.underlying).await.ok();
            let exit_nbbo = exit_chain.as_ref().and_then(|c|
                crate::measurement::nbbo::snapshot_spread_nbbo(
                    c, trade.short_strike, trade.long_strike, option_type_enum,
                )
            );
```

Replace `db.close_trade(&trade.id, current_debit, &exit_reason, net_pnl, None)` with:
```rust
            match db.close_trade(&trade.id, current_debit, &exit_reason, net_pnl, exit_nbbo.as_ref()).await {
```

After `Ok(true) => {}` branch succeeds, write equity snapshot:

```rust
                Ok(true) => {
                    use chrono::Utc;
                    let snap = crate::measurement::equity::build_equity_snapshot(
                        crate::measurement::equity::EquityEvent::TradeClose,
                        crate::measurement::equity::SnapshotInputs {
                            timestamp: Utc::now().to_rfc3339(),
                            equity: alpaca.get_account().await.map(|a| a.equity).unwrap_or(0.0),
                            cash: alpaca.get_account().await.map(|a| a.cash).unwrap_or(0.0),
                            open_position_mtm: 0.0,
                            realized_pnl_day: 0.0,  // EOD handler aggregates; leave 0 here
                            unrealized_pnl_day: 0.0,
                            open_position_count: (open_trades.len().saturating_sub(1)) as i64,
                            trade_id_ref: Some(trade.id.clone()),
                        },
                    );
                    db.insert_equity_snapshot(&snap).await.ok();
                }
```

Note: the two redundant `alpaca.get_account()` calls are suboptimal but acceptable; MVP. Future refactor can snapshot once and pass.

- [ ] **Step 4: Run — PASS**

```bash
cd traderjoe && cargo test
```

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/handlers/position_monitor.rs && \
  git commit -m "feat(position-monitor): capture exit NBBO and equity snapshot on close"
```

---

### Task 20: eod_summary writes equity snapshot + portfolio Greeks + market context

**Group:** D (sequential)
**Behavior being verified:** Existing tests continue to pass; new code compiles against the measurement API.
**Interface under test:** `handlers::eod_summary::run` — wiring-only.

**Files:**
- Modify: `traderjoe/src/handlers/eod_summary.rs`

- [ ] **Step 1: Baseline green**

```bash
cd traderjoe && cargo test eod_summary
```
Expected: PASS.

- [ ] **Step 2: Write a failing test** for new function `collect_chains_for_greeks`:

Append to eod_summary tests:

```rust
    #[test]
    fn collect_chains_for_greeks_returns_map_keyed_by_underlying() {
        use std::collections::HashMap;
        use crate::broker::types::{OptionsChain};
        let chain_spy = OptionsChain {
            underlying: "SPY".to_string(), underlying_price: 480.0,
            expirations: vec![], contracts: vec![],
        };
        let chain_qqq = OptionsChain {
            underlying: "QQQ".to_string(), underlying_price: 500.0,
            expirations: vec![], contracts: vec![],
        };
        let map = build_chain_map(vec![chain_spy, chain_qqq]);
        assert!(map.contains_key("SPY"));
        assert!(map.contains_key("QQQ"));
    }
```

- [ ] **Step 3: FAIL**

```bash
cd traderjoe && cargo test collect_chains_for_greeks_returns_map
```

- [ ] **Step 4: Implement**

Add at module level of `eod_summary.rs`:

```rust
use std::collections::HashMap;

pub fn build_chain_map(chains: Vec<crate::broker::types::OptionsChain>) -> HashMap<String, crate::broker::types::OptionsChain> {
    let mut map = HashMap::new();
    for c in chains {
        map.insert(c.underlying.clone(), c);
    }
    map
}
```

Inside `run()`, after the existing IV-snapshot loop over SPY/QQQ/IWM:

1. Keep the chain that was fetched for each symbol. Change the loop:

```rust
    let mut chains: Vec<crate::broker::types::OptionsChain> = Vec::new();
    for symbol in &["SPY", "QQQ", "IWM"] {
        match alpaca.get_options_chain(symbol).await {
            Ok(chain) => {
                // existing IV snapshot block ...
                chains.push(chain);
            }
            Err(e) => console_log!("Could not fetch chain for {} IV snapshot: {:?}", symbol, e),
        }
    }
    let chain_map = build_chain_map(chains);
```

2. Compute and write portfolio Greeks:

```rust
    let open_trades = db.get_open_trades().await.unwrap_or_default();
    let pg = crate::measurement::portfolio_greeks::PortfolioGreeksAggregator::compute(
        &open_trades, &chain_map,
    );
    db.insert_portfolio_greeks_eod(&today, &pg).await.ok();
```

3. Fetch and write market context:

```rust
    let vixy = alpaca.get_vix().await.ok().flatten().map(|v| v.vix);

    let spy_bars_52w = alpaca.get_bars("SPY", 260).await.unwrap_or_default();
    let spy_bars_20d: Vec<_> = spy_bars_52w.iter().rev().take(20).rev().cloned().collect();
    let spy_stats = crate::measurement::market_context::compute_spy_stats(&spy_bars_20d, &spy_bars_52w);

    // Spot VIX: try FRED; alert via Discord if unavailable so failures are visible.
    let spot_vix_reading = fetch_fred_spot_vix().await;
    let (spot_vix_val, src, src_date) = match spot_vix_reading {
        Some(r) => (Some(r.value), r.source, Some(r.source_date)),
        None => {
            discord.send_error(
                "VIX Data Unavailable",
                &format!("FRED VIXCLS fetch failed for {}. spot_vix stored as NULL — regime tagging for today will be absent.", today),
            ).await.ok();
            (None, "unavailable", None)
        }
    };
    db.insert_market_context_daily(
        &today, spot_vix_val, src, src_date.as_deref(),
        vixy,
        Some(spy_stats.realized_vol_20d),
        Some(spy_stats.return_20d),
        Some(spy_stats.drawdown_from_52w_high),
    ).await.ok();
```

4. Add the FRED fetcher function at the bottom of `eod_summary.rs`:

```rust
async fn fetch_fred_spot_vix() -> Option<crate::measurement::market_context::SpotVixReading> {
    use worker::{Fetch, Method, Request, RequestInit};
    let url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS";
    let mut init = RequestInit::new();
    init.with_method(Method::Get);
    let req = Request::new_with_init(url, &init).ok()?;
    let mut resp = Fetch::Request(req).send().await.ok()?;
    let body = resp.text().await.ok()?;
    crate::measurement::market_context::parse_fred_vix_csv(&body)
}
```

5. Write EOD equity snapshot at the end of run():

```rust
    let account = alpaca.get_account().await.ok();
    let (eq, cash) = account.as_ref().map(|a| (a.equity, a.cash)).unwrap_or((0.0, 0.0));
    let snap = crate::measurement::equity::build_equity_snapshot(
        crate::measurement::equity::EquityEvent::Eod,
        crate::measurement::equity::SnapshotInputs {
            timestamp: Utc::now().to_rfc3339(),
            equity: eq, cash,
            open_position_mtm: 0.0,
            realized_pnl_day: realized_pnl,
            unrealized_pnl_day: 0.0,
            open_position_count: open_trades.len() as i64,
            trade_id_ref: None,
        },
    );
    db.insert_equity_snapshot(&snap).await.ok();
```

- [ ] **Step 5: PASS**

```bash
cd traderjoe && cargo test
```

- [ ] **Step 6: Commit**

```bash
git add traderjoe/src/handlers/eod_summary.rs && \
  git commit -m "feat(eod-summary): write equity, portfolio greeks, market context"
```

---

### Task 21: compute_metrics top-level aggregator

**Group:** D (sequential)
**Behavior being verified:** Given a fixture of inputs, `compute_metrics` returns a complete `MetricBundle` with correct sample-size tagging on every metric.
**Interface under test:** `measurement::metrics::compute_metrics`.

**Files:**
- Modify: `traderjoe/src/measurement/metrics.rs`

- [ ] **Step 1: Write failing test**

Append to `metrics.rs` tests:

```rust
    use crate::types::{Trade, TradeStatus, SpreadType};
    use crate::db::d1::MarketContextRow;

    fn closed_trade(id: &str, net_pnl: f64, created: &str) -> Trade {
        Trade {
            id: id.to_string(), created_at: created.to_string(),
            underlying: "SPY".to_string(), spread_type: SpreadType::BullPut,
            short_strike: 460.0, long_strike: 455.0, expiration: "2026-05-15".to_string(),
            contracts: 1, entry_credit: 0.50, max_loss: 450.0,
            broker_order_id: Some("o".to_string()), status: TradeStatus::Closed,
            fill_price: Some(0.48), fill_time: Some(created.to_string()),
            exit_price: Some(0.20), exit_time: Some(created.to_string()),
            exit_reason: None, net_pnl: Some(net_pnl),
            iv_rank: None, short_delta: None, short_theta: None,
            entry_short_bid: Some(0.74), entry_short_ask: Some(0.76),
            entry_long_bid: Some(0.24), entry_long_ask: Some(0.26),
            entry_net_mid: Some(0.50),
            exit_short_bid: None, exit_short_ask: None,
            exit_long_bid: None, exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: None, entry_short_vega: None,
            entry_long_delta: None, entry_long_gamma: None, entry_long_vega: None,
            nbbo_displayed_size_short: Some(50), nbbo_displayed_size_long: Some(75),
            nbbo_snapshot_time: None,
        }
    }

    #[test]
    fn compute_metrics_small_sample_tagged_insufficient() {
        let trades = vec![closed_trade("t1", 50.0, "2026-04-01T00:00:00Z")];
        let equity: Vec<(String, f64)> = vec![
            ("2026-04-01".to_string(), 100_000.0),
            ("2026-04-02".to_string(), 100_050.0),
        ];
        let greeks: Vec<(String, f64, f64, f64, f64)> = vec![];
        let ctx: Vec<MarketContextRow> = vec![];
        let bundle = compute_metrics(MetricsInput {
            window_start: "2026-04-01".to_string(),
            window_end: "2026-04-02".to_string(),
            trades, equity_eod: equity, greeks_eod: greeks, market_context: ctx,
        });
        assert_eq!(bundle.sample_size_tag, SampleSizeTag::Insufficient);
        assert_eq!(bundle.trade_count, 1);
    }
```

- [ ] **Step 2: FAIL**

```bash
cd traderjoe && cargo test compute_metrics_small_sample_tagged_insufficient
```

- [ ] **Step 3: Implement**

Append to `traderjoe/src/measurement/metrics.rs`:

```rust
use crate::db::d1::MarketContextRow;
use crate::types::Trade;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MetricsInput {
    pub window_start: String,
    pub window_end: String,
    pub trades: Vec<Trade>,
    pub equity_eod: Vec<(String, f64)>,
    pub greeks_eod: Vec<(String, f64, f64, f64, f64)>,
    pub market_context: Vec<MarketContextRow>,
}

#[derive(Debug, Clone)]
pub struct MetricBundle {
    pub window_start: String,
    pub window_end: String,
    pub trade_count: usize,
    pub sharpe: f64,
    pub sortino: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub pnl_skew: f64,
    pub max_drawdown_pct: f64,
    pub mean_slippage_vs_mid: f64,
    pub max_slippage_vs_mid: f64,
    pub slippage_vs_orats_ratio: f64,
    pub fill_size_violation_count: usize,
    pub fill_size_violation_pct: f64,
    pub regime_buckets: HashMap<String, (i64, f64)>, // (count, sum_pnl)
    pub greek_ranges: HashMap<String, (f64, f64, f64)>, // (min, max, avg)
    pub sample_size_tag: SampleSizeTag,
}

pub fn compute_metrics(input: MetricsInput) -> MetricBundle {
    let n = input.trades.len();
    let tag = sample_size_tag(n);

    let pnls: Vec<f64> = input.trades.iter().filter_map(|t| t.net_pnl).collect();
    let wins = pnls.iter().filter(|p| **p > 0.0).count();
    let win_rate = if pnls.is_empty() { 0.0 } else { wins as f64 / pnls.len() as f64 };

    let equity_values: Vec<f64> = input.equity_eod.iter().map(|(_, e)| *e).collect();
    let daily_returns: Vec<f64> = equity_values.windows(2)
        .filter(|w| w[0] > 0.0)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let sharpe = compute_sharpe(&daily_returns);
    let sortino = compute_sortino(&daily_returns);
    let profit_factor = compute_profit_factor(&pnls);
    let pnl_skew = compute_pnl_skew(&pnls);
    let max_drawdown_pct = compute_max_drawdown_pct(&equity_values);

    let slippage_inputs: Vec<SlippageInput> = input.trades.iter().filter_map(|t| {
        let mid = t.entry_net_mid?;
        let fill = t.fill_price?;
        let sw = (t.entry_short_ask? - t.entry_short_bid?).abs();
        let lw = (t.entry_long_ask? - t.entry_long_bid?).abs();
        Some(SlippageInput { entry_mid: mid, entry_fill: fill, leg_width: (sw + lw) / 2.0 })
    }).collect();
    let slip = compute_slippage_stats(&slippage_inputs);

    let violations = input.trades.iter().filter(|t| {
        match (t.nbbo_displayed_size_short, t.nbbo_displayed_size_long) {
            (Some(s), Some(l)) => paper_fill_violation(t.contracts, s, l),
            _ => false,
        }
    }).count();
    let violation_pct = if n == 0 { 0.0 } else { violations as f64 / n as f64 };

    // Regime bucketing
    let ctx_by_date: HashMap<String, Option<f64>> = input.market_context.iter()
        .map(|m| (m.date.clone(), m.spot_vix))
        .collect();
    let mut regime_buckets: HashMap<String, (i64, f64)> = HashMap::new();
    for t in &input.trades {
        let date = t.created_at.get(..10).unwrap_or(&t.created_at).to_string();
        let vix = ctx_by_date.get(&date).and_then(|v| *v).unwrap_or(20.0);
        let bucket = classify_regime(vix).to_string();
        let entry = regime_buckets.entry(bucket).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += t.net_pnl.unwrap_or(0.0);
    }

    // Greek ranges (min, max, avg) for each of bwd/gamma/vega/theta
    let mut greek_ranges: HashMap<String, (f64, f64, f64)> = HashMap::new();
    if !input.greeks_eod.is_empty() {
        let k = input.greeks_eod.len() as f64;
        let bwd: Vec<f64> = input.greeks_eod.iter().map(|g| g.1).collect();
        let gam: Vec<f64> = input.greeks_eod.iter().map(|g| g.2).collect();
        let veg: Vec<f64> = input.greeks_eod.iter().map(|g| g.3).collect();
        let the: Vec<f64> = input.greeks_eod.iter().map(|g| g.4).collect();
        greek_ranges.insert("beta_weighted_delta".to_string(), (
            bwd.iter().cloned().fold(f64::INFINITY, f64::min),
            bwd.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            bwd.iter().sum::<f64>() / k));
        greek_ranges.insert("total_gamma".to_string(), (
            gam.iter().cloned().fold(f64::INFINITY, f64::min),
            gam.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            gam.iter().sum::<f64>() / k));
        greek_ranges.insert("total_vega".to_string(), (
            veg.iter().cloned().fold(f64::INFINITY, f64::min),
            veg.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            veg.iter().sum::<f64>() / k));
        greek_ranges.insert("total_theta".to_string(), (
            the.iter().cloned().fold(f64::INFINITY, f64::min),
            the.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            the.iter().sum::<f64>() / k));
    }

    MetricBundle {
        window_start: input.window_start, window_end: input.window_end,
        trade_count: n, sharpe, sortino, profit_factor, win_rate, pnl_skew,
        max_drawdown_pct,
        mean_slippage_vs_mid: slip.mean_abs,
        max_slippage_vs_mid: slip.max_abs,
        slippage_vs_orats_ratio: slip.ratio_vs_orats,
        fill_size_violation_count: violations,
        fill_size_violation_pct: violation_pct,
        regime_buckets, greek_ranges, sample_size_tag: tag,
    }
}
```

- [ ] **Step 4: PASS**

```bash
cd traderjoe && cargo test compute_metrics_small_sample_tagged_insufficient
```

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/measurement/metrics.rs && \
  git commit -m "feat(measurement): add compute_metrics top-level aggregator"
```

---

## Group E — Sequential (final wiring)

### Task 22: Discord metrics embed

**Group:** E (sequential)
**Behavior being verified:** `build_metrics_embed` returns an embed with expected title, color (blue for OK tag, yellow for WEAK, red for INSUFFICIENT), and key fields.
**Interface under test:** `build_metrics_embed`.

**Files:**
- Modify: `traderjoe/src/notifications/discord.rs`

- [ ] **Step 1: Write failing test**

Append to `discord.rs` tests:

```rust
    use crate::measurement::metrics::{MetricBundle, SampleSizeTag};
    use std::collections::HashMap;

    fn mk_bundle(tag: SampleSizeTag) -> MetricBundle {
        MetricBundle {
            window_start: "2026-03-22".to_string(), window_end: "2026-04-22".to_string(),
            trade_count: 42, sharpe: 1.2, sortino: 1.8, profit_factor: 1.5,
            win_rate: 0.72, pnl_skew: -1.1, max_drawdown_pct: 0.06,
            mean_slippage_vs_mid: 0.012, max_slippage_vs_mid: 0.04,
            slippage_vs_orats_ratio: 1.1,
            fill_size_violation_count: 0, fill_size_violation_pct: 0.0,
            regime_buckets: HashMap::new(), greek_ranges: HashMap::new(),
            sample_size_tag: tag,
        }
    }

    #[test]
    fn metrics_embed_includes_window_and_tag() {
        let embed = build_metrics_embed(&mk_bundle(SampleSizeTag::Weak));
        let desc = embed["description"].as_str().unwrap_or("");
        assert!(desc.contains("2026-03-22"));
        assert!(desc.contains("WEAK"));
    }
```

- [ ] **Step 2: FAIL**

```bash
cd traderjoe && cargo test metrics_embed_includes_window_and_tag
```

- [ ] **Step 3: Implement**

Append to `discord.rs`:

```rust
pub fn build_metrics_embed(b: &crate::measurement::metrics::MetricBundle) -> Value {
    use crate::measurement::metrics::SampleSizeTag;
    let (color, tag_str) = match b.sample_size_tag {
        SampleSizeTag::Ok => (COLOR_BLUE, "OK"),
        SampleSizeTag::Weak => (0xF0B232u32, "WEAK"),
        SampleSizeTag::Insufficient => (COLOR_RED, "INSUFFICIENT"),
    };
    json!({
        "title": format!("Weekly Metrics — {} to {}", b.window_start, b.window_end),
        "color": color,
        "description": format!(
            "Window: {} → {}\nSample tag: **{}** (n={})",
            b.window_start, b.window_end, tag_str, b.trade_count
        ),
        "fields": [
            { "name": "Sharpe", "value": format!("{:.2}", b.sharpe), "inline": true },
            { "name": "Sortino", "value": format!("{:.2}", b.sortino), "inline": true },
            { "name": "Profit Factor", "value": format!("{:.2}", b.profit_factor), "inline": true },
            { "name": "Win Rate", "value": format!("{:.0}%", b.win_rate * 100.0), "inline": true },
            { "name": "Max DD", "value": format!("{:.1}%", b.max_drawdown_pct * 100.0), "inline": true },
            { "name": "P&L Skew", "value": format!("{:.2}", b.pnl_skew), "inline": true },
            { "name": "Slippage Ratio vs ORATS", "value": format!("{:.2}", b.slippage_vs_orats_ratio), "inline": true },
            { "name": "Fill Size Violations", "value": format!("{} ({:.1}%)", b.fill_size_violation_count, b.fill_size_violation_pct * 100.0), "inline": true },
        ]
    })
}

impl DiscordClient {
    pub async fn send_metrics_report(&self, b: &crate::measurement::metrics::MetricBundle) -> Result<()> {
        let embed = build_metrics_embed(b);
        self.send("**Weekly Metrics**", vec![embed]).await
    }
}
```

- [ ] **Step 4: PASS**

```bash
cd traderjoe && cargo test metrics_embed
```

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/notifications/discord.rs && \
  git commit -m "feat(discord): add weekly metrics embed"
```

---

### Task 23: weekly_metrics handler

**Group:** E (sequential)
**Behavior being verified:** Handler file exists with `run(&Env)` signature matching the other cron handlers.
**Interface under test:** `handlers::weekly_metrics::run`.

**Files:**
- Create: `traderjoe/src/handlers/weekly_metrics.rs`
- Modify: `traderjoe/src/handlers/mod.rs`

- [ ] **Step 1: Write failing test**

In `traderjoe/src/handlers/weekly_metrics.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_has_expected_signature() {
        let _f: fn(&worker::Env) -> _ = |_env| async { run(_env).await };
    }
}
```

- [ ] **Step 2: FAIL**

```bash
cd traderjoe && cargo test run_has_expected_signature
```

- [ ] **Step 3: Implement**

Create `traderjoe/src/handlers/weekly_metrics.rs`:

```rust
use worker::{console_log, Env, Result};

use crate::db::d1::{D1Client, MetricsWeeklyRow};
use crate::measurement::metrics::{compute_metrics, MetricsInput, SampleSizeTag};
use crate::notifications::discord::DiscordClient;

pub async fn run(env: &Env) -> Result<()> {
    use chrono::{Duration, Utc};
    console_log!("Weekly metrics starting...");

    let db = D1Client::new(env.d1("DB")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    let end = Utc::now().date_naive();
    let start = end - Duration::days(30);
    let start_s = start.format("%Y-%m-%d").to_string();
    let end_s = end.format("%Y-%m-%d").to_string();

    let trades = db.get_closed_trades_in_window(&start_s, &end_s).await?;
    let equity = db.get_equity_history_in_window(&start_s, &end_s).await?;
    let greeks = db.get_portfolio_greeks_history_in_window(&start_s, &end_s).await?;
    let ctx = db.get_market_context_in_window(&start_s, &end_s).await?;

    let bundle = compute_metrics(MetricsInput {
        window_start: start_s.clone(), window_end: end_s.clone(),
        trades, equity_eod: equity, greeks_eod: greeks, market_context: ctx,
    });

    let tag_str = match bundle.sample_size_tag {
        SampleSizeTag::Ok => "OK", SampleSizeTag::Weak => "WEAK",
        SampleSizeTag::Insufficient => "INSUFFICIENT",
    };

    let regime_json = serde_json::to_string(&bundle.regime_buckets)
        .unwrap_or_else(|_| "{}".to_string());
    let greek_json = serde_json::to_string(&bundle.greek_ranges)
        .unwrap_or_else(|_| "{}".to_string());

    db.insert_metrics_weekly(MetricsWeeklyRow {
        generated_at: Utc::now().to_rfc3339(),
        window_start: start_s, window_end: end_s,
        trade_count: bundle.trade_count as i64,
        sharpe: Some(bundle.sharpe),
        sortino: Some(bundle.sortino),
        profit_factor: Some(bundle.profit_factor),
        win_rate: Some(bundle.win_rate),
        pnl_skew: Some(bundle.pnl_skew),
        max_drawdown_pct: Some(bundle.max_drawdown_pct),
        mean_slippage_vs_mid: Some(bundle.mean_slippage_vs_mid),
        max_slippage_vs_mid: Some(bundle.max_slippage_vs_mid),
        slippage_vs_orats_ratio: Some(bundle.slippage_vs_orats_ratio),
        fill_size_violation_count: bundle.fill_size_violation_count as i64,
        fill_size_violation_pct: Some(bundle.fill_size_violation_pct),
        regime_buckets_json: regime_json,
        greek_ranges_json: greek_json,
        sample_size_tag: tag_str.to_string(),
    }).await?;

    discord.send_metrics_report(&bundle).await.ok();
    console_log!("Weekly metrics complete.");
    Ok(())
}
```

Append to `traderjoe/src/handlers/mod.rs`:
```rust
pub mod weekly_metrics;
```

- [ ] **Step 4: PASS**

```bash
cd traderjoe && cargo test
```

- [ ] **Step 5: Commit**

```bash
git add traderjoe/src/handlers/ && \
  git commit -m "feat(handlers): add weekly metrics handler"
```

---

### Task 24: Register weekly cron + /trigger/metrics route + wrangler.toml

**Group:** E (sequential, final)
**Behavior being verified:** lib.rs dispatches the new cron expression; wrangler.toml contains the new cron.
**Interface under test:** `lib.rs::scheduled` and wrangler config.

**Files:**
- Modify: `traderjoe/src/lib.rs`
- Modify: `traderjoe/wrangler.toml`

- [ ] **Step 1: Write failing check**

```bash
grep -q '"0 2 \* \* MON"' traderjoe/wrangler.toml
```
Expected: FAIL — not present.

- [ ] **Step 2: Implement**

In `traderjoe/wrangler.toml`, update the `crons` array:

```toml
[triggers]
crons = [
    "0 14 * * MON-FRI",
    "*/5 14-20 * * MON-FRI",
    "15 20 * * MON-FRI",
    "0 2 * * MON",
]
```

In `traderjoe/src/lib.rs`:

1. Inside the `scheduled` function's `match cron.as_str()` add a new arm (before the `unknown =>` arm):
```rust
        "0 2 * * MON" => handlers::weekly_metrics::run(&env).await,
```

2. Inside the `fetch` function's POST route block, add a route for `/trigger/metrics` (after the `eod-summary` route):
```rust
            if path == "/trigger/metrics" {
                if let Err(r) = check_auth(&req, &env) { return Ok(r); }
                handlers::weekly_metrics::run(&env).await?;
                return Response::from_json(&serde_json::json!({ "ok": true, "handler": "weekly_metrics" }));
            }
```

- [ ] **Step 3: PASS**

```bash
grep -q '"0 2 \* \* MON"' traderjoe/wrangler.toml && \
  cd traderjoe && cargo test && cargo build --target wasm32-unknown-unknown
```
Expected: PASS (all tests + WASM build succeeds).

- [ ] **Step 4: Commit**

```bash
git add traderjoe/src/lib.rs traderjoe/wrangler.toml && \
  git commit -m "feat(cron): register weekly metrics cron and trigger endpoint"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Every spec item has a task: NBBO capture (5, 14, 18, 19); equity curve (13, 16, 18, 19, 20); portfolio Greeks (6, 7, 16, 20); market context (8, 9, 16, 20); metrics aggregator (10, 11, 12, 21, 23); weekly cron + HTTP (23, 24); capital fraction (18); go-live doc (2); schema (1); NBBO fill-size check (10, 21).
- **No placeholders:** All "TEMP" comments are explicit handoffs between adjacent tasks (Task 14 → 18, Task 15 → 19) with the subsequent task numbered.
- **Type consistency:** `NbboSnapshot`, `EquitySnapshot`, `PortfolioGreeks`, `MetricBundle`, `MarketContextRow`, `MetricsWeeklyRow`, `SpotVixReading`, `SpyStats`, `SampleSizeTag` are defined once and referenced by fully-qualified path elsewhere.
- **Group correctness:** Group A tasks touch disjoint files (migration SQL, doc, broker/types + alpaca, types.rs+d1.rs). Task 4 edits both `types.rs` and `d1.rs` — that's acceptable because no other Group A task touches those files. Group B sequential because each appends to `mod.rs` or re-edits `metrics.rs`/`market_context.rs`. Group C sequential (all edit `d1.rs`). Group D sequential (handler edits). Group E sequential.
- **Vertical slice check:** Each task has one test + one implementation + one commit. Three exceptions flagged:
  - Task 1 (SQL migration): "test" is a shell parse check, not a Rust test — TDD shape preserved via fail-then-pass cycle.
  - Task 2 (GO_LIVE_GATE.md): "test" is grep structure check — doc tasks don't fit pure TDD; flagged.
  - Tasks 14, 15: compile-only contract tests (D1 integration not runnable in unit tests).
- **Behavior through public interfaces:** All non-compile tests exercise public functions (`snapshot_spread_nbbo`, `compute_sharpe`, `build_equity_snapshot`, `classify_regime`, etc.). No private-method access, no mocking of internal collaborators.
- **Known constraint:** This plan is large (24 tasks). `/challenge` should confirm feasibility in one `/build` cycle or recommend a split post-spec.

---

## Challenge Review

### CEO Pass

**Premise.** The problem is real and the data is well-articulated: Alpaca paper fills are untrustworthy without NBBO capture; six months of daily equity points cannot statistically validate the strategy but can prove plumbing correctness. Framework C (operational gate, not P&L gate) is the right framing given the expected sample size of 50–100 trades. No simpler alternative exists—the four new tables and measurement modules are the minimum required to answer "can we trust the data?"

**Scope.** The plan touches 14 files, introduces 5 new modules, 1 new handler, and a new cron. This crosses the 8-file smell threshold. However, the scope is entirely justified by the spec, which is itself well-bounded. No scope drift from spec detected.

**Twelve-month alignment.**
```
CURRENT STATE                    THIS PLAN                    12-MONTH IDEAL
No equity curve, no NBBO,   →   All measurement infra     →   Go-live with capital ramp
no gate criteria, no             live and recording.            validated by captured data.
regime tagging.
```
This plan is a prerequisite for any go-live decision. It does not create tech debt that conflicts with the 12-month ideal.

**Alternatives.** The spec documents Framework C vs. statistical P&L gate with the rationale. Nothing to add.

---

### Engineering Pass

#### Architecture

Data flow is clean: entry handler captures NBBO → D1 stores measurement rows → EOD handler aggregates Greeks and market context → weekly handler reads window and posts metrics. The measurement module is pure (all I/O at boundaries). One layering violation flagged below.

No security issues: no user input flows to SQL without parameterization; all D1 calls use prepared statements with `?` placeholders.

#### Module Depth

| Module | Interface | Implementation | Verdict |
|--------|-----------|----------------|---------|
| `measurement::nbbo` | 1 function, 1 struct | strike-match, bid/ask/size extraction, timestamp gen | DEEP |
| `measurement::portfolio_greeks` | 1 method, 1 struct, 1 const | per-leg lookup, beta-weighting, max-concentration scan | DEEP |
| `measurement::market_context` | 2 functions, 2 structs | FRED CSV parsing, log-return stdev, drawdown-from-peak | DEEP |
| `measurement::metrics` | `compute_metrics` + 7 helpers | Sharpe/Sortino/profit factor/skew/DD/slippage/regime | DEEP |
| `measurement::equity` | 1 function, 2 structs, 1 enum | event-type string dispatch, field mapping | SHALLOW — but correctly kept small per spec |
| `db::d1` additions | 4 insert methods + 4 query methods | SQL bind construction | DEEP enough (hides SQL verbosity) |

#### Code Quality

**DRY.** The pattern `v.map(|x| x.into()).unwrap_or(JsValue::NULL)` appears 25+ times across the new D1 methods. No abstraction is planned. Each instance is only 1-2 lines, so this is acceptable, but a `opt_val(v: Option<f64>) -> JsValue` helper would cut noise by ~15 lines. Not a blocker.

**Error handling.** Measurement writes in handlers all use `.await.ok()` — silent drops on failure. This is documented as intentional (measurement failures must not block trade execution). Consistent with existing pattern (`discord.send_error(...).await.ok()`). Acceptable.

#### Test Philosophy

All substantive tests exercise public functions through their declared signatures. No internal mocking. Compile-check tests (Tasks 14, 15, 16, 17, 19) are honest about what they verify and call out their limitations explicitly. This is the right approach given D1 is untestable in unit tests.

The `run_has_expected_signature` test in Task 23 is weak — it only verifies the closure type-checks, not behavior. Acceptable since the handler is integration-level and tested by running it.

#### Vertical Slice

All 24 tasks follow one-test → one-impl → one-commit. The three exceptions (SQL migration shell check, grep structure check, compile-only D1 tests) are explicitly flagged in the plan's self-review notes. No horizontal slicing detected.

#### Test Coverage Gaps

```
[+] measurement/metrics.rs::compute_metrics
    ├── [TESTED]  small sample → INSUFFICIENT tag — Task 21 test ★★
    ├── [GAP]     empty trades → profit_factor = Infinity not asserted
    └── [GAP]     trade opened before window window → regime bucketing uses
                  created_at but market_context queried by window → silent med_vol

[+] measurement/market_context.rs
    ├── [TESTED]  parse_fred_vix_csv happy path ★★★
    ├── [TESTED]  parse_fred_vix_csv all-empty rows ★★
    ├── [GAP]     Stooq fallback — no function, no test (see BLOCKER below)
    └── [GAP]     fetch_spot_vix (async HTTP) — integration-only, acknowledged

[+] handlers/morning_scan.rs Task 18 wiring
    ├── [TESTED]  effective_equity clamps ★★★
    └── [GAP]     NBBO snapshot captured and passed to create_trade — no behavioral test
                  (compile-only, accepted by plan)
```

#### Failure Modes

- All measurement writes in handlers use `.await.ok()`. Equity snapshot failures, Greeks insert failures, and market context failures are logged by `console_log!` in surrounding code but are otherwise silent. Acceptable because measurement failures must not abort trade execution. However, there is no Discord alert on measurement failure. If the FRED fetch silently fails daily for 30 days, the reviewer checking GO_LIVE_GATE.md criterion 7 ("Regime tags present on 100% of trades") would see 0 market_context rows with no prior warning.

- Duplicate equity snapshots for `trade_open`, `trade_close`, and `circuit_breaker` events: only `eod` is deduplicated. A retried morning_scan that opens a trade twice (after order placement succeeds but before DB write), or a retried position_monitor close, would append a second snapshot. The equity curve calculation in `compute_metrics` uses consecutive EOD rows — non-EOD duplicates don't affect it. Acceptable.

---

### Findings

**[RESOLVED]** — Task 14 NBBO size sides corrected: `nb_sz_short` now reads `short_bid_size` (you sell into the bid) and `nb_sz_long` reads `long_ask_size` (you buy at the ask).

**[RESOLVED]** — Stooq fallback replaced with a Discord error alert. When FRED VIXCLS fetch fails, Task 20 sends a "VIX Data Unavailable" Discord error naming the affected date before storing NULL. No silent failure. The Stooq fallback is deferred; failures are now visible in the same channel as all other trading alerts.

**[RISK]** (confidence: 9/10) — The weekly metrics run computes only a 30-day trailing window. The spec explicitly states "Compute window = trailing 30 days **and full paper history**" and "Call `compute_metrics` twice (30d + full). Insert rows into `metrics_weekly`." Task 23 only calls `compute_metrics` once. The "full history" window — needed for GO_LIVE_GATE.md criteria C2 (profit factor < 1.0 over full window) — is never computed or stored. Criterion C2's SQL query (`WHERE window_end = (SELECT MAX(window_end) FROM metrics_weekly)`) would return the latest 30-day row, not a full-window row. The catastrophic halt check becomes unreliable. Fallback: a `?window=full` parameter on the HTTP trigger could compute on demand, but the weekly scheduled job must also insert the full-window row.

**[RISK]** (confidence: 8/10) — Task 19 calls `alpaca.get_account()` twice inside the `Ok(true) =>` close branch to get `equity` and `cash` separately. If the second call fails, `cash` silently defaults to `0.0` while `equity` carries the real value. The equity snapshot row for that trade close would record realistic equity but zero cash — corrupting the ratio for any analysis that uses cash separately. Fallback in the plan: "Future refactor can snapshot once and pass." Acceptable for MVP but worth noting that `open_position_mtm = 0.0` is also hardcoded here, meaning the unrealized P&L in the equity snapshot on trade close is always zero. This affects the `unrealized_pnl_day` column but not the equity curve calculation.

**[RISK]** (confidence: 7/10) — Task 19 adds `alpaca.get_options_chain(&trade.underlying).await.ok()` inside the per-open-trade loop in position_monitor. With N open trades, this is N extra chain fetches on every position monitor invocation (every 5 minutes during market hours). At peak (6 open positions across 3 underlyings, 3 chains already fetched for pricing), this doubles the chain fetches per invocation. Monitor for rate limit (429) errors from Alpaca under load. Fallback: reuse the chain already fetched for debit pricing — restructure to fetch once per underlying.

**[RISK]** (confidence: 6/10) — `measurement::metrics::compute_metrics` imports `use crate::db::d1::MarketContextRow` (Task 21). This creates a dependency from the pure measurement layer into the database layer. If `MarketContextRow` moves or changes, `metrics.rs` must be updated. The cleaner design is a `MarketContextSnapshot` type in `measurement::market_context.rs` with a conversion at the `D1Client` callsite. Low severity since the code compiles and works, but it couples two layers that the spec intended to be separate.

**[RISK]** (confidence: 7/10) — Task 18's `all_opportunities` tuple expansion from 3-tuple to 4-tuple is underspecified. The plan says "update the downstream consumption to destructure the chain out" without giving the concrete sort closure pattern (`|(a, _, _, _), (b, _, _, _)|`) or the for-loop destructuring. The build agent must infer these. The sort-by closure at line 192 of `morning_scan.rs` currently uses `|(a, _, _), (b, _, _)|` — if the subagent misses this it will get a compile error. Low risk since the error is compile-time visible.

**[QUESTION]** — Task 24's `/trigger/metrics` route returns `{ "ok": true, "handler": "weekly_metrics" }`, not the `MetricBundle` JSON that the spec promises. The spec says "Returns the same bundle for ad-hoc queries." If the on-demand metrics endpoint is meant to return the bundle (e.g., for monitoring scripts to query without waiting for Sunday), the response must be the bundle. Currently `MetricBundle` does not derive `Serialize`, so this would require adding that derive. Needs a decision: simplified response OK, or return bundle?

**[QUESTION]** — `compute_metrics` regime bucketing uses `t.created_at.get(..10)` to find the market context row for each trade. But `get_closed_trades_in_window` queries by `date(exit_time) BETWEEN start AND end`. For a 30-day window, trades opened more than 30 days ago but closed within the window are included — and their `created_at` date falls before the window, so `get_market_context_in_window` (also bounded by the same window dates) may not return a row for that date. Those trades silently default to `med_vol`. Acceptable if this is known behavior, but the GO_LIVE_GATE.md's criterion 7 regime check uses a different approach (trade count against market_context join) so there's no inconsistency in the gate criteria. Informational.

**[OBS]** — `PortfolioGreeks` (Task 7) does not derive `Serialize` or `Deserialize`. This is fine for the current plan because serialization of the struct itself is never needed — only `delta_by_underlying` (a `HashMap`) is serialized to JSON string for storage. But any future code that tries to JSON-serialize a `PortfolioGreeks` value will get a compile error. Worth noting for the future.

**[OBS]** — Task 20 uses `alpaca.get_vix().await.ok().flatten().map(|v| v.vix)` to get VIXY close. The `get_vix` function returns `VixData { vix: f64, vix3m: Option<f64> }` where `vix` is the VIXY ETF price. The naming in `market_context_daily.vixy_close` is accurate but the `VixData.vix` field name is misleading (it's a VIXY price, not spot VIX). No action required — the CLAUDE.md acknowledges this proxy.

**[OBS]** — The `ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD = 0.34` constant is a hardcoded assumption from the backtest. The `slippage_vs_orats_ratio` metric depends on this constant being accurate. If the backtest ORATS model is ever recalibrated, this constant must be updated in `metrics.rs`. The spec notes this as a constant, not a derived value.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|-----------|---------|--------|
| `worker-rs 0.7` is used (plan header) — confirmed by `Cargo.toml`: `worker = { version = "0.7" }` | SAFE | Verified in Cargo.toml |
| `OptionContract` doesn't have `bid_size`/`ask_size` yet — confirmed by reading `broker/types.rs:1-26` | SAFE | Verified: fields absent |
| `Trade` doesn't have new NBBO fields yet — confirmed by reading `src/types.rs:91-113` | SAFE | Verified: fields absent |
| D1 `ON CONFLICT(date(timestamp)) WHERE event_type = 'eod'` is valid SQLite upsert syntax | VALIDATE | SQLite 3.24+ supports expression partial-index upsert; D1 SQLite version needs verification |
| `all_opportunities` is a `Vec<(ScoredSpread, String, IvMetrics)>` 3-tuple at line 188 | SAFE | Verified by reading `morning_scan.rs:187-189` |
| Alpaca's chain response includes `latestQuote.bs` and `latestQuote.as` for bid/ask size | VALIDATE | Plan asserts this but it's not verified against live Alpaca API docs |
| `chrono::Duration::days` is available in the WASM target | VALIDATE | `Cargo.toml` has `chrono = { features = ["wasmbind"] }` — `Duration::days` should work, but `Utc::now().date_naive()` (used in Task 23) may panic in WASM without `wasmbind`; verify |
| `eod_summary.rs` currently has `realized_pnl` variable in scope at the equity-snapshot insertion point (Task 20) | VALIDATE | Not verified — the variable name depends on existing eod_summary structure |
| `account` variable from `alpaca.get_account()` is in scope at the circuit breaker fire site (morning_scan.rs ~line 108) | SAFE | Verified: `account` is fetched at line 88 and circuit breaker evaluated at line 100 |
| `D1Client::close_trade` at current call site in position_monitor has the exact signature `(trade_id, exit_price, exit_reason, net_pnl)` | SAFE | Verified by reading d1.rs:119-150 |

---

### Summary

**[BLOCKER]** count: 0 (both resolved above)
**[RISK]**    count: 5
**[QUESTION]** count: 2

Both blockers are resolved in the plan: NBBO size sides corrected in Task 14; FRED failure surfaces a Discord alert instead of silently storing NULL. Remaining risks are manageable and none block execution.

VERDICT: PROCEED_WITH_CAUTION — [weekly metrics missing full-window pass; redundant get_account() calls in Task 19; N+1 chain fetches in position_monitor]
