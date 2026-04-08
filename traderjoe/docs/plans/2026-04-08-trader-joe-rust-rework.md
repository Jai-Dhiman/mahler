# TraderJoe Rust Rework Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the Python Cloudflare Worker with a lean Rust Worker that mechanically executes backtest-validated parameters.
**Spec:** docs/specs/2026-04-08-trader-joe-rust-rework-design.md
**Style:** Follow CLAUDE.md — uv for Python (none here), bun for JS (none here), explicit exceptions not fallbacks.

---

## Task Groups

- **Group A** (no deps): Task 1 (cleanup), Task 2 (migration SQL)
- **Group B** (depends on A): Task 3 (workspace + crate scaffold)
- **Group C** (depends on B, parallel): Task 4 (types.rs), Task 5 (config.rs)
- **Group D** (depends on C, parallel): Task 6 (greeks.rs), Task 7 (iv_rank.rs), Task 8 (broker/types.rs)
- **Group E** (depends on D, parallel): Task 9 (circuit_breaker.rs), Task 10 (position_sizer.rs), Task 11 (screener.rs)
- **Group F** (depends on D, parallel): Task 12 (db/d1.rs), Task 13 (db/kv.rs), Task 14 (discord.rs), Task 15 (alpaca.rs)
- **Group G** (depends on E + F): Task 16 (morning_scan.rs), Task 17 (position_monitor.rs), Task 18 (eod_summary.rs)
- **Group H** (depends on G): Task 19 (lib.rs wiring)

---

### Task 1: Delete Dead Python Code and Update wrangler.toml

**Group:** A (parallel with Task 2)

**Behavior being verified:** `wrangler.toml` no longer references Python, removed bindings, or dead crons; dead Python directories no longer exist.

**Interface under test:** File system state and wrangler.toml content.

**Files:**
- Delete: `src/handlers/afternoon_scan.py`
- Delete: `src/handlers/midday_check.py`
- Delete: `src/core/agents/` (entire directory)
- Delete: `src/core/memory/` (entire directory)
- Delete: `src/core/analysis/regime.py`
- Delete: `src/core/analysis/three_perspective.py` (lives in `src/core/risk/`)
- Delete: `src/core/risk/three_perspective.py`
- Delete: `src/core/risk/dynamic_beta.py`
- Delete: `src/core/risk/dynamic_exit.py`
- Delete: `src/core/risk/validators.py` (only used by dynamic_exit)
- Delete: `src/core/risk/weight_optimizer.py`
- Delete: `src/core/analysis/weight_optimizer.py`
- Delete: `src/core/analysis/exit_optimizer.py`
- Delete: `src/core/analysis/rule_validator.py`
- Delete: `src/core/analysis/greeks_vollib.py`
- Delete: `scripts/` (entire directory)
- Delete: `tests/` (entire directory)
- Delete: `pyproject.toml`
- Delete: `uv.lock`
- Modify: `wrangler.toml`

- [ ] **Step 1: Write the failing test**

```bash
# Verify files exist before deletion (test will fail if they're already gone)
test -f src/handlers/afternoon_scan.py && echo "EXISTS: afternoon_scan.py" || echo "MISSING: afternoon_scan.py"
test -d src/core/agents && echo "EXISTS: agents/" || echo "MISSING: agents/"
test -d src/core/memory && echo "EXISTS: memory/" || echo "MISSING: memory/"
test -f src/core/analysis/regime.py && echo "EXISTS: regime.py" || echo "MISSING: regime.py"
grep -c "python_workers" wrangler.toml && echo "HAS: python_workers flag" || echo "MISSING: python_workers flag"
grep -c "EPISODIC_MEMORY" wrangler.toml && echo "HAS: vectorize binding" || echo "MISSING: vectorize"
```

Expected: All "EXISTS" lines print, `python_workers` flag found, `EPISODIC_MEMORY` found.

- [ ] **Step 2: Run test — verify it FAILS (i.e. current state matches "things to delete exist")**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
test -f src/handlers/afternoon_scan.py && \
test -d src/core/agents && \
test -d src/core/memory && \
test -f src/core/analysis/regime.py && \
grep -q "python_workers" wrangler.toml && \
grep -q "EPISODIC_MEMORY" wrangler.toml && \
echo "PASS: all targets exist, cleanup needed" || echo "FAIL: some targets already gone"
```

Expected: PASS (confirms cleanup is needed).

- [ ] **Step 3: Delete files and update wrangler.toml**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe

# Delete dead handlers
rm -f src/handlers/afternoon_scan.py
rm -f src/handlers/midday_check.py

# Delete dead core modules
rm -rf src/core/agents/
rm -rf src/core/memory/
rm -f src/core/analysis/regime.py
rm -f src/core/risk/three_perspective.py
rm -f src/core/risk/dynamic_beta.py
rm -f src/core/risk/dynamic_exit.py
rm -f src/core/risk/validators.py
rm -f src/core/risk/weight_optimizer.py
rm -f src/core/analysis/weight_optimizer.py
rm -f src/core/analysis/exit_optimizer.py
rm -f src/core/analysis/rule_validator.py
rm -f src/core/analysis/greeks_vollib.py

# Delete Python tooling
rm -rf scripts/
rm -rf tests/
rm -f pyproject.toml
rm -f uv.lock
```

Replace `wrangler.toml` with:

```toml
name = "trader-joe"
main = "trader-joe/build/worker/shim.mjs"
compatibility_date = "2024-12-01"

[build]
command = "cd trader-joe && cargo install worker-build --version 0.4.0 && worker-build --release"

[observability]
enabled = true

[triggers]
crons = [
    "0 14 * * MON-FRI",
    "*/5 14-20 * * MON-FRI",
    "15 20 * * MON-FRI",
]

[[d1_databases]]
binding = "DB"
database_name = "mahler-db"
database_id = "b6cb2eac-2903-46bd-baea-b4ff2dc904d0"
migrations_dir = "src/migrations"

[[kv_namespaces]]
binding = "KV"
id = "4e63db1305a1424ead3565522a47b5f4"

[limits]
cpu_ms = 120_000

[vars]
ENVIRONMENT = "paper"
LOG_LEVEL = "INFO"
```

- [ ] **Step 4: Run test — verify cleanup is complete**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
test ! -f src/handlers/afternoon_scan.py && \
test ! -d src/core/agents && \
test ! -d src/core/memory && \
test ! -f src/core/analysis/regime.py && \
! grep -q "python_workers" wrangler.toml && \
! grep -q "EPISODIC_MEMORY" wrangler.toml && \
grep -q "trader-joe" wrangler.toml && \
echo "PASS: cleanup complete" || echo "FAIL: some targets remain"
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add -A && git commit -m "chore: delete dead python code, update wrangler.toml for rust worker"
```

---

### Task 2: D1 Migration for New Tables

**Group:** A (parallel with Task 1)

**Behavior being verified:** Migration SQL creates the three new TraderJoe tables without affecting the assistant's tables.

**Interface under test:** SQL file content; tables do not conflict with `email_triage_log` or `triage_state`.

**Files:**
- Create: `src/migrations/0017_trader_joe_tables.sql`

- [ ] **Step 1: Write the failing test**

```bash
# Verify migration file does not yet exist
test ! -f /Users/jdhiman/Documents/mahler/traderjoe/src/migrations/0017_trader_joe_tables.sql && \
echo "PASS: migration does not exist yet" || echo "FAIL: already exists"
```

Expected: PASS

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
test ! -f src/migrations/0017_trader_joe_tables.sql && echo "CONFIRM: file absent" || echo "ALREADY EXISTS"
```

Expected: CONFIRM: file absent

- [ ] **Step 3: Create migration file**

```sql
-- trader-joe tables
-- Adds three tables to mahler-db alongside assistant tables (email_triage_log, triage_state)
-- Run with: wrangler d1 execute mahler-db --file=src/migrations/0017_trader_joe_tables.sql

-- All placed trades, full lifecycle from entry to exit
CREATE TABLE IF NOT EXISTS trades (
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

-- Per-symbol daily IV for iv_rank lookback (252-day window)
CREATE TABLE IF NOT EXISTS iv_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    iv REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(symbol, date)
);

-- One row per scan run for ops visibility and Discord summaries
CREATE TABLE IF NOT EXISTS scan_log (
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

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying);
CREATE INDEX IF NOT EXISTS idx_iv_history_symbol_date ON iv_history(symbol, date);
CREATE INDEX IF NOT EXISTS idx_scan_log_time ON scan_log(scan_time);
```

- [ ] **Step 4: Run test — verify file exists and has no conflicts**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
test -f src/migrations/0017_trader_joe_tables.sql && \
! grep -q "email_triage_log\|triage_state" src/migrations/0017_trader_joe_tables.sql && \
grep -q "CREATE TABLE IF NOT EXISTS trades" src/migrations/0017_trader_joe_tables.sql && \
grep -q "CREATE TABLE IF NOT EXISTS iv_history" src/migrations/0017_trader_joe_tables.sql && \
grep -q "CREATE TABLE IF NOT EXISTS scan_log" src/migrations/0017_trader_joe_tables.sql && \
echo "PASS: migration file correct" || echo "FAIL"
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add src/migrations/0017_trader_joe_tables.sql && \
git commit -m "feat(db): add trader-joe tables to mahler-db"
```

---

### Task 3: Workspace Cargo.toml and trader-joe Crate Scaffold

**Group:** B (depends on Group A)

**Behavior being verified:** `cargo build` succeeds for both crates; `trader-joe` has `crate-type = ["cdylib"]` required for WASM.

**Interface under test:** `cargo build --package trader-joe` and `cargo build --package mahler-backtest` both compile without errors.

**Files:**
- Create: `traderjoe/Cargo.toml` (workspace)
- Create: `traderjoe/trader-joe/Cargo.toml`
- Create: `traderjoe/trader-joe/src/lib.rs` (skeleton)
- Create: `traderjoe/trader-joe/src/handlers/mod.rs`
- Create: `traderjoe/trader-joe/src/broker/mod.rs`
- Create: `traderjoe/trader-joe/src/db/mod.rs`
- Create: `traderjoe/trader-joe/src/analysis/mod.rs`
- Create: `traderjoe/trader-joe/src/risk/mod.rs`
- Create: `traderjoe/trader-joe/src/notifications/mod.rs`

- [ ] **Step 1: Write the failing test**

```bash
# Verify workspace does not yet exist
test ! -f /Users/jdhiman/Documents/mahler/traderjoe/Cargo.toml && \
echo "PASS: workspace not yet created" || echo "FAIL: already exists"
```

Expected: PASS

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
test ! -f Cargo.toml && echo "CONFIRM: no workspace yet" || echo "EXISTS"
```

Expected: CONFIRM: no workspace yet

- [ ] **Step 3: Create workspace and crate**

Create `traderjoe/Cargo.toml`:
```toml
[workspace]
members = [
    "mahler-backtest",
    "trader-joe",
]
resolver = "2"
```

Create `traderjoe/trader-joe/Cargo.toml`:
```toml
[package]
name = "trader-joe"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
worker = { version = "0.4", features = ["d1", "kv"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["wasmbind", "serde"] }
thiserror = "1.0"

[dev-dependencies]
# No async in tests — pure functions only
```

Create `traderjoe/trader-joe/src/lib.rs`:
```rust
use worker::*;

mod analysis;
mod broker;
mod config;
mod db;
mod handlers;
mod notifications;
mod risk;
mod types;

#[event(fetch)]
async fn fetch(_req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    Response::ok("trader-joe")
}

#[event(scheduled)]
async fn scheduled(event: ScheduledEvent, env: Env, _ctx: ScheduleContext) {
    let cron = event.cron();
    let result = match cron.as_str() {
        "0 14 * * MON-FRI" => handlers::morning_scan::run(&env).await,
        "*/5 14-20 * * MON-FRI" => handlers::position_monitor::run(&env).await,
        "15 20 * * MON-FRI" => handlers::eod_summary::run(&env).await,
        unknown => {
            console_error!("Unknown cron: {}", unknown);
            Ok(())
        }
    };

    if let Err(e) = result {
        console_error!("Handler error for {}: {:?}", cron, e);
    }
}
```

Create `traderjoe/trader-joe/src/handlers/mod.rs`:
```rust
pub mod eod_summary;
pub mod morning_scan;
pub mod position_monitor;
```

Create `traderjoe/trader-joe/src/broker/mod.rs`:
```rust
pub mod alpaca;
pub mod types;
```

Create `traderjoe/trader-joe/src/db/mod.rs`:
```rust
pub mod d1;
pub mod kv;
```

Create `traderjoe/trader-joe/src/analysis/mod.rs`:
```rust
pub mod greeks;
pub mod iv_rank;
pub mod screener;
```

Create `traderjoe/trader-joe/src/risk/mod.rs`:
```rust
pub mod circuit_breaker;
pub mod position_sizer;
```

Create `traderjoe/trader-joe/src/notifications/mod.rs`:
```rust
pub mod discord;
```

Each handler stub:

`traderjoe/trader-joe/src/handlers/morning_scan.rs`:
```rust
use worker::{Env, Result};

pub async fn run(_env: &Env) -> Result<()> {
    todo!("morning_scan handler — implemented in Task 16")
}
```

`traderjoe/trader-joe/src/handlers/position_monitor.rs`:
```rust
use worker::{Env, Result};

pub async fn run(_env: &Env) -> Result<()> {
    todo!("position_monitor handler — implemented in Task 17")
}
```

`traderjoe/trader-joe/src/handlers/eod_summary.rs`:
```rust
use worker::{Env, Result};

pub async fn run(_env: &Env) -> Result<()> {
    todo!("eod_summary handler — implemented in Task 18")
}
```

Create empty placeholder files for the other modules (implemented in later tasks):
- `trader-joe/src/types.rs` → `// implemented in Task 4`
- `trader-joe/src/config.rs` → `// implemented in Task 5`
- `trader-joe/src/analysis/greeks.rs` → `// implemented in Task 6`
- `trader-joe/src/analysis/iv_rank.rs` → `// implemented in Task 7`
- `trader-joe/src/broker/types.rs` → `// implemented in Task 8`
- `trader-joe/src/risk/circuit_breaker.rs` → `// implemented in Task 9`
- `trader-joe/src/risk/position_sizer.rs` → `// implemented in Task 10`
- `trader-joe/src/analysis/screener.rs` → `// implemented in Task 11`
- `trader-joe/src/db/d1.rs` → `// implemented in Task 12`
- `trader-joe/src/db/kv.rs` → `// implemented in Task 13`
- `trader-joe/src/notifications/discord.rs` → `// implemented in Task 14`
- `trader-joe/src/broker/alpaca.rs` → `// implemented in Task 15`

- [ ] **Step 4: Run test — verify both crates exist in workspace**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
test -f Cargo.toml && \
test -f trader-joe/Cargo.toml && \
test -f trader-joe/src/lib.rs && \
grep -q "mahler-backtest" Cargo.toml && \
grep -q "trader-joe" Cargo.toml && \
grep -q "cdylib" trader-joe/Cargo.toml && \
echo "PASS: workspace and crate scaffold correct" || echo "FAIL"
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add Cargo.toml trader-joe/ && \
git commit -m "feat(trader-joe): initialize rust cloudflare worker crate and workspace"
```

---

### Task 4: types.rs — Domain Types

**Group:** C (parallel with Task 5)

**Behavior being verified:** `CreditSpread::max_loss_per_contract()` computes (width - credit) * 100 correctly.

**Interface under test:** `CreditSpread::max_loss_per_contract(&self) -> f64`

**Files:**
- Modify: `trader-joe/src/types.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In trader-joe/src/types.rs (add after struct definitions)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_spread_max_loss_is_width_minus_credit_times_100() {
        let spread = CreditSpread {
            underlying: "SPY".to_string(),
            spread_type: SpreadType::BullPut,
            short_strike: 480.0,
            long_strike: 475.0,
            expiration: "2026-05-15".to_string(),
            entry_credit: 0.50,
            short_delta: Some(-0.10),
            short_theta: Some(0.05),
            short_iv: Some(0.22),
            long_iv: Some(0.24),
        };
        // width = 5.0, credit = 0.50/share, max_loss = (5.0 - 0.50) * 100 = 450.0
        assert!((spread.max_loss_per_contract() - 450.0).abs() < f64::EPSILON);
    }

    #[test]
    fn trade_status_open_means_not_closed() {
        let status = TradeStatus::Open;
        assert!(!status.is_closed());
    }

    #[test]
    fn trade_status_closed_is_closed() {
        let status = TradeStatus::Closed;
        assert!(status.is_closed());
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- types 2>&1 | tail -5
```

Expected: FAIL — `error[E0425]: cannot find function/struct` or compilation error (types not defined yet).

- [ ] **Step 3: Implement types.rs**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpreadType {
    BullPut,
    BearCall,
}

impl SpreadType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpreadType::BullPut => "bull_put",
            SpreadType::BearCall => "bear_call",
        }
    }
}

impl std::fmt::Display for SpreadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditSpread {
    pub underlying: String,
    pub spread_type: SpreadType,
    pub short_strike: f64,
    pub long_strike: f64,
    pub expiration: String,
    pub entry_credit: f64,
    pub short_delta: Option<f64>,
    pub short_theta: Option<f64>,
    pub short_iv: Option<f64>,
    pub long_iv: Option<f64>,
}

impl CreditSpread {
    pub fn width(&self) -> f64 {
        (self.short_strike - self.long_strike).abs()
    }

    pub fn max_loss_per_contract(&self) -> f64 {
        (self.width() - self.entry_credit) * 100.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradeStatus {
    PendingFill,
    Open,
    Closed,
    Cancelled,
}

impl TradeStatus {
    pub fn is_closed(&self) -> bool {
        matches!(self, TradeStatus::Closed | TradeStatus::Cancelled)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            TradeStatus::PendingFill => "pending_fill",
            TradeStatus::Open => "open",
            TradeStatus::Closed => "closed",
            TradeStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    ProfitTarget,
    StopLoss,
    Expiration,
    Manual,
}

impl ExitReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExitReason::ProfitTarget => "profit_target",
            ExitReason::StopLoss => "stop_loss",
            ExitReason::Expiration => "expiration",
            ExitReason::Manual => "manual",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub created_at: String,
    pub underlying: String,
    pub spread_type: SpreadType,
    pub short_strike: f64,
    pub long_strike: f64,
    pub expiration: String,
    pub contracts: i64,
    pub entry_credit: f64,
    pub max_loss: f64,
    pub broker_order_id: Option<String>,
    pub status: TradeStatus,
    pub fill_price: Option<f64>,
    pub fill_time: Option<String>,
    pub exit_price: Option<f64>,
    pub exit_time: Option<String>,
    pub exit_reason: Option<ExitReason>,
    pub net_pnl: Option<f64>,
    pub iv_rank: Option<f64>,
    pub short_delta: Option<f64>,
    pub short_theta: Option<f64>,
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- types 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/types.rs && \
git commit -m "feat(trader-joe): add domain types (CreditSpread, Trade, TradeStatus)"
```

---

### Task 5: config.rs — SpreadConfig

**Group:** C (parallel with Task 4)

**Behavior being verified:** `SpreadConfig::default()` contains the autoresearch-validated parameters; `should_exit_profit` and `should_exit_stop_loss` return correct booleans for known P&L values.

**Interface under test:** `SpreadConfig::default()`, `SpreadConfig::should_exit_profit(entry_credit, current_debit) -> bool`, `SpreadConfig::should_exit_stop_loss(entry_credit, current_debit) -> bool`

**Files:**
- Modify: `trader-joe/src/config.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_autoresearch_validated_params() {
        let cfg = SpreadConfig::default();
        assert_eq!(cfg.profit_target_pct, 0.25);
        assert_eq!(cfg.stop_loss_pct, 1.25);
        assert_eq!(cfg.min_dte, 30);
        assert_eq!(cfg.max_dte, 45);
        assert!((cfg.min_delta - 0.05).abs() < 1e-9);
        assert!((cfg.max_delta - 0.15).abs() < 1e-9);
    }

    #[test]
    fn exits_at_profit_target_when_debit_is_25_pct_of_credit() {
        let cfg = SpreadConfig::default();
        // Entry credit: $1.00, current debit to close: $0.75
        // P&L = 1.00 - 0.75 = 0.25, which is 25% of 1.00 -> exit
        assert!(cfg.should_exit_profit(1.00, 0.75));
    }

    #[test]
    fn does_not_exit_when_profit_is_below_target() {
        let cfg = SpreadConfig::default();
        // P&L = 1.00 - 0.85 = 0.15, which is 15% of 1.00 -> no exit
        assert!(!cfg.should_exit_profit(1.00, 0.85));
    }

    #[test]
    fn exits_at_stop_loss_when_debit_is_125_pct_of_credit() {
        let cfg = SpreadConfig::default();
        // Entry credit: $1.00, current debit: $2.25
        // Loss = 2.25 - 1.00 = 1.25 = 125% of credit -> exit
        assert!(cfg.should_exit_stop_loss(1.00, 2.25));
    }

    #[test]
    fn does_not_trigger_stop_loss_below_threshold() {
        let cfg = SpreadConfig::default();
        // Entry credit: $1.00, current debit: $2.00 (100% loss, below 125% stop)
        assert!(!cfg.should_exit_stop_loss(1.00, 2.00));
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- config 2>&1 | tail -5
```

Expected: FAIL — compilation error (SpreadConfig not defined).

- [ ] **Step 3: Implement config.rs**

```rust
/// Spread configuration with backtest-validated parameters.
///
/// profit_target=0.25 and stop_loss=1.25 are from autoresearch:
/// iter 1 (profit_target=25%) → Sharpe 0.14, win_rate 57.7% vs baseline 0.10/51%
#[derive(Debug, Clone)]
pub struct SpreadConfig {
    /// Close when profit >= this fraction of entry credit (25%)
    pub profit_target_pct: f64,
    /// Close when loss >= this fraction of entry credit (125%)
    pub stop_loss_pct: f64,
    pub min_dte: i64,
    pub max_dte: i64,
    pub min_delta: f64,
    pub max_delta: f64,
    /// Minimum credit as fraction of spread width (10%)
    pub min_credit_pct: f64,
    /// Minimum spread width in dollars
    pub min_spread_width: f64,
    /// Force-close when DTE reaches this value (gamma explosion risk)
    pub gamma_exit_dte: i64,
    /// Underlyings to scan
    pub underlyings: &'static [&'static str],
    /// Max trades to place per morning scan
    pub max_trades_per_scan: usize,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        SpreadConfig {
            profit_target_pct: 0.25,
            stop_loss_pct: 1.25,
            min_dte: 30,
            max_dte: 45,
            min_delta: 0.05,
            max_delta: 0.15,
            min_credit_pct: 0.10,
            min_spread_width: 2.0,
            gamma_exit_dte: 7,
            underlyings: &["SPY", "QQQ", "IWM"],
            max_trades_per_scan: 3,
        }
    }
}

impl SpreadConfig {
    /// Returns true if the position should be closed for profit.
    ///
    /// Profit condition: current debit to close is <= entry_credit * (1 - profit_target_pct)
    pub fn should_exit_profit(&self, entry_credit: f64, current_debit: f64) -> bool {
        let profit = entry_credit - current_debit;
        profit / entry_credit >= self.profit_target_pct
    }

    /// Returns true if the position should be closed for a stop loss.
    ///
    /// Stop condition: current debit to close is >= entry_credit * (1 + stop_loss_pct)
    pub fn should_exit_stop_loss(&self, entry_credit: f64, current_debit: f64) -> bool {
        let loss = current_debit - entry_credit;
        loss / entry_credit >= self.stop_loss_pct
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- config 2>&1 | tail -10
```

Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/config.rs && \
git commit -m "feat(trader-joe): add SpreadConfig with autoresearch-validated parameters"
```

---

### Task 6: analysis/greeks.rs — Black-Scholes Delta and DTE

**Group:** D (parallel with Tasks 7 and 8)

**Behavior being verified:** `black_scholes_delta` returns correct sign and order-of-magnitude for known option scenarios; `days_to_expiry` returns the correct integer count.

**Interface under test:** `black_scholes_delta(option_type: &str, spot: f64, strike: f64, tte: f64, vol: f64, risk_free: f64) -> f64`, `days_to_expiry(expiration: &str) -> Option<i64>`

**Files:**
- Modify: `trader-joe/src/analysis/greeks.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_delta_is_negative_for_otm_put() {
        // SPY 480, put at 460, 45 DTE, 20% IV, 5% risk-free
        let delta = black_scholes_delta("put", 480.0, 460.0, 45.0 / 365.0, 0.20, 0.05);
        assert!(delta < 0.0, "put delta must be negative, got {}", delta);
        assert!(delta > -1.0, "put delta must be > -1.0, got {}", delta);
        // 460 put is ~4% OTM on SPY 480 with 20% IV at 45 DTE — should be small delta
        assert!(delta.abs() < 0.20, "OTM put delta abs should be < 0.20, got {}", delta.abs());
    }

    #[test]
    fn call_delta_is_positive_for_otm_call() {
        // SPY 480, call at 500, 45 DTE, 20% IV, 5% risk-free
        let delta = black_scholes_delta("call", 480.0, 500.0, 45.0 / 365.0, 0.20, 0.05);
        assert!(delta > 0.0, "call delta must be positive, got {}", delta);
        assert!(delta < 1.0, "call delta must be < 1.0, got {}", delta);
        assert!(delta < 0.20, "OTM call delta should be < 0.20, got {}", delta);
    }

    #[test]
    fn atm_put_delta_is_approximately_negative_half() {
        // ATM option (spot == strike) delta should be close to -0.50 for put
        let delta = black_scholes_delta("put", 480.0, 480.0, 30.0 / 365.0, 0.20, 0.00);
        assert!(
            (delta - (-0.50)).abs() < 0.05,
            "ATM put delta should be ~-0.50, got {}",
            delta
        );
    }

    #[test]
    fn days_to_expiry_returns_correct_positive_count() {
        // Use a fixed future date relative to a known "today"
        // We test the function works with a well-formed date string
        let far_future = "2099-12-31";
        let dte = days_to_expiry(far_future);
        assert!(dte.is_some());
        assert!(dte.unwrap() > 0, "future date must have positive DTE");
    }

    #[test]
    fn days_to_expiry_returns_none_for_invalid_date() {
        let dte = days_to_expiry("not-a-date");
        assert!(dte.is_none());
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- analysis::greeks 2>&1 | tail -5
```

Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement greeks.rs**

```rust
use chrono::{NaiveDate, Utc};

/// Standard normal CDF via Hart (1968) rational approximation.
/// Error < 1.5e-7 for all x.
fn standard_normal_cdf(x: f64) -> f64 {
    const A1: f64 = 0.319381530;
    const A2: f64 = -0.356563782;
    const A3: f64 = 1.781477937;
    const A4: f64 = -1.821255978;
    const A5: f64 = 1.330274429;
    const P: f64 = 0.2316419;

    let t = 1.0 / (1.0 + P * x.abs());
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;

    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

/// Compute Black-Scholes delta for a European option.
///
/// # Arguments
/// - `option_type`: "put" or "call" (case-insensitive)
/// - `spot`: current underlying price
/// - `strike`: option strike price
/// - `tte`: time to expiry in years (e.g. 45.0/365.0)
/// - `vol`: implied volatility (e.g. 0.20 for 20%)
/// - `risk_free`: risk-free rate (e.g. 0.05 for 5%)
///
/// Returns delta in [-1, 1]. Returns 0.0 if inputs are invalid (tte <= 0, vol <= 0).
pub fn black_scholes_delta(
    option_type: &str,
    spot: f64,
    strike: f64,
    tte: f64,
    vol: f64,
    risk_free: f64,
) -> f64 {
    if tte <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return 0.0;
    }

    let d1 = ((spot / strike).ln() + (risk_free + 0.5 * vol * vol) * tte)
        / (vol * tte.sqrt());

    match option_type.to_lowercase().as_str() {
        "call" => standard_normal_cdf(d1),
        "put" => standard_normal_cdf(d1) - 1.0,
        _ => 0.0,
    }
}

/// Returns calendar days until expiration from today (UTC).
///
/// Returns None if the date string is not valid YYYY-MM-DD format.
pub fn days_to_expiry(expiration: &str) -> Option<i64> {
    let exp_date = NaiveDate::parse_from_str(expiration, "%Y-%m-%d").ok()?;
    let today = Utc::now().date_naive();
    Some((exp_date - today).num_days())
}

/// Returns time to expiry in years. Returns 0.0 if expiry is in the past.
pub fn years_to_expiry(expiration: &str) -> f64 {
    days_to_expiry(expiration)
        .map(|d| (d.max(0) as f64) / 365.0)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_delta_is_negative_for_otm_put() {
        let delta = black_scholes_delta("put", 480.0, 460.0, 45.0 / 365.0, 0.20, 0.05);
        assert!(delta < 0.0, "put delta must be negative, got {}", delta);
        assert!(delta > -1.0, "put delta must be > -1.0, got {}", delta);
        assert!(delta.abs() < 0.20, "OTM put delta abs should be < 0.20, got {}", delta.abs());
    }

    #[test]
    fn call_delta_is_positive_for_otm_call() {
        let delta = black_scholes_delta("call", 480.0, 500.0, 45.0 / 365.0, 0.20, 0.05);
        assert!(delta > 0.0, "call delta must be positive, got {}", delta);
        assert!(delta < 1.0, "call delta must be < 1.0, got {}", delta);
        assert!(delta < 0.20, "OTM call delta should be < 0.20, got {}", delta);
    }

    #[test]
    fn atm_put_delta_is_approximately_negative_half() {
        let delta = black_scholes_delta("put", 480.0, 480.0, 30.0 / 365.0, 0.20, 0.00);
        assert!(
            (delta - (-0.50)).abs() < 0.05,
            "ATM put delta should be ~-0.50, got {}",
            delta
        );
    }

    #[test]
    fn days_to_expiry_returns_correct_positive_count() {
        let far_future = "2099-12-31";
        let dte = days_to_expiry(far_future);
        assert!(dte.is_some());
        assert!(dte.unwrap() > 0, "future date must have positive DTE");
    }

    #[test]
    fn days_to_expiry_returns_none_for_invalid_date() {
        let dte = days_to_expiry("not-a-date");
        assert!(dte.is_none());
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- analysis::greeks 2>&1 | tail -10
```

Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/analysis/greeks.rs && \
git commit -m "feat(trader-joe): add black-scholes delta and days_to_expiry"
```

---

### Task 7: analysis/iv_rank.rs — IV Metrics

**Group:** D (parallel with Tasks 6 and 8)

**Behavior being verified:** `calculate_iv_metrics` returns correct iv_rank, iv_percentile, iv_high, and iv_low for a controlled history; handles empty history gracefully.

**Interface under test:** `calculate_iv_metrics(current_iv: f64, history: &[f64]) -> IVMetrics`

**Files:**
- Modify: `trader-joe/src/analysis/iv_rank.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iv_rank_at_midpoint_of_range() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.30, &history);
        // rank = (0.30 - 0.10) / (0.50 - 0.10) * 100 = 50.0
        assert!((metrics.iv_rank - 50.0).abs() < 1e-9);
    }

    #[test]
    fn iv_percentile_counts_days_below_current() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.30, &history);
        // 2 values (0.10, 0.20) are < 0.30 → 2/5 = 40.0%
        assert!((metrics.iv_percentile - 40.0).abs() < 1e-9);
    }

    #[test]
    fn iv_rank_at_maximum_returns_100() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.50, &history);
        assert!((metrics.iv_rank - 100.0).abs() < 1e-9);
    }

    #[test]
    fn iv_rank_at_minimum_returns_0() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.10, &history);
        assert!((metrics.iv_rank - 0.0).abs() < 1e-9);
    }

    #[test]
    fn empty_history_returns_neutral_defaults() {
        let metrics = calculate_iv_metrics(0.25, &[]);
        assert!((metrics.iv_rank - 50.0).abs() < 1e-9);
        assert!((metrics.iv_percentile - 50.0).abs() < 1e-9);
        assert!((metrics.iv_high - 0.25).abs() < 1e-9);
        assert!((metrics.iv_low - 0.25).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- analysis::iv_rank 2>&1 | tail -5
```

Expected: FAIL — `IVMetrics` and `calculate_iv_metrics` not defined.

- [ ] **Step 3: Implement iv_rank.rs**

```rust
/// IV metrics for an underlying, derived from historical data.
#[derive(Debug, Clone)]
pub struct IVMetrics {
    pub current_iv: f64,
    /// IV Rank: (current - 52wk_low) / (52wk_high - 52wk_low) * 100
    pub iv_rank: f64,
    /// IV Percentile: % of historical days where IV was lower than current
    pub iv_percentile: f64,
    pub iv_high: f64,
    pub iv_low: f64,
}

/// Calculates IV Rank and IV Percentile from historical IV data.
///
/// Returns neutral defaults (rank=50, percentile=50) when history is empty.
/// Uses the full history slice — caller is responsible for windowing to 252 days.
pub fn calculate_iv_metrics(current_iv: f64, history: &[f64]) -> IVMetrics {
    if history.is_empty() {
        return IVMetrics {
            current_iv,
            iv_rank: 50.0,
            iv_percentile: 50.0,
            iv_high: current_iv,
            iv_low: current_iv,
        };
    }

    let iv_high = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let iv_low = history.iter().cloned().fold(f64::INFINITY, f64::min);

    let iv_rank = if (iv_high - iv_low).abs() < 1e-12 {
        50.0
    } else {
        ((current_iv - iv_low) / (iv_high - iv_low) * 100.0)
            .max(0.0)
            .min(100.0)
    };

    let days_lower = history.iter().filter(|&&iv| iv < current_iv).count();
    let iv_percentile = (days_lower as f64 / history.len() as f64) * 100.0;

    IVMetrics {
        current_iv,
        iv_rank,
        iv_percentile,
        iv_high,
        iv_low,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iv_rank_at_midpoint_of_range() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.30, &history);
        assert!((metrics.iv_rank - 50.0).abs() < 1e-9);
    }

    #[test]
    fn iv_percentile_counts_days_below_current() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.30, &history);
        assert!((metrics.iv_percentile - 40.0).abs() < 1e-9);
    }

    #[test]
    fn iv_rank_at_maximum_returns_100() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.50, &history);
        assert!((metrics.iv_rank - 100.0).abs() < 1e-9);
    }

    #[test]
    fn iv_rank_at_minimum_returns_0() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.10, &history);
        assert!((metrics.iv_rank - 0.0).abs() < 1e-9);
    }

    #[test]
    fn empty_history_returns_neutral_defaults() {
        let metrics = calculate_iv_metrics(0.25, &[]);
        assert!((metrics.iv_rank - 50.0).abs() < 1e-9);
        assert!((metrics.iv_percentile - 50.0).abs() < 1e-9);
        assert!((metrics.iv_high - 0.25).abs() < 1e-9);
        assert!((metrics.iv_low - 0.25).abs() < 1e-9);
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- analysis::iv_rank 2>&1 | tail -10
```

Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/analysis/iv_rank.rs && \
git commit -m "feat(trader-joe): add IV rank and percentile calculation"
```

---

### Task 8: broker/types.rs — Broker Domain Types

**Group:** D (parallel with Tasks 6 and 7)

**Behavior being verified:** `OptionsChain::get_puts` returns only put contracts for the given expiration; `OptionsChain::get_calls` returns only calls.

**Interface under test:** `OptionsChain::get_puts(expiration: &str) -> Vec<OptionContract>`, `OptionsChain::get_calls(expiration: &str) -> Vec<OptionContract>`

**Files:**
- Modify: `trader-joe/src/broker/types.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_contract(strike: f64, option_type: OptionType, expiration: &str) -> OptionContract {
        OptionContract {
            symbol: format!("TEST{}", strike),
            underlying: "TEST".to_string(),
            expiration: expiration.to_string(),
            strike,
            option_type,
            bid: 1.00,
            ask: 1.05,
            last: 1.02,
            volume: 100,
            open_interest: 500,
            implied_volatility: Some(0.22),
            delta: Some(-0.10),
            gamma: None,
            theta: None,
            vega: None,
        }
    }

    #[test]
    fn get_puts_filters_to_puts_for_given_expiration() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string(), "2026-06-20".to_string()],
            contracts: vec![
                make_contract(460.0, OptionType::Put, "2026-05-15"),
                make_contract(480.0, OptionType::Call, "2026-05-15"),
                make_contract(450.0, OptionType::Put, "2026-06-20"),
            ],
        };
        let puts = chain.get_puts("2026-05-15");
        assert_eq!(puts.len(), 1);
        assert_eq!(puts[0].strike, 460.0);
        assert!(matches!(puts[0].option_type, OptionType::Put));
    }

    #[test]
    fn get_calls_filters_to_calls_for_given_expiration() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                make_contract(460.0, OptionType::Put, "2026-05-15"),
                make_contract(500.0, OptionType::Call, "2026-05-15"),
            ],
        };
        let calls = chain.get_calls("2026-05-15");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].strike, 500.0);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- broker::types 2>&1 | tail -5
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement broker/types.rs**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

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
}

impl OptionContract {
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    pub fn bid_ask_spread_pct(&self) -> f64 {
        let mid = self.mid_price();
        if mid <= 0.0 {
            return 1.0;
        }
        (self.ask - self.bid) / mid
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsChain {
    pub underlying: String,
    pub underlying_price: f64,
    pub expirations: Vec<String>,
    pub contracts: Vec<OptionContract>,
}

impl OptionsChain {
    pub fn get_puts(&self, expiration: &str) -> Vec<OptionContract> {
        self.contracts
            .iter()
            .filter(|c| c.expiration == expiration && c.option_type == OptionType::Put)
            .cloned()
            .collect()
    }

    pub fn get_calls(&self, expiration: &str) -> Vec<OptionContract> {
        self.contracts
            .iter()
            .filter(|c| c.expiration == expiration && c.option_type == OptionType::Call)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub equity: f64,
    pub buying_power: f64,
    pub cash: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub status: OrderStatus,
    pub filled_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SpreadOrder {
    pub underlying: String,
    pub short_occ_symbol: String,
    pub long_occ_symbol: String,
    pub contracts: i64,
    pub limit_price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VixData {
    pub vix: f64,
    pub vix3m: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_contract(strike: f64, option_type: OptionType, expiration: &str) -> OptionContract {
        OptionContract {
            symbol: format!("TEST{}", strike),
            underlying: "TEST".to_string(),
            expiration: expiration.to_string(),
            strike,
            option_type,
            bid: 1.00,
            ask: 1.05,
            last: 1.02,
            volume: 100,
            open_interest: 500,
            implied_volatility: Some(0.22),
            delta: Some(-0.10),
            gamma: None,
            theta: None,
            vega: None,
        }
    }

    #[test]
    fn get_puts_filters_to_puts_for_given_expiration() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string(), "2026-06-20".to_string()],
            contracts: vec![
                make_contract(460.0, OptionType::Put, "2026-05-15"),
                make_contract(480.0, OptionType::Call, "2026-05-15"),
                make_contract(450.0, OptionType::Put, "2026-06-20"),
            ],
        };
        let puts = chain.get_puts("2026-05-15");
        assert_eq!(puts.len(), 1);
        assert_eq!(puts[0].strike, 460.0);
        assert!(matches!(puts[0].option_type, OptionType::Put));
    }

    #[test]
    fn get_calls_filters_to_calls_for_given_expiration() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                make_contract(460.0, OptionType::Put, "2026-05-15"),
                make_contract(500.0, OptionType::Call, "2026-05-15"),
            ],
        };
        let calls = chain.get_calls("2026-05-15");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].strike, 500.0);
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- broker::types 2>&1 | tail -10
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/broker/types.rs && \
git commit -m "feat(trader-joe): add broker domain types (OptionsChain, OptionContract, Order)"
```

---

### Task 9: risk/circuit_breaker.rs — Circuit Breaker

**Group:** E (parallel with Tasks 10 and 11)

**Behavior being verified:** `CircuitBreaker::evaluate` halts at the 2% daily loss threshold; reduces size at intermediate thresholds; returns NORMAL for safe conditions.

**Interface under test:** `CircuitBreaker::evaluate(params: &EvaluateParams) -> RiskState`

**Files:**
- Modify: `trader-joe/src/risk/circuit_breaker.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn safe_params() -> EvaluateParams {
        EvaluateParams {
            starting_daily_equity: 100_000.0,
            current_equity: 100_000.0,
            starting_weekly_equity: 100_000.0,
            peak_equity: 100_000.0,
            current_vix: 18.0,
        }
    }

    #[test]
    fn returns_normal_when_no_thresholds_breached() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&safe_params());
        assert_eq!(state.level, RiskLevel::Normal);
        assert!((state.size_multiplier - 1.0).abs() < 1e-9);
    }

    #[test]
    fn halts_when_daily_loss_exceeds_2_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_equity: 97_900.0, // 2.1% loss
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
        assert!((state.size_multiplier - 0.0).abs() < 1e-9);
    }

    #[test]
    fn reduces_size_when_daily_loss_between_1_5_and_2_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_equity: 98_400.0, // 1.6% loss
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Caution);
        assert!(state.size_multiplier < 1.0, "size must be reduced at caution, got {}", state.size_multiplier);
        assert!(state.size_multiplier > 0.0);
    }

    #[test]
    fn halts_when_vix_exceeds_50() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_vix: 55.0,
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
    }

    #[test]
    fn halts_when_weekly_loss_exceeds_5_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            starting_weekly_equity: 100_000.0,
            current_equity: 94_900.0, // 5.1% weekly loss
            starting_daily_equity: 94_900.0, // no daily loss today
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- risk::circuit_breaker 2>&1 | tail -5
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement circuit_breaker.rs**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Normal,
    Elevated,
    Caution,
    High,
    Critical,
    Halted,
}

#[derive(Debug, Clone)]
pub struct RiskState {
    pub level: RiskLevel,
    pub size_multiplier: f64,
    pub reason: Option<String>,
}

impl RiskState {
    fn normal() -> Self {
        RiskState { level: RiskLevel::Normal, size_multiplier: 1.0, reason: None }
    }

    fn halted(reason: impl Into<String>) -> Self {
        RiskState { level: RiskLevel::Halted, size_multiplier: 0.0, reason: Some(reason.into()) }
    }

    fn caution(multiplier: f64, reason: impl Into<String>) -> Self {
        RiskState { level: RiskLevel::Caution, size_multiplier: multiplier, reason: Some(reason.into()) }
    }

    fn elevated(reason: impl Into<String>) -> Self {
        RiskState { level: RiskLevel::Elevated, size_multiplier: 1.0, reason: Some(reason.into()) }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluateParams {
    pub starting_daily_equity: f64,
    pub current_equity: f64,
    pub starting_weekly_equity: f64,
    pub peak_equity: f64,
    pub current_vix: f64,
}

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Halt if daily loss exceeds this (2%)
    pub daily_halt_pct: f64,
    /// Reduce size if daily loss exceeds this (1.5%)
    pub daily_reduce_pct: f64,
    /// Alert if daily loss exceeds this (1%)
    pub daily_alert_pct: f64,
    /// Halt if weekly loss exceeds this (5%)
    pub weekly_halt_pct: f64,
    /// Reduce size if weekly loss exceeds this (3%)
    pub weekly_caution_pct: f64,
    /// Halt if drawdown from peak exceeds this (15%)
    pub drawdown_halt_pct: f64,
    /// Reduce if drawdown from peak exceeds this (10%)
    pub drawdown_caution_pct: f64,
    /// Halt if VIX exceeds this
    pub vix_halt: f64,
    /// Reduce size significantly if VIX exceeds this
    pub vix_high: f64,
    /// Reduce size if VIX exceeds this
    pub vix_caution: f64,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        CircuitBreaker {
            daily_halt_pct: 0.02,
            daily_reduce_pct: 0.015,
            daily_alert_pct: 0.01,
            weekly_halt_pct: 0.05,
            weekly_caution_pct: 0.03,
            drawdown_halt_pct: 0.15,
            drawdown_caution_pct: 0.10,
            vix_halt: 50.0,
            vix_high: 40.0,
            vix_caution: 30.0,
        }
    }
}

impl CircuitBreaker {
    /// Returns the worst RiskState across all six trigger dimensions.
    ///
    /// Checks daily loss, weekly loss, drawdown, and VIX in order of severity.
    /// The most severe condition wins.
    pub fn evaluate(&self, params: &EvaluateParams) -> RiskState {
        let daily_loss_pct = (params.starting_daily_equity - params.current_equity)
            / params.starting_daily_equity;
        let weekly_loss_pct = (params.starting_weekly_equity - params.current_equity)
            / params.starting_weekly_equity;
        let drawdown_pct = (params.peak_equity - params.current_equity) / params.peak_equity;

        // Check each dimension, worst wins
        let mut worst = RiskState::normal();

        // Daily loss checks
        if daily_loss_pct >= self.daily_halt_pct {
            return RiskState::halted(format!(
                "Daily loss {:.1}% exceeds halt threshold {:.0}%",
                daily_loss_pct * 100.0,
                self.daily_halt_pct * 100.0
            ));
        } else if daily_loss_pct >= self.daily_reduce_pct {
            worst = RiskState::caution(0.50, format!(
                "Daily loss {:.1}% — reducing position size to 50%",
                daily_loss_pct * 100.0
            ));
        } else if daily_loss_pct >= self.daily_alert_pct {
            worst = RiskState::elevated(format!(
                "Daily loss {:.1}% — alert",
                daily_loss_pct * 100.0
            ));
        }

        // Weekly loss checks
        if weekly_loss_pct >= self.weekly_halt_pct {
            return RiskState::halted(format!(
                "Weekly loss {:.1}% exceeds halt threshold {:.0}%",
                weekly_loss_pct * 100.0,
                self.weekly_halt_pct * 100.0
            ));
        } else if weekly_loss_pct >= self.weekly_caution_pct && worst.size_multiplier > 0.50 {
            worst = RiskState::caution(0.50, format!(
                "Weekly loss {:.1}% — reducing to 50%",
                weekly_loss_pct * 100.0
            ));
        }

        // Drawdown checks
        if drawdown_pct >= self.drawdown_halt_pct {
            return RiskState::halted(format!(
                "Drawdown {:.1}% exceeds maximum {:.0}%",
                drawdown_pct * 100.0,
                self.drawdown_halt_pct * 100.0
            ));
        }

        // VIX checks
        if params.current_vix >= self.vix_halt {
            return RiskState::halted(format!("VIX {:.0} exceeds halt threshold {:.0}", params.current_vix, self.vix_halt));
        } else if params.current_vix >= self.vix_high && worst.size_multiplier > 0.25 {
            worst = RiskState::caution(0.25, format!("VIX {:.0} — reducing to 25%", params.current_vix));
        } else if params.current_vix >= self.vix_caution && worst.size_multiplier > 0.50 {
            worst = RiskState::caution(0.50, format!("VIX {:.0} — reducing to 50%", params.current_vix));
        }

        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safe_params() -> EvaluateParams {
        EvaluateParams {
            starting_daily_equity: 100_000.0,
            current_equity: 100_000.0,
            starting_weekly_equity: 100_000.0,
            peak_equity: 100_000.0,
            current_vix: 18.0,
        }
    }

    #[test]
    fn returns_normal_when_no_thresholds_breached() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&safe_params());
        assert_eq!(state.level, RiskLevel::Normal);
        assert!((state.size_multiplier - 1.0).abs() < 1e-9);
    }

    #[test]
    fn halts_when_daily_loss_exceeds_2_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_equity: 97_900.0,
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
        assert!((state.size_multiplier - 0.0).abs() < 1e-9);
    }

    #[test]
    fn reduces_size_when_daily_loss_between_1_5_and_2_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_equity: 98_400.0,
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Caution);
        assert!(state.size_multiplier < 1.0, "size must be reduced at caution, got {}", state.size_multiplier);
        assert!(state.size_multiplier > 0.0);
    }

    #[test]
    fn halts_when_vix_exceeds_50() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_vix: 55.0,
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
    }

    #[test]
    fn halts_when_weekly_loss_exceeds_5_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            starting_weekly_equity: 100_000.0,
            current_equity: 94_900.0,
            starting_daily_equity: 94_900.0,
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- risk::circuit_breaker 2>&1 | tail -10
```

Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/risk/circuit_breaker.rs && \
git commit -m "feat(trader-joe): add circuit breaker with graduated risk levels"
```

---

### Task 10: risk/position_sizer.rs — Position Sizer

**Group:** E (parallel with Tasks 9 and 11)

**Behavior being verified:** `PositionSizer::calculate` returns at most 2% account risk per trade; returns 0 when portfolio heat limit is reached; respects VIX reduction.

**Interface under test:** `PositionSizer::calculate(max_loss_per_contract: f64, equity: f64, positions: &[ExistingPosition], vix: f64) -> SizeResult`

**Files:**
- Modify: `trader-joe/src/risk/position_sizer.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_to_2_pct_account_risk() {
        let sizer = PositionSizer::default();
        // max_risk = 100_000 * 0.02 = $2,000
        // max_loss_per_contract = $500
        // max contracts = 2000 / 500 = 4
        let result = sizer.calculate(500.0, 100_000.0, &[], 20.0);
        assert_eq!(result.contracts, 4);
    }

    #[test]
    fn returns_zero_when_portfolio_heat_at_limit() {
        let sizer = PositionSizer::default();
        // max_heat = 100_000 * 0.20 = $20,000
        // existing positions already use $20,000 in max_loss
        let positions = vec![
            ExistingPosition { underlying: "SPY".to_string(), total_max_loss: 20_000.0 },
        ];
        let result = sizer.calculate(500.0, 100_000.0, &positions, 20.0);
        assert_eq!(result.contracts, 0);
        assert!(result.reason.is_some());
    }

    #[test]
    fn reduces_size_by_50_pct_at_high_vix() {
        let sizer = PositionSizer::default();
        // Normal max = 4 contracts. High VIX (>40) should halve it to 2.
        let result = sizer.calculate(500.0, 100_000.0, &[], 42.0);
        assert_eq!(result.contracts, 2);
    }

    #[test]
    fn minimum_is_1_contract_when_any_size_allowed() {
        let sizer = PositionSizer::default();
        // Very large max_loss per contract forces rounding down, but minimum is 1
        // max_risk = 2000, max_loss = 1999 → 1 contract
        let result = sizer.calculate(1_999.0, 100_000.0, &[], 20.0);
        assert_eq!(result.contracts, 1);
    }

    #[test]
    fn returns_zero_for_single_underlying_at_concentration_limit() {
        let sizer = PositionSizer::default();
        // max_per_underlying = 20% * 100_000 = $20,000
        let positions = vec![
            ExistingPosition { underlying: "SPY".to_string(), total_max_loss: 20_000.0 },
        ];
        let result = sizer.calculate_for_underlying("SPY", 500.0, 100_000.0, &positions, 20.0);
        assert_eq!(result.contracts, 0);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- risk::position_sizer 2>&1 | tail -5
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement position_sizer.rs**

```rust
#[derive(Debug, Clone)]
pub struct ExistingPosition {
    pub underlying: String,
    pub total_max_loss: f64,
}

#[derive(Debug, Clone)]
pub struct SizeResult {
    pub contracts: i64,
    pub reason: Option<String>,
}

impl SizeResult {
    fn zero(reason: impl Into<String>) -> Self {
        SizeResult { contracts: 0, reason: Some(reason.into()) }
    }
}

#[derive(Debug, Clone)]
pub struct PositionSizer {
    pub max_risk_per_trade_pct: f64,
    pub max_portfolio_heat_pct: f64,
    pub max_per_underlying_pct: f64,
    pub high_vix_threshold: f64,
    pub high_vix_reduction: f64,
    pub extreme_vix_threshold: f64,
}

impl Default for PositionSizer {
    fn default() -> Self {
        PositionSizer {
            max_risk_per_trade_pct: 0.02,
            max_portfolio_heat_pct: 0.20,
            max_per_underlying_pct: 0.20,
            high_vix_threshold: 40.0,
            high_vix_reduction: 0.50,
            extreme_vix_threshold: 50.0,
        }
    }
}

impl PositionSizer {
    /// Calculate position size given a spread's max_loss_per_contract and portfolio context.
    ///
    /// Applies four limits in order: per-trade risk, portfolio heat, VIX reduction.
    /// Returns the minimum of all applicable limits, floored to 1 if any contracts allowed.
    pub fn calculate(
        &self,
        max_loss_per_contract: f64,
        equity: f64,
        positions: &[ExistingPosition],
        vix: f64,
    ) -> SizeResult {
        // Check portfolio heat
        let current_heat: f64 = positions.iter().map(|p| p.total_max_loss).sum();
        let max_heat = equity * self.max_portfolio_heat_pct;
        if current_heat >= max_heat {
            return SizeResult::zero(format!(
                "Portfolio heat {:.0}% at limit {:.0}%",
                current_heat / equity * 100.0,
                self.max_portfolio_heat_pct * 100.0
            ));
        }

        if max_loss_per_contract <= 0.0 {
            return SizeResult::zero("max_loss_per_contract must be positive");
        }

        // Max contracts from per-trade risk limit
        let max_risk = equity * self.max_risk_per_trade_pct;
        let mut contracts = (max_risk / max_loss_per_contract).floor() as i64;

        // Also cap by remaining heat capacity
        let remaining_heat = max_heat - current_heat;
        let heat_cap = (remaining_heat / max_loss_per_contract).floor() as i64;
        contracts = contracts.min(heat_cap);

        // VIX reduction
        if vix >= self.extreme_vix_threshold {
            contracts = (contracts as f64 * 0.0).floor() as i64;
        } else if vix >= self.high_vix_threshold {
            contracts = (contracts as f64 * self.high_vix_reduction).floor() as i64;
        }

        if contracts <= 0 {
            return SizeResult::zero("Position size rounds to zero after risk limits");
        }

        // Enforce minimum of 1 when at least 1 contract fits within per-trade risk
        contracts = contracts.max(1);

        SizeResult { contracts, reason: None }
    }

    /// Calculate position size for a specific underlying, checking concentration limit.
    pub fn calculate_for_underlying(
        &self,
        underlying: &str,
        max_loss_per_contract: f64,
        equity: f64,
        positions: &[ExistingPosition],
        vix: f64,
    ) -> SizeResult {
        // Check per-underlying concentration
        let underlying_heat: f64 = positions
            .iter()
            .filter(|p| p.underlying == underlying)
            .map(|p| p.total_max_loss)
            .sum();

        let max_underlying = equity * self.max_per_underlying_pct;
        if underlying_heat >= max_underlying {
            return SizeResult::zero(format!(
                "{} position at concentration limit {:.0}%",
                underlying,
                self.max_per_underlying_pct * 100.0
            ));
        }

        self.calculate(max_loss_per_contract, equity, positions, vix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_to_2_pct_account_risk() {
        let sizer = PositionSizer::default();
        let result = sizer.calculate(500.0, 100_000.0, &[], 20.0);
        assert_eq!(result.contracts, 4);
    }

    #[test]
    fn returns_zero_when_portfolio_heat_at_limit() {
        let sizer = PositionSizer::default();
        let positions = vec![
            ExistingPosition { underlying: "SPY".to_string(), total_max_loss: 20_000.0 },
        ];
        let result = sizer.calculate(500.0, 100_000.0, &positions, 20.0);
        assert_eq!(result.contracts, 0);
        assert!(result.reason.is_some());
    }

    #[test]
    fn reduces_size_by_50_pct_at_high_vix() {
        let sizer = PositionSizer::default();
        let result = sizer.calculate(500.0, 100_000.0, &[], 42.0);
        assert_eq!(result.contracts, 2);
    }

    #[test]
    fn minimum_is_1_contract_when_any_size_allowed() {
        let sizer = PositionSizer::default();
        let result = sizer.calculate(1_999.0, 100_000.0, &[], 20.0);
        assert_eq!(result.contracts, 1);
    }

    #[test]
    fn returns_zero_for_single_underlying_at_concentration_limit() {
        let sizer = PositionSizer::default();
        let positions = vec![
            ExistingPosition { underlying: "SPY".to_string(), total_max_loss: 20_000.0 },
        ];
        let result = sizer.calculate_for_underlying("SPY", 500.0, 100_000.0, &positions, 20.0);
        assert_eq!(result.contracts, 0);
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- risk::position_sizer 2>&1 | tail -10
```

Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/risk/position_sizer.rs && \
git commit -m "feat(trader-joe): add position sizer with heat limit and vix reduction"
```

---

### Task 11: analysis/screener.rs — Options Screener

**Group:** E (parallel with Tasks 9 and 10; depends on Tasks 6, 7, 8)

**Behavior being verified:** `OptionsScreener::screen_chain` returns empty when all options have delta outside 0.05–0.15; returns scored spreads when a valid spread exists; scores higher for better IV percentile.

**Interface under test:** `OptionsScreener::screen_chain(chain: &OptionsChain, iv_metrics: &IVMetrics) -> Vec<ScoredSpread>`

**Files:**
- Modify: `trader-joe/src/analysis/screener.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::iv_rank::IVMetrics;
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};

    fn make_put(strike: f64, delta: f64, bid: f64, ask: f64, expiration: &str) -> OptionContract {
        OptionContract {
            symbol: format!("SPY{}", strike),
            underlying: "SPY".to_string(),
            expiration: expiration.to_string(),
            strike,
            option_type: OptionType::Put,
            bid,
            ask,
            last: (bid + ask) / 2.0,
            volume: 200,
            open_interest: 1000,
            implied_volatility: Some(0.22),
            delta: Some(-delta),
            gamma: None,
            theta: None,
            vega: None,
        }
    }

    fn good_iv() -> IVMetrics {
        IVMetrics { current_iv: 0.25, iv_rank: 60.0, iv_percentile: 65.0, iv_high: 0.40, iv_low: 0.12 }
    }

    // Expiration 40 DTE from now (well within 30-45 range)
    fn exp_40dte() -> String {
        use chrono::{Duration, Utc};
        (Utc::now() + Duration::days(40)).format("%Y-%m-%d").to_string()
    }

    #[test]
    fn rejects_options_with_delta_outside_range() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(460.0, 0.30, 2.00, 2.10, &exp), // delta 0.30 > max 0.15
                make_put(455.0, 0.28, 1.80, 1.90, &exp), // still out of range
            ],
        };
        let results = OptionsScreener::default().screen_chain(&chain, &good_iv());
        assert!(results.is_empty(), "should reject options with delta 0.30");
    }

    #[test]
    fn accepts_valid_bull_put_spread() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(462.0, 0.10, 1.50, 1.60, &exp), // short put, delta 0.10 - in range
                make_put(457.0, 0.07, 0.80, 0.90, &exp), // long put, $5 wide spread
            ],
        };
        let results = OptionsScreener::default().screen_chain(&chain, &good_iv());
        assert!(!results.is_empty(), "should find a valid bull put spread");
        assert!(results[0].score > 0.0);
        assert!(results[0].score <= 1.0);
    }

    #[test]
    fn higher_iv_percentile_scores_higher() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(462.0, 0.10, 1.50, 1.60, &exp),
                make_put(457.0, 0.07, 0.80, 0.90, &exp),
            ],
        };
        let screener = OptionsScreener::default();
        let low_iv = IVMetrics { current_iv: 0.15, iv_rank: 20.0, iv_percentile: 20.0, iv_high: 0.40, iv_low: 0.12 };
        let high_iv = IVMetrics { current_iv: 0.35, iv_rank: 80.0, iv_percentile: 85.0, iv_high: 0.40, iv_low: 0.12 };

        let low_results = screener.screen_chain(&chain, &low_iv);
        let high_results = screener.screen_chain(&chain, &high_iv);

        if !low_results.is_empty() && !high_results.is_empty() {
            assert!(
                high_results[0].score > low_results[0].score,
                "high IV should score higher: {} vs {}",
                high_results[0].score,
                low_results[0].score
            );
        }
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- analysis::screener 2>&1 | tail -5
```

Expected: FAIL — `OptionsScreener` not defined.

- [ ] **Step 3: Implement screener.rs**

```rust
use crate::analysis::greeks::{black_scholes_delta, days_to_expiry, years_to_expiry};
use crate::analysis::iv_rank::IVMetrics;
use crate::broker::types::{OptionContract, OptionType, OptionsChain};
use crate::config::SpreadConfig;
use crate::types::{CreditSpread, SpreadType};

#[derive(Debug, Clone)]
pub struct ScreenerConfig {
    pub min_dte: i64,
    pub max_dte: i64,
    pub min_delta: f64,
    pub max_delta: f64,
    pub min_credit_pct: f64,
    pub min_spread_width: f64,
    pub max_spread_width: f64,
    pub min_open_interest: i64,
    pub min_volume: i64,
    pub max_bid_ask_spread_pct: f64,
}

impl Default for ScreenerConfig {
    fn default() -> Self {
        let cfg = SpreadConfig::default();
        ScreenerConfig {
            min_dte: cfg.min_dte,
            max_dte: cfg.max_dte,
            min_delta: cfg.min_delta,
            max_delta: cfg.max_delta,
            min_credit_pct: cfg.min_credit_pct,
            min_spread_width: cfg.min_spread_width,
            max_spread_width: 10.0,
            min_open_interest: 50,
            min_volume: 1,
            max_bid_ask_spread_pct: 0.12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredSpread {
    pub spread: CreditSpread,
    pub score: f64,
    pub probability_otm: f64,
    pub expected_value: f64,
}

pub struct OptionsScreener {
    config: ScreenerConfig,
}

impl Default for OptionsScreener {
    fn default() -> Self {
        OptionsScreener { config: ScreenerConfig::default() }
    }
}

impl OptionsScreener {
    pub fn new(config: ScreenerConfig) -> Self {
        OptionsScreener { config }
    }

    /// Screen an options chain for credit spread opportunities.
    ///
    /// Returns spreads sorted by score descending. Checks all valid expirations
    /// within the configured DTE range and builds both bull_put and bear_call spreads.
    pub fn screen_chain(&self, chain: &OptionsChain, iv_metrics: &IVMetrics) -> Vec<ScoredSpread> {
        let valid_expirations: Vec<&String> = chain
            .expirations
            .iter()
            .filter(|exp| {
                if let Some(dte) = days_to_expiry(exp) {
                    dte >= self.config.min_dte && dte <= self.config.max_dte
                } else {
                    false
                }
            })
            .collect();

        let mut results = Vec::new();

        for exp in valid_expirations {
            let tte = years_to_expiry(exp);

            let mut puts = self.filter_liquidity(chain.get_puts(exp));
            puts.sort_by(|a, b| b.strike.partial_cmp(&a.strike).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..puts.len() {
                let short = &puts[i];
                let short_delta = self.get_delta(short, chain.underlying_price, tte, iv_metrics.current_iv, "put");
                let abs_delta = short_delta.abs();
                if abs_delta < self.config.min_delta || abs_delta > self.config.max_delta {
                    continue;
                }

                for j in (i + 1)..puts.len() {
                    let long = &puts[j];
                    let width = short.strike - long.strike;
                    if width < self.config.min_spread_width || width > self.config.max_spread_width {
                        continue;
                    }

                    let credit = short.mid_price() - long.mid_price();
                    if credit <= 0.0 {
                        continue;
                    }
                    if credit / width < self.config.min_credit_pct {
                        continue;
                    }

                    let spread = CreditSpread {
                        underlying: chain.underlying.clone(),
                        spread_type: SpreadType::BullPut,
                        short_strike: short.strike,
                        long_strike: long.strike,
                        expiration: exp.clone(),
                        entry_credit: credit,
                        short_delta: Some(-abs_delta),
                        short_theta: short.delta.map(|_| 0.0),
                        short_iv: short.implied_volatility,
                        long_iv: long.implied_volatility,
                    };

                    let scored = self.score_spread(&spread, iv_metrics, abs_delta);
                    results.push(scored);
                    break; // One long strike per short strike (best width)
                }
            }

            let mut calls = self.filter_liquidity(chain.get_calls(exp));
            calls.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..calls.len() {
                let short = &calls[i];
                let short_delta = self.get_delta(short, chain.underlying_price, tte, iv_metrics.current_iv, "call");
                let abs_delta = short_delta.abs();
                if abs_delta < self.config.min_delta || abs_delta > self.config.max_delta {
                    continue;
                }

                for j in (i + 1)..calls.len() {
                    let long = &calls[j];
                    let width = long.strike - short.strike;
                    if width < self.config.min_spread_width || width > self.config.max_spread_width {
                        continue;
                    }

                    let credit = short.mid_price() - long.mid_price();
                    if credit <= 0.0 {
                        continue;
                    }
                    if credit / width < self.config.min_credit_pct {
                        continue;
                    }

                    let spread = CreditSpread {
                        underlying: chain.underlying.clone(),
                        spread_type: SpreadType::BearCall,
                        short_strike: short.strike,
                        long_strike: long.strike,
                        expiration: exp.clone(),
                        entry_credit: credit,
                        short_delta: Some(abs_delta),
                        short_theta: None,
                        short_iv: short.implied_volatility,
                        long_iv: long.implied_volatility,
                    };

                    let scored = self.score_spread(&spread, iv_metrics, abs_delta);
                    results.push(scored);
                    break;
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn filter_liquidity(&self, contracts: Vec<OptionContract>) -> Vec<OptionContract> {
        contracts
            .into_iter()
            .filter(|c| {
                c.open_interest >= self.config.min_open_interest
                    && c.volume >= self.config.min_volume
                    && c.bid > 0.0
                    && c.ask > 0.0
                    && c.bid_ask_spread_pct() <= self.config.max_bid_ask_spread_pct
            })
            .collect()
    }

    fn get_delta(&self, contract: &OptionContract, spot: f64, tte: f64, iv: f64, option_type: &str) -> f64 {
        if let Some(d) = contract.delta {
            return d;
        }
        black_scholes_delta(option_type, spot, contract.strike, tte, iv, 0.05)
    }

    fn score_spread(&self, spread: &CreditSpread, iv_metrics: &IVMetrics, short_delta: f64) -> ScoredSpread {
        let prob_otm = 1.0 - short_delta;
        let credit_per_contract = spread.entry_credit * 100.0;
        let max_loss = spread.max_loss_per_contract();
        let expected_value = credit_per_contract * prob_otm - max_loss * (1.0 - prob_otm);

        // Equal-weight four factors, all normalized to 0-1
        let iv_score = iv_metrics.iv_percentile / 100.0;
        // Delta score peaks at center of range (0.10 for 0.05-0.15 range)
        let delta_score = (1.0 - (short_delta - 0.10).abs() * 10.0).max(0.0).min(1.0);
        let credit_score = (spread.entry_credit / spread.width()).min(0.5) * 2.0;
        let ev_score = (expected_value / (spread.width() * 100.0)).max(0.0).min(1.0);

        let score = (iv_score + delta_score + credit_score + ev_score) / 4.0;

        ScoredSpread { spread: spread.clone(), score, probability_otm: prob_otm, expected_value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};

    fn make_put(strike: f64, delta: f64, bid: f64, ask: f64, expiration: &str) -> OptionContract {
        OptionContract {
            symbol: format!("SPY{}", strike),
            underlying: "SPY".to_string(),
            expiration: expiration.to_string(),
            strike,
            option_type: OptionType::Put,
            bid,
            ask,
            last: (bid + ask) / 2.0,
            volume: 200,
            open_interest: 1000,
            implied_volatility: Some(0.22),
            delta: Some(-delta),
            gamma: None,
            theta: None,
            vega: None,
        }
    }

    fn good_iv() -> IVMetrics {
        IVMetrics { current_iv: 0.25, iv_rank: 60.0, iv_percentile: 65.0, iv_high: 0.40, iv_low: 0.12 }
    }

    fn exp_40dte() -> String {
        use chrono::{Duration, Utc};
        (Utc::now() + Duration::days(40)).format("%Y-%m-%d").to_string()
    }

    #[test]
    fn rejects_options_with_delta_outside_range() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(460.0, 0.30, 2.00, 2.10, &exp),
                make_put(455.0, 0.28, 1.80, 1.90, &exp),
            ],
        };
        let results = OptionsScreener::default().screen_chain(&chain, &good_iv());
        assert!(results.is_empty(), "should reject options with delta 0.30");
    }

    #[test]
    fn accepts_valid_bull_put_spread() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(462.0, 0.10, 1.50, 1.60, &exp),
                make_put(457.0, 0.07, 0.80, 0.90, &exp),
            ],
        };
        let results = OptionsScreener::default().screen_chain(&chain, &good_iv());
        assert!(!results.is_empty(), "should find a valid bull put spread");
        assert!(results[0].score > 0.0);
        assert!(results[0].score <= 1.0);
    }

    #[test]
    fn higher_iv_percentile_scores_higher() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(462.0, 0.10, 1.50, 1.60, &exp),
                make_put(457.0, 0.07, 0.80, 0.90, &exp),
            ],
        };
        let screener = OptionsScreener::default();
        let low_iv = IVMetrics { current_iv: 0.15, iv_rank: 20.0, iv_percentile: 20.0, iv_high: 0.40, iv_low: 0.12 };
        let high_iv = IVMetrics { current_iv: 0.35, iv_rank: 80.0, iv_percentile: 85.0, iv_high: 0.40, iv_low: 0.12 };

        let low_results = screener.screen_chain(&chain, &low_iv);
        let high_results = screener.screen_chain(&chain, &high_iv);

        if !low_results.is_empty() && !high_results.is_empty() {
            assert!(
                high_results[0].score > low_results[0].score,
                "high IV should score higher: {} vs {}",
                high_results[0].score,
                low_results[0].score
            );
        }
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- analysis::screener 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/analysis/screener.rs && \
git commit -m "feat(trader-joe): add options screener with delta/credit/liquidity filters"
```

---

### Task 12: db/d1.rs — D1 Client

**Group:** F (parallel with Tasks 13, 14, 15; no testable pure logic — tests verify serialization helpers)

**Behavior being verified:** `TradeRow::from_d1_result` correctly deserializes a D1 row into a `Trade`; `TradeRow::status_str` returns the correct status string.

**Interface under test:** `TradeRow::from_d1_result(row: &serde_json::Value) -> Option<Trade>`, `Trade::status_str(&self) -> &str`

**Files:**
- Modify: `trader-joe/src/db/d1.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_open_trade_from_d1_row() {
        let row = json!({
            "id": "trade-123",
            "created_at": "2026-04-08T10:00:00",
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 460.0,
            "long_strike": 455.0,
            "expiration": "2026-05-15",
            "contracts": 2,
            "entry_credit": 0.75,
            "max_loss": 425.0,
            "broker_order_id": "order-abc",
            "status": "open",
            "fill_price": 0.74,
            "fill_time": "2026-04-08T10:05:00",
            "exit_price": null,
            "exit_time": null,
            "exit_reason": null,
            "net_pnl": null,
            "iv_rank": 65.0,
            "short_delta": -0.10,
            "short_theta": 0.05
        });

        let trade = TradeRow::from_d1_row(&row).expect("should parse valid row");
        assert_eq!(trade.id, "trade-123");
        assert_eq!(trade.underlying, "SPY");
        assert!(matches!(trade.status, crate::types::TradeStatus::Open));
        assert_eq!(trade.contracts, 2);
        assert!((trade.entry_credit - 0.75).abs() < 1e-9);
    }

    #[test]
    fn returns_none_for_row_with_missing_required_field() {
        let row = json!({ "id": "trade-123" }); // Missing required fields
        let result = TradeRow::from_d1_row(&row);
        assert!(result.is_none());
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- db::d1 2>&1 | tail -5
```

Expected: FAIL — `TradeRow` not defined.

- [ ] **Step 3: Implement db/d1.rs**

```rust
use serde_json::Value;
use worker::{D1Database, Result};

use crate::types::{ExitReason, SpreadType, Trade, TradeStatus};

/// Serialization helpers for D1 row deserialization.
pub struct TradeRow;

impl TradeRow {
    /// Deserialize a D1 row (as JSON Value) into a Trade.
    ///
    /// Returns None if any required field is missing or unparseable.
    pub fn from_d1_row(row: &Value) -> Option<Trade> {
        Some(Trade {
            id: row["id"].as_str()?.to_string(),
            created_at: row["created_at"].as_str()?.to_string(),
            underlying: row["underlying"].as_str()?.to_string(),
            spread_type: match row["spread_type"].as_str()? {
                "bull_put" => SpreadType::BullPut,
                "bear_call" => SpreadType::BearCall,
                _ => return None,
            },
            short_strike: row["short_strike"].as_f64()?,
            long_strike: row["long_strike"].as_f64()?,
            expiration: row["expiration"].as_str()?.to_string(),
            contracts: row["contracts"].as_i64()?,
            entry_credit: row["entry_credit"].as_f64()?,
            max_loss: row["max_loss"].as_f64()?,
            broker_order_id: row["broker_order_id"].as_str().map(str::to_string),
            status: match row["status"].as_str()? {
                "pending_fill" => TradeStatus::PendingFill,
                "open" => TradeStatus::Open,
                "closed" => TradeStatus::Closed,
                "cancelled" => TradeStatus::Cancelled,
                _ => return None,
            },
            fill_price: row["fill_price"].as_f64(),
            fill_time: row["fill_time"].as_str().map(str::to_string),
            exit_price: row["exit_price"].as_f64(),
            exit_time: row["exit_time"].as_str().map(str::to_string),
            exit_reason: row["exit_reason"].as_str().and_then(|s| match s {
                "profit_target" => Some(ExitReason::ProfitTarget),
                "stop_loss" => Some(ExitReason::StopLoss),
                "expiration" => Some(ExitReason::Expiration),
                "manual" => Some(ExitReason::Manual),
                _ => None,
            }),
            net_pnl: row["net_pnl"].as_f64(),
            iv_rank: row["iv_rank"].as_f64(),
            short_delta: row["short_delta"].as_f64(),
            short_theta: row["short_theta"].as_f64(),
        })
    }
}

/// Client for Cloudflare D1 database operations.
pub struct D1Client {
    db: D1Database,
}

impl D1Client {
    pub fn new(db: D1Database) -> Self {
        D1Client { db }
    }

    /// Insert a new trade. Panics (via ?) if the underlying D1 call fails.
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
    ) -> Result<()> {
        self.db
            .prepare(
                "INSERT INTO trades (id, underlying, spread_type, short_strike, long_strike,
                expiration, contracts, entry_credit, max_loss, broker_order_id,
                iv_rank, short_delta, short_theta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                id.into(),
                underlying.into(),
                spread_type.as_str().into(),
                short_strike.into(),
                long_strike.into(),
                expiration.into(),
                contracts.into(),
                entry_credit.into(),
                max_loss.into(),
                broker_order_id.map(|s| s.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                iv_rank.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                short_delta.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                short_theta.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
            ])?
            .run()
            .await?;
        Ok(())
    }

    /// Get all open trades (status = 'open').
    pub async fn get_open_trades(&self) -> Result<Vec<Trade>> {
        let result = self
            .db
            .prepare("SELECT * FROM trades WHERE status IN ('open', 'pending_fill')")
            .all()
            .await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(TradeRow::from_d1_row).collect())
    }

    /// Update trade status and exit fields when closing a position.
    pub async fn close_trade(
        &self,
        trade_id: &str,
        exit_price: f64,
        exit_reason: &ExitReason,
        net_pnl: f64,
    ) -> Result<()> {
        use chrono::Utc;
        let now = Utc::now().to_rfc3339();
        self.db
            .prepare(
                "UPDATE trades SET status = 'closed', exit_price = ?, exit_time = ?,
                exit_reason = ?, net_pnl = ? WHERE id = ?",
            )
            .bind(&[
                exit_price.into(),
                now.as_str().into(),
                exit_reason.as_str().into(),
                net_pnl.into(),
                trade_id.into(),
            ])?
            .run()
            .await?;
        Ok(())
    }

    /// Mark a pending_fill trade as open after broker confirms fill.
    pub async fn mark_trade_filled(&self, trade_id: &str, fill_price: f64) -> Result<()> {
        use chrono::Utc;
        let now = Utc::now().to_rfc3339();
        self.db
            .prepare(
                "UPDATE trades SET status = 'open', fill_price = ?, fill_time = ? WHERE id = ?",
            )
            .bind(&[fill_price.into(), now.as_str().into(), trade_id.into()])?
            .run()
            .await?;
        Ok(())
    }

    /// Get historical IV values for a symbol (up to lookback_days).
    pub async fn get_iv_history(&self, symbol: &str, lookback_days: i64) -> Result<Vec<f64>> {
        let result = self
            .db
            .prepare(
                "SELECT iv FROM iv_history WHERE symbol = ?
                AND date >= date('now', ? || ' days')
                ORDER BY date ASC",
            )
            .bind(&[symbol.into(), format!("-{}", lookback_days).as_str().into()])?
            .all()
            .await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(|r| r["iv"].as_f64()).collect())
    }

    /// Upsert today's IV for a symbol.
    pub async fn upsert_iv_history(&self, symbol: &str, date: &str, iv: f64) -> Result<()> {
        self.db
            .prepare(
                "INSERT INTO iv_history (symbol, date, iv) VALUES (?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET iv = excluded.iv",
            )
            .bind(&[symbol.into(), date.into(), iv.into()])?
            .run()
            .await?;
        Ok(())
    }

    /// Insert a scan log entry.
    pub async fn log_scan(
        &self,
        scan_type: &str,
        underlyings_scanned: i64,
        opportunities_found: i64,
        trades_placed: i64,
        vix: Option<f64>,
        circuit_breaker_active: bool,
        duration_ms: i64,
        notes: Option<&str>,
    ) -> Result<()> {
        use chrono::Utc;
        let now = Utc::now().to_rfc3339();
        self.db
            .prepare(
                "INSERT INTO scan_log (scan_time, scan_type, underlyings_scanned,
                opportunities_found, trades_placed, vix, circuit_breaker_active,
                duration_ms, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                now.as_str().into(),
                scan_type.into(),
                underlyings_scanned.into(),
                opportunities_found.into(),
                trades_placed.into(),
                vix.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                (circuit_breaker_active as i32).into(),
                duration_ms.into(),
                notes.map(|s| s.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
            ])?
            .run()
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_open_trade_from_d1_row() {
        let row = json!({
            "id": "trade-123",
            "created_at": "2026-04-08T10:00:00",
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 460.0,
            "long_strike": 455.0,
            "expiration": "2026-05-15",
            "contracts": 2,
            "entry_credit": 0.75,
            "max_loss": 425.0,
            "broker_order_id": "order-abc",
            "status": "open",
            "fill_price": 0.74,
            "fill_time": "2026-04-08T10:05:00",
            "exit_price": null,
            "exit_time": null,
            "exit_reason": null,
            "net_pnl": null,
            "iv_rank": 65.0,
            "short_delta": -0.10,
            "short_theta": 0.05
        });

        let trade = TradeRow::from_d1_row(&row).expect("should parse valid row");
        assert_eq!(trade.id, "trade-123");
        assert_eq!(trade.underlying, "SPY");
        assert!(matches!(trade.status, crate::types::TradeStatus::Open));
        assert_eq!(trade.contracts, 2);
        assert!((trade.entry_credit - 0.75).abs() < 1e-9);
    }

    #[test]
    fn returns_none_for_row_with_missing_required_field() {
        let row = json!({ "id": "trade-123" });
        let result = TradeRow::from_d1_row(&row);
        assert!(result.is_none());
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- db::d1 2>&1 | tail -10
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/db/d1.rs && \
git commit -m "feat(trader-joe): add D1 client with trade CRUD, iv_history, scan_log"
```

---

### Task 13: db/kv.rs — KV Client

**Group:** F (parallel with Tasks 12, 14, 15)

**Behavior being verified:** `CircuitBreakerState::serialize` round-trips through JSON; `DailyStats::default` has expected zero values.

**Interface under test:** `CircuitBreakerState` serialization, `DailyStats::default()`

**Files:**
- Modify: `trader-joe/src/db/kv.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_state_round_trips_json() {
        let state = CircuitBreakerState {
            halted: true,
            reason: Some("Daily loss 2.1%".to_string()),
            triggered_at: Some("2026-04-08T10:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let deserialized: CircuitBreakerState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.halted, true);
        assert_eq!(deserialized.reason, Some("Daily loss 2.1%".to_string()));
    }

    #[test]
    fn daily_stats_default_has_zero_pnl() {
        let stats = DailyStats::default();
        assert!((stats.realized_pnl - 0.0).abs() < 1e-9);
        assert_eq!(stats.trades_closed, 0);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- db::kv 2>&1 | tail -5
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement db/kv.rs**

```rust
use serde::{Deserialize, Serialize};
use worker::{kv::KvStore, Result};

const KEY_CIRCUIT_BREAKER: &str = "circuit_breaker:status";
const KEY_DAILY_STATS: &str = "daily_stats";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    pub halted: bool,
    pub reason: Option<String>,
    pub triggered_at: Option<String>,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        CircuitBreakerState { halted: false, reason: None, triggered_at: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStats {
    pub date: String,
    pub starting_equity: f64,
    pub realized_pnl: f64,
    pub trades_placed: i64,
    pub trades_closed: i64,
}

impl Default for DailyStats {
    fn default() -> Self {
        DailyStats {
            date: String::new(),
            starting_equity: 0.0,
            realized_pnl: 0.0,
            trades_placed: 0,
            trades_closed: 0,
        }
    }
}

pub struct KvClient {
    kv: KvStore,
}

impl KvClient {
    pub fn new(kv: KvStore) -> Self {
        KvClient { kv }
    }

    pub async fn get_circuit_breaker(&self) -> Result<CircuitBreakerState> {
        match self.kv.get(KEY_CIRCUIT_BREAKER).json::<CircuitBreakerState>().await? {
            Some(state) => Ok(state),
            None => Ok(CircuitBreakerState::default()),
        }
    }

    pub async fn set_circuit_breaker(&self, state: &CircuitBreakerState) -> Result<()> {
        let json = serde_json::to_string(state)
            .map_err(|e| worker::Error::RustError(e.to_string()))?;
        self.kv.put(KEY_CIRCUIT_BREAKER, json)?.execute().await?;
        Ok(())
    }

    pub async fn reset_circuit_breaker(&self) -> Result<()> {
        self.set_circuit_breaker(&CircuitBreakerState::default()).await
    }

    pub async fn get_daily_stats(&self) -> Result<DailyStats> {
        use chrono::Utc;
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let key = format!("{}:{}", KEY_DAILY_STATS, today);
        match self.kv.get(&key).json::<DailyStats>().await? {
            Some(stats) => Ok(stats),
            None => Ok(DailyStats { date: today, ..DailyStats::default() }),
        }
    }

    pub async fn set_daily_stats(&self, stats: &DailyStats) -> Result<()> {
        let key = format!("{}:{}", KEY_DAILY_STATS, stats.date);
        let json = serde_json::to_string(stats)
            .map_err(|e| worker::Error::RustError(e.to_string()))?;
        // TTL: 3 days (to survive weekends)
        self.kv.put(&key, json)?.expiration_ttl(60 * 60 * 24 * 3).execute().await?;
        Ok(())
    }

    /// Store a deduplication key with a TTL (for circuit breaker alert dedup).
    pub async fn set_with_ttl(&self, key: &str, value: &str, ttl_seconds: u64) -> Result<()> {
        self.kv.put(key, value)?.expiration_ttl(ttl_seconds).execute().await?;
        Ok(())
    }

    pub async fn get_raw(&self, key: &str) -> Result<Option<String>> {
        self.kv.get(key).text().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_state_round_trips_json() {
        let state = CircuitBreakerState {
            halted: true,
            reason: Some("Daily loss 2.1%".to_string()),
            triggered_at: Some("2026-04-08T10:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let deserialized: CircuitBreakerState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.halted, true);
        assert_eq!(deserialized.reason, Some("Daily loss 2.1%".to_string()));
    }

    #[test]
    fn daily_stats_default_has_zero_pnl() {
        let stats = DailyStats::default();
        assert!((stats.realized_pnl - 0.0).abs() < 1e-9);
        assert_eq!(stats.trades_closed, 0);
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- db::kv 2>&1 | tail -10
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/db/kv.rs && \
git commit -m "feat(trader-joe): add KV client for circuit breaker state and daily stats"
```

---

### Task 14: notifications/discord.rs — Discord Client

**Group:** F (parallel with Tasks 12, 13, 15)

**Behavior being verified:** `build_trade_embed` returns an embed JSON with the correct color (green for trade placed), symbol, and strike fields.

**Interface under test:** `build_trade_embed(underlying: &str, short_strike: f64, long_strike: f64, contracts: i64, credit: f64, order_id: &str) -> serde_json::Value`

**Files:**
- Modify: `trader-joe/src/notifications/discord.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_embed_has_green_color_and_correct_title() {
        let embed = build_trade_embed("SPY", 460.0, 455.0, 2, 0.75, "order-abc");
        assert_eq!(embed["color"], 0x57F287); // Discord green
        let title = embed["title"].as_str().expect("title field");
        assert!(title.contains("SPY"), "title should mention underlying");
    }

    #[test]
    fn error_embed_has_red_color() {
        let embed = build_error_embed("Scan failed", "NullPointerError: at line 42");
        assert_eq!(embed["color"], 0xED4245); // Discord red
    }

    #[test]
    fn scan_summary_embed_reports_correct_trade_count() {
        let embed = build_scan_summary_embed(
            "morning",
            3,     // underlyings_scanned
            5,     // opportunities_found
            2,     // trades_placed
            Some(22.5), // vix
        );
        let description = embed["description"].as_str().expect("description");
        assert!(description.contains("2"), "should mention 2 trades placed");
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- notifications::discord 2>&1 | tail -5
```

Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement notifications/discord.rs**

```rust
use serde_json::{json, Value};
use worker::{Fetch, Headers, Method, Request, RequestInit, Result};

const COLOR_GREEN: u32 = 0x57F287;
const COLOR_RED: u32 = 0xED4245;
const COLOR_BLUE: u32 = 0x5865F2;
const COLOR_YELLOW: u32 = 0xFEE75C;

/// Build a Discord embed for a placed trade.
pub fn build_trade_embed(
    underlying: &str,
    short_strike: f64,
    long_strike: f64,
    contracts: i64,
    credit: f64,
    order_id: &str,
) -> Value {
    json!({
        "title": format!("Trade Placed: {} Bull Put Spread", underlying),
        "color": COLOR_GREEN,
        "description": format!(
            "Placed {} contract(s) of ${}/{} spread",
            contracts, short_strike, long_strike
        ),
        "fields": [
            { "name": "Credit", "value": format!("${:.2}/share", credit), "inline": true },
            { "name": "Max Loss", "value": format!("${:.0}/contract", (short_strike - long_strike - credit) * 100.0), "inline": true },
            { "name": "Order ID", "value": format!("`{}`", order_id), "inline": false }
        ]
    })
}

/// Build a Discord embed for a position exit.
pub fn build_exit_embed(
    underlying: &str,
    short_strike: f64,
    long_strike: f64,
    exit_reason: &str,
    net_pnl: f64,
) -> Value {
    let color = if net_pnl >= 0.0 { COLOR_GREEN } else { COLOR_RED };
    json!({
        "title": format!("Position Closed: {}", underlying),
        "color": color,
        "description": format!(
            "${}/{} spread closed via {}. P&L: ${:.0}",
            short_strike, long_strike, exit_reason, net_pnl
        )
    })
}

/// Build a Discord embed for scan summary.
pub fn build_scan_summary_embed(
    scan_type: &str,
    underlyings_scanned: i64,
    opportunities_found: i64,
    trades_placed: i64,
    vix: Option<f64>,
) -> Value {
    let vix_str = vix.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "N/A".to_string());
    json!({
        "title": format!("{} Scan Complete", scan_type.replace('_', " ").to_uppercase()),
        "color": COLOR_BLUE,
        "description": format!(
            "Scanned {} underlyings, found {} opportunities, placed {} trade(s). VIX: {}",
            underlyings_scanned, opportunities_found, trades_placed, vix_str
        )
    })
}

/// Build a Discord embed for an error.
pub fn build_error_embed(title: &str, error: &str) -> Value {
    json!({
        "title": title,
        "color": COLOR_RED,
        "description": format!("```\n{}\n```", &error[..error.len().min(1500)])
    })
}

/// Build a Discord embed for EOD summary.
pub fn build_eod_embed(
    date: &str,
    realized_pnl: f64,
    trades_closed: i64,
    win_rate: f64,
) -> Value {
    let color = if realized_pnl >= 0.0 { COLOR_GREEN } else { COLOR_RED };
    json!({
        "title": format!("EOD Summary — {}", date),
        "color": color,
        "fields": [
            { "name": "Realized P&L", "value": format!("${:.0}", realized_pnl), "inline": true },
            { "name": "Trades Closed", "value": trades_closed.to_string(), "inline": true },
            { "name": "Win Rate", "value": format!("{:.0}%", win_rate * 100.0), "inline": true }
        ]
    })
}

/// Client for sending Discord notifications via webhook/bot.
pub struct DiscordClient {
    bot_token: String,
    channel_id: String,
}

impl DiscordClient {
    pub fn new(bot_token: impl Into<String>, channel_id: impl Into<String>) -> Self {
        DiscordClient {
            bot_token: bot_token.into(),
            channel_id: channel_id.into(),
        }
    }

    /// Send embeds to the configured Discord channel.
    pub async fn send(&self, content: &str, embeds: Vec<Value>) -> Result<()> {
        let url = format!(
            "https://discord.com/api/v10/channels/{}/messages",
            self.channel_id
        );
        let body = serde_json::to_string(&json!({
            "content": content,
            "embeds": embeds
        }))
        .map_err(|e| worker::Error::RustError(e.to_string()))?;

        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json")?;
        headers.set("Authorization", &format!("Bot {}", self.bot_token))?;

        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));

        let req = Request::new_with_init(&url, &init)?;
        Fetch::Request(req).send().await?;
        Ok(())
    }

    pub async fn send_trade_placed(
        &self,
        underlying: &str,
        short_strike: f64,
        long_strike: f64,
        contracts: i64,
        credit: f64,
        order_id: &str,
    ) -> Result<()> {
        let embed = build_trade_embed(underlying, short_strike, long_strike, contracts, credit, order_id);
        self.send("**Trade Placed**", vec![embed]).await
    }

    pub async fn send_scan_summary(
        &self,
        scan_type: &str,
        underlyings_scanned: i64,
        opportunities_found: i64,
        trades_placed: i64,
        vix: Option<f64>,
    ) -> Result<()> {
        let embed = build_scan_summary_embed(scan_type, underlyings_scanned, opportunities_found, trades_placed, vix);
        self.send("", vec![embed]).await
    }

    pub async fn send_error(&self, title: &str, error: &str) -> Result<()> {
        let embed = build_error_embed(title, error);
        self.send("**Error**", vec![embed]).await
    }

    pub async fn send_eod_summary(&self, date: &str, pnl: f64, closed: i64, win_rate: f64) -> Result<()> {
        let embed = build_eod_embed(date, pnl, closed, win_rate);
        self.send("", vec![embed]).await
    }

    pub async fn send_position_exit(
        &self,
        underlying: &str,
        short_strike: f64,
        long_strike: f64,
        exit_reason: &str,
        net_pnl: f64,
    ) -> Result<()> {
        let embed = build_exit_embed(underlying, short_strike, long_strike, exit_reason, net_pnl);
        self.send("", vec![embed]).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_embed_has_green_color_and_correct_title() {
        let embed = build_trade_embed("SPY", 460.0, 455.0, 2, 0.75, "order-abc");
        assert_eq!(embed["color"], COLOR_GREEN);
        let title = embed["title"].as_str().expect("title field");
        assert!(title.contains("SPY"), "title should mention underlying");
    }

    #[test]
    fn error_embed_has_red_color() {
        let embed = build_error_embed("Scan failed", "NullPointerError: at line 42");
        assert_eq!(embed["color"], COLOR_RED);
    }

    #[test]
    fn scan_summary_embed_reports_correct_trade_count() {
        let embed = build_scan_summary_embed("morning", 3, 5, 2, Some(22.5));
        let description = embed["description"].as_str().expect("description");
        assert!(description.contains("2"), "should mention 2 trades placed");
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- notifications::discord 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/notifications/discord.rs && \
git commit -m "feat(trader-joe): add discord client with trade/scan/exit/error embeds"
```

---

### Task 15: broker/alpaca.rs — Alpaca HTTP Client

**Group:** F (parallel with Tasks 12, 13, 14)

**Behavior being verified:** `build_occ_symbol` produces the correct OCC-format option symbol for a known SPY option.

**Interface under test:** `build_occ_symbol(underlying: &str, expiration: &str, option_type: &str, strike: f64) -> String`

**Files:**
- Modify: `trader-joe/src/broker/alpaca.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occ_symbol_format_matches_alpaca_convention() {
        // SPY put, expiration 2026-05-15, strike $460.00
        // Expected: SPY260515P00460000
        let symbol = build_occ_symbol("SPY", "2026-05-15", "P", 460.0);
        assert_eq!(symbol, "SPY260515P00460000");
    }

    #[test]
    fn occ_symbol_with_fractional_strike() {
        // Strike $462.50 → 00462500
        let symbol = build_occ_symbol("SPY", "2026-05-15", "P", 462.5);
        assert_eq!(symbol, "SPY260515P00462500");
    }

    #[test]
    fn occ_symbol_for_call() {
        let symbol = build_occ_symbol("QQQ", "2026-06-20", "C", 500.0);
        assert_eq!(symbol, "QQQ260620C00500000");
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- broker::alpaca 2>&1 | tail -5
```

Expected: FAIL — `build_occ_symbol` not defined.

- [ ] **Step 3: Implement broker/alpaca.rs**

```rust
use serde_json::Value;
use worker::{Fetch, Headers, Method, Request, RequestInit, Result};

use crate::broker::types::{
    Account, Bar, OptionContract, OptionType, OptionsChain, Order, OrderStatus,
    SpreadOrder, VixData,
};

/// Build an OCC-format option symbol.
///
/// Format: UNDERLYINGYYMMDDCBBBBB (C=Call, P=Put, BBBBB=strike * 1000)
/// Example: SPY260515P00460000
pub fn build_occ_symbol(underlying: &str, expiration: &str, option_type: &str, strike: f64) -> String {
    // expiration: "2026-05-15" → "260515"
    let parts: Vec<&str> = expiration.split('-').collect();
    let yy = &parts[0][2..]; // last 2 digits of year
    let mm = parts[1];
    let dd = parts[2];
    let strike_int = (strike * 1000.0).round() as u64;
    format!("{}{}{}{}{}{:08}", underlying, yy, mm, dd, option_type, strike_int)
}

pub struct AlpacaClient {
    api_key: String,
    secret_key: String,
    base_url: String,
    data_url: String,
}

impl AlpacaClient {
    pub fn new(api_key: impl Into<String>, secret_key: impl Into<String>, paper: bool) -> Self {
        AlpacaClient {
            api_key: api_key.into(),
            secret_key: secret_key.into(),
            base_url: if paper {
                "https://paper-api.alpaca.markets".to_string()
            } else {
                "https://api.alpaca.markets".to_string()
            },
            data_url: "https://data.alpaca.markets".to_string(),
        }
    }

    fn auth_headers(&self) -> Result<Headers> {
        let mut h = Headers::new();
        h.set("APCA-API-KEY-ID", &self.api_key)?;
        h.set("APCA-API-SECRET-KEY", &self.secret_key)?;
        h.set("Accept", "application/json")?;
        Ok(h)
    }

    async fn get(&self, url: &str) -> Result<Value> {
        let headers = self.auth_headers()?;
        let mut init = RequestInit::new();
        init.with_method(Method::Get).with_headers(headers);
        let req = Request::new_with_init(url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        resp.json().await
    }

    pub async fn is_market_open(&self) -> Result<bool> {
        let url = format!("{}/v2/clock", self.base_url);
        let json: Value = self.get(&url).await?;
        Ok(json["is_open"].as_bool().unwrap_or(false))
    }

    pub async fn get_account(&self) -> Result<Account> {
        let url = format!("{}/v2/account", self.base_url);
        let json: Value = self.get(&url).await?;
        Ok(Account {
            equity: json["equity"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            buying_power: json["buying_power"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            cash: json["cash"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
        })
    }

    pub async fn get_vix(&self) -> Result<Option<VixData>> {
        let url = format!(
            "{}/v2/stocks/snapshots?symbols=VIXY,UVXY&feed=iex",
            self.data_url
        );
        match self.get(&url).await {
            Ok(json) => {
                let vix = json["VIXY"]["latestTrade"]["p"].as_f64();
                Ok(vix.map(|v| VixData { vix: v, vix3m: None }))
            }
            Err(_) => Ok(None),
        }
    }

    pub async fn get_options_chain(&self, symbol: &str) -> Result<OptionsChain> {
        let url = format!(
            "{}/v2/options/snapshots/{}?feed=indicative&limit=1000",
            self.data_url, symbol
        );
        let json: Value = self.get(&url).await?;

        let underlying_price = json["underlying"]["latestTrade"]["p"]
            .as_f64()
            .unwrap_or(0.0);

        let mut contracts: Vec<OptionContract> = Vec::new();
        let mut expirations: std::collections::HashSet<String> = std::collections::HashSet::new();

        if let Some(snapshots) = json["snapshots"].as_object() {
            for (occ_symbol, data) in snapshots {
                let greeks = &data["greeks"];
                let quote = &data["latestQuote"];

                // Parse OCC symbol to extract fields
                // Format: SPY260515P00460000
                if occ_symbol.len() < 15 {
                    continue;
                }
                let underlying = &occ_symbol[..symbol.len()];
                let rest = &occ_symbol[symbol.len()..];
                if rest.len() < 9 {
                    continue;
                }
                let date_part = &rest[..6]; // YYMMDD
                let option_char = &rest[6..7]; // C or P
                let strike_part = &rest[7..]; // 8 digits

                let strike = strike_part.parse::<f64>().unwrap_or(0.0) / 1000.0;
                let expiration = format!(
                    "20{}-{}-{}",
                    &date_part[..2],
                    &date_part[2..4],
                    &date_part[4..6]
                );
                let option_type = if option_char == "C" { OptionType::Call } else { OptionType::Put };

                expirations.insert(expiration.clone());

                contracts.push(OptionContract {
                    symbol: occ_symbol.clone(),
                    underlying: underlying.to_string(),
                    expiration,
                    strike,
                    option_type,
                    bid: quote["bp"].as_f64().unwrap_or(0.0),
                    ask: quote["ap"].as_f64().unwrap_or(0.0),
                    last: quote["ap"].as_f64().unwrap_or(0.0),
                    volume: data["dailyBar"]["v"].as_i64().unwrap_or(0),
                    open_interest: data["openInterest"].as_i64().unwrap_or(0),
                    implied_volatility: data["impliedVolatility"].as_f64(),
                    delta: greeks["delta"].as_f64(),
                    gamma: greeks["gamma"].as_f64(),
                    theta: greeks["theta"].as_f64(),
                    vega: greeks["vega"].as_f64(),
                });
            }
        }

        let mut expirations_vec: Vec<String> = expirations.into_iter().collect();
        expirations_vec.sort();

        Ok(OptionsChain {
            underlying: symbol.to_string(),
            underlying_price,
            expirations: expirations_vec,
            contracts,
        })
    }

    pub async fn get_bars(&self, symbol: &str, limit: i64) -> Result<Vec<Bar>> {
        let url = format!(
            "{}/v2/stocks/{}/bars?timeframe=1Day&limit={}&adjustment=raw&feed=iex",
            self.data_url, symbol, limit
        );
        let json: Value = self.get(&url).await?;
        let bars = json["bars"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|b| Bar {
                timestamp: b["t"].as_str().unwrap_or("").to_string(),
                open: b["o"].as_f64().unwrap_or(0.0),
                high: b["h"].as_f64().unwrap_or(0.0),
                low: b["l"].as_f64().unwrap_or(0.0),
                close: b["c"].as_f64().unwrap_or(0.0),
                volume: b["v"].as_i64().unwrap_or(0),
            })
            .collect();
        Ok(bars)
    }

    pub async fn place_spread_order(&self, order: &SpreadOrder) -> Result<Order> {
        let url = format!("{}/v2/orders", self.base_url);
        let body = serde_json::to_string(&serde_json::json!({
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "limit_price": format!("{:.2}", order.limit_price),
            "legs": [
                {
                    "symbol": order.short_occ_symbol,
                    "ratio_qty": order.contracts.to_string(),
                    "side": "sell_to_open",
                    "position_effect": "open"
                },
                {
                    "symbol": order.long_occ_symbol,
                    "ratio_qty": order.contracts.to_string(),
                    "side": "buy_to_open",
                    "position_effect": "open"
                }
            ]
        }))
        .map_err(|e| worker::Error::RustError(e.to_string()))?;

        let mut headers = self.auth_headers()?;
        headers.set("Content-Type", "application/json")?;

        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));

        let req = Request::new_with_init(&url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        let json: Value = resp.json().await?;

        Ok(Order {
            id: json["id"].as_str().unwrap_or("").to_string(),
            status: OrderStatus::Pending,
            filled_price: None,
        })
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let url = format!("{}/v2/orders/{}", self.base_url, order_id);
        let headers = self.auth_headers()?;
        let mut init = RequestInit::new();
        init.with_method(Method::Delete).with_headers(headers);
        let req = Request::new_with_init(&url, &init)?;
        Fetch::Request(req).send().await?;
        Ok(())
    }

    pub async fn get_order(&self, order_id: &str) -> Result<Order> {
        let url = format!("{}/v2/orders/{}", self.base_url, order_id);
        let json: Value = self.get(&url).await?;
        let status = match json["status"].as_str().unwrap_or("") {
            "filled" => OrderStatus::Filled,
            "canceled" | "cancelled" => OrderStatus::Cancelled,
            "rejected" => OrderStatus::Rejected,
            _ => OrderStatus::Pending,
        };
        let filled_price = json["filled_avg_price"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok());
        Ok(Order { id: order_id.to_string(), status, filled_price })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occ_symbol_format_matches_alpaca_convention() {
        let symbol = build_occ_symbol("SPY", "2026-05-15", "P", 460.0);
        assert_eq!(symbol, "SPY260515P00460000");
    }

    #[test]
    fn occ_symbol_with_fractional_strike() {
        let symbol = build_occ_symbol("SPY", "2026-05-15", "P", 462.5);
        assert_eq!(symbol, "SPY260515P00462500");
    }

    #[test]
    fn occ_symbol_for_call() {
        let symbol = build_occ_symbol("QQQ", "2026-06-20", "C", 500.0);
        assert_eq!(symbol, "QQQ260620C00500000");
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- broker::alpaca 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/broker/alpaca.rs && \
git commit -m "feat(trader-joe): add alpaca HTTP client with OCC symbol builder and order placement"
```

---

### Task 16: handlers/morning_scan.rs — Morning Scan Handler

**Group:** G (depends on Group E + F)

**Behavior being verified:** The morning scan skips all processing when circuit breaker is halted (today's date) and sends a Discord notification that it's skipping.

**Interface under test:** Handler coordination logic extracted as `should_skip_due_to_circuit_breaker(halted: bool, triggered_today: bool) -> bool`

**Files:**
- Modify: `trader-joe/src/handlers/morning_scan.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_when_circuit_breaker_is_halted_today() {
        assert!(should_skip_due_to_circuit_breaker(true, true));
    }

    #[test]
    fn does_not_skip_when_circuit_breaker_halted_on_previous_day() {
        // Previous day halt → auto-reset, do not skip
        assert!(!should_skip_due_to_circuit_breaker(true, false));
    }

    #[test]
    fn does_not_skip_when_circuit_breaker_not_halted() {
        assert!(!should_skip_due_to_circuit_breaker(false, false));
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- handlers::morning_scan 2>&1 | tail -5
```

Expected: FAIL — function not defined.

- [ ] **Step 3: Implement handlers/morning_scan.rs**

```rust
use worker::{console_log, Env, Result};

use crate::analysis::iv_rank::calculate_iv_metrics;
use crate::analysis::screener::OptionsScreener;
use crate::broker::alpaca::{build_occ_symbol, AlpacaClient};
use crate::broker::types::SpreadOrder;
use crate::config::SpreadConfig;
use crate::db::d1::D1Client;
use crate::db::kv::KvClient;
use crate::notifications::discord::DiscordClient;
use crate::risk::circuit_breaker::{CircuitBreaker, EvaluateParams, RiskLevel};
use crate::risk::position_sizer::{ExistingPosition, PositionSizer};
use crate::types::SpreadType;

/// Returns true if scanning should be skipped because the circuit breaker
/// was tripped today (not yesterday — yesterday gets auto-reset).
pub fn should_skip_due_to_circuit_breaker(halted: bool, triggered_today: bool) -> bool {
    halted && triggered_today
}

/// Extract the date portion (YYYY-MM-DD) from an ISO timestamp string.
fn date_from_iso(iso: &str) -> &str {
    &iso[..iso.len().min(10)]
}

pub async fn run(env: &Env) -> Result<()> {
    use chrono::Utc;
    let start = Utc::now();
    console_log!("Morning scan starting at {}", start.to_rfc3339());

    let db = D1Client::new(env.d1("DB")?);
    let kv = KvClient::new(env.kv("KV")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    // Check circuit breaker
    let cb_state = kv.get_circuit_breaker().await?;
    let today = Utc::now().format("%Y-%m-%d").to_string();
    let triggered_today = cb_state.triggered_at.as_deref()
        .map(|t| date_from_iso(t) == today)
        .unwrap_or(false);

    if should_skip_due_to_circuit_breaker(cb_state.halted, triggered_today) {
        console_log!("Circuit breaker active today: {}", cb_state.reason.as_deref().unwrap_or("unknown"));
        discord.send_error(
            "Morning Scan Skipped",
            &format!("Circuit breaker active: {}", cb_state.reason.as_deref().unwrap_or("unknown")),
        ).await?;
        return Ok(());
    }

    // Auto-reset circuit breaker if it was tripped on a prior day
    if cb_state.halted && !triggered_today {
        console_log!("Auto-resetting circuit breaker (previous day trigger)");
        kv.reset_circuit_breaker().await?;
    }

    let alpaca = AlpacaClient::new(
        env.var("ALPACA_API_KEY")?.to_string(),
        env.var("ALPACA_SECRET_KEY")?.to_string(),
        env.var("ENVIRONMENT")?.to_string() == "paper",
    );

    if !alpaca.is_market_open().await? {
        console_log!("Market is closed, skipping scan");
        return Ok(());
    }

    let account = alpaca.get_account().await?;
    let vix_data = alpaca.get_vix().await?;
    let current_vix = vix_data.as_ref().map(|v| v.vix).unwrap_or(20.0);

    // Graduated risk evaluation
    let daily_stats = kv.get_daily_stats().await?;
    let starting_daily = if daily_stats.starting_equity > 0.0 {
        daily_stats.starting_equity
    } else {
        account.equity
    };

    let cb = CircuitBreaker::default();
    let risk_state = cb.evaluate(&EvaluateParams {
        starting_daily_equity: starting_daily,
        current_equity: account.equity,
        starting_weekly_equity: starting_daily, // simplified: use daily as weekly proxy
        peak_equity: account.equity.max(starting_daily),
        current_vix,
    });

    if risk_state.level == RiskLevel::Halted {
        let reason = risk_state.reason.as_deref().unwrap_or("risk threshold exceeded");
        console_log!("Risk evaluation halted: {}", reason);
        let mut new_state = crate::db::kv::CircuitBreakerState {
            halted: true,
            reason: Some(reason.to_string()),
            triggered_at: Some(Utc::now().to_rfc3339()),
        };
        kv.set_circuit_breaker(&new_state).await?;
        discord.send_error("Trading Halted", reason).await?;
        return Ok(());
    }

    let screener = OptionsScreener::default();
    let sizer = PositionSizer::default();
    let cfg = SpreadConfig::default();

    let open_trades = db.get_open_trades().await?;
    let existing_positions: Vec<ExistingPosition> = open_trades
        .iter()
        .map(|t| ExistingPosition {
            underlying: t.underlying.clone(),
            total_max_loss: t.max_loss * t.contracts as f64,
        })
        .collect();

    let mut all_opportunities = Vec::new();

    for symbol in cfg.underlyings {
        console_log!("Scanning {}...", symbol);

        let chain = match alpaca.get_options_chain(symbol).await {
            Ok(c) => c,
            Err(e) => {
                console_log!("Failed to fetch chain for {}: {:?}", symbol, e);
                continue;
            }
        };

        if chain.contracts.is_empty() {
            console_log!("{}: no contracts", symbol);
            continue;
        }

        let history = db.get_iv_history(symbol, 252).await.unwrap_or_default();

        let atm_iv = chain.contracts.iter()
            .filter(|c| (c.strike - chain.underlying_price).abs() < chain.underlying_price * 0.02)
            .filter_map(|c| c.implied_volatility)
            .fold((0.0f64, 0usize), |(sum, n), iv| (sum + iv, n + 1));
        let current_iv = if atm_iv.1 > 0 { atm_iv.0 / atm_iv.1 as f64 } else { 0.20 };
        let iv_metrics = calculate_iv_metrics(current_iv, &history);

        let mut spreads = screener.screen_chain(&chain, &iv_metrics);
        for scored in spreads.drain(..).take(2) {
            all_opportunities.push((scored, symbol.to_string(), iv_metrics.clone()));
        }
    }

    all_opportunities.sort_by(|(a, _, _), (b, _, _)| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut trades_placed = 0usize;

    for (scored, _symbol, iv_metrics) in all_opportunities.into_iter().take(cfg.max_trades_per_scan) {
        let spread = &scored.spread;

        let option_type = match spread.spread_type {
            SpreadType::BullPut => "P",
            SpreadType::BearCall => "C",
        };

        let size_result = sizer.calculate_for_underlying(
            &spread.underlying,
            spread.max_loss_per_contract(),
            account.equity,
            &existing_positions,
            current_vix,
        );

        let size_result = crate::risk::position_sizer::SizeResult {
            contracts: (size_result.contracts as f64 * risk_state.size_multiplier) as i64,
            reason: size_result.reason,
        };

        if size_result.contracts == 0 {
            console_log!(
                "Skipping {}: size = 0 ({})",
                spread.underlying,
                size_result.reason.as_deref().unwrap_or("unknown")
            );
            continue;
        }

        let short_occ = build_occ_symbol(&spread.underlying, &spread.expiration, option_type, spread.short_strike);
        let long_occ = build_occ_symbol(&spread.underlying, &spread.expiration, option_type, spread.long_strike);

        let order = SpreadOrder {
            underlying: spread.underlying.clone(),
            short_occ_symbol: short_occ,
            long_occ_symbol: long_occ,
            contracts: size_result.contracts,
            limit_price: spread.entry_credit,
        };

        let placed = match alpaca.place_spread_order(&order).await {
            Ok(o) => o,
            Err(e) => {
                console_log!("Order placement failed for {}: {:?}", spread.underlying, e);
                discord.send_error(
                    &format!("Order Failed: {}", spread.underlying),
                    &format!("{:?}", e),
                ).await.ok();
                continue;
            }
        };

        let trade_id = uuid_v4();
        match db.create_trade(
            &trade_id,
            &spread.underlying,
            &spread.spread_type,
            spread.short_strike,
            spread.long_strike,
            &spread.expiration,
            size_result.contracts,
            spread.entry_credit,
            spread.max_loss_per_contract(),
            Some(&placed.id),
            Some(iv_metrics.iv_rank),
            spread.short_delta,
            spread.short_theta,
        ).await {
            Ok(_) => {}
            Err(e) => {
                // Ghost trade prevention: DB write failed after order placed — cancel immediately
                console_log!("CRITICAL: DB write failed for order {}. Cancelling.", placed.id);
                alpaca.cancel_order(&placed.id).await.ok();
                discord.send_error(
                    "GHOST TRADE ALERT",
                    &format!("Order {} placed but DB write failed: {:?}. Order cancelled.", placed.id, e),
                ).await.ok();
                continue;
            }
        }

        discord.send_trade_placed(
            &spread.underlying,
            spread.short_strike,
            spread.long_strike,
            size_result.contracts,
            spread.entry_credit,
            &placed.id,
        ).await.ok();

        trades_placed += 1;
    }

    let duration_ms = (Utc::now() - start).num_milliseconds();
    db.log_scan(
        "morning",
        cfg.underlyings.len() as i64,
        0, // opportunities_found simplified
        trades_placed as i64,
        vix_data.map(|v| v.vix),
        false,
        duration_ms,
        None,
    ).await.ok();

    discord.send_scan_summary(
        "morning",
        cfg.underlyings.len() as i64,
        0,
        trades_placed as i64,
        vix_data.map(|v| v.vix),
    ).await.ok();

    console_log!("Morning scan complete. Placed {} trade(s) in {}ms", trades_placed, duration_ms);
    Ok(())
}

/// Minimal UUID v4 for Cloudflare Workers (no rand crate needed for WASM).
fn uuid_v4() -> String {
    use worker::js_sys::Math;
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (Math::random() * 0xffff_ffff_u64 as f64) as u64,
        (Math::random() * 0xffff_u64 as f64) as u64,
        (Math::random() * 0x0fff_u64 as f64) as u64,
        ((Math::random() * 0x3fff_u64 as f64) as u64) | 0x8000,
        (Math::random() * 0x0000_ffff_ffff_u64 as f64) as u64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_when_circuit_breaker_is_halted_today() {
        assert!(should_skip_due_to_circuit_breaker(true, true));
    }

    #[test]
    fn does_not_skip_when_circuit_breaker_halted_on_previous_day() {
        assert!(!should_skip_due_to_circuit_breaker(true, false));
    }

    #[test]
    fn does_not_skip_when_circuit_breaker_not_halted() {
        assert!(!should_skip_due_to_circuit_breaker(false, false));
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- handlers::morning_scan 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/handlers/morning_scan.rs && \
git commit -m "feat(trader-joe): implement morning scan handler with ghost trade prevention"
```

---

### Task 17: handlers/position_monitor.rs — Position Monitor

**Group:** G (depends on Group E + F)

**Behavior being verified:** `check_exit_conditions` correctly identifies a profit target exit, a stop loss exit, and a gamma explosion exit for known P&L values.

**Interface under test:** `check_exit_conditions(entry_credit: f64, current_debit: f64, dte: i64, cfg: &SpreadConfig) -> Option<ExitReason>`

**Files:**
- Modify: `trader-joe/src/handlers/position_monitor.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpreadConfig;

    #[test]
    fn identifies_profit_target_exit() {
        let cfg = SpreadConfig::default();
        // Entry credit $1.00, current debit to close $0.70 → 30% profit > 25% target
        let result = check_exit_conditions(1.00, 0.70, 40, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::ProfitTarget)));
    }

    #[test]
    fn identifies_stop_loss_exit() {
        let cfg = SpreadConfig::default();
        // Entry credit $1.00, current debit $2.30 → 130% loss > 125% stop
        let result = check_exit_conditions(1.00, 2.30, 40, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::StopLoss)));
    }

    #[test]
    fn identifies_gamma_explosion_exit() {
        let cfg = SpreadConfig::default();
        // DTE = 5, below gamma_exit_dte (7) — force exit regardless of P&L
        let result = check_exit_conditions(1.00, 0.90, 5, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::Expiration)));
    }

    #[test]
    fn no_exit_when_within_normal_range() {
        let cfg = SpreadConfig::default();
        // 10% profit at 30 DTE — no exit conditions met
        let result = check_exit_conditions(1.00, 0.90, 30, &cfg);
        assert!(result.is_none());
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- handlers::position_monitor 2>&1 | tail -5
```

Expected: FAIL — `check_exit_conditions` not defined.

- [ ] **Step 3: Implement handlers/position_monitor.rs**

```rust
use worker::{console_log, Env, Result};

use crate::broker::alpaca::AlpacaClient;
use crate::config::SpreadConfig;
use crate::db::d1::D1Client;
use crate::db::kv::KvClient;
use crate::notifications::discord::DiscordClient;
use crate::types::ExitReason;

/// Determines if an open position should be closed, and why.
///
/// Checks in priority order: gamma explosion (DTE) → profit target → stop loss.
pub fn check_exit_conditions(
    entry_credit: f64,
    current_debit: f64,
    dte: i64,
    cfg: &SpreadConfig,
) -> Option<ExitReason> {
    // Force exit when gamma risk becomes excessive
    if dte <= cfg.gamma_exit_dte {
        return Some(ExitReason::Expiration);
    }
    if cfg.should_exit_profit(entry_credit, current_debit) {
        return Some(ExitReason::ProfitTarget);
    }
    if cfg.should_exit_stop_loss(entry_credit, current_debit) {
        return Some(ExitReason::StopLoss);
    }
    None
}

pub async fn run(env: &Env) -> Result<()> {
    use crate::analysis::greeks::days_to_expiry;

    console_log!("Position monitor starting...");

    let db = D1Client::new(env.d1("DB")?);
    let kv = KvClient::new(env.kv("KV")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    // Skip if circuit breaker is halted
    let cb_state = kv.get_circuit_breaker().await?;
    if cb_state.halted {
        return Ok(());
    }

    let alpaca = AlpacaClient::new(
        env.var("ALPACA_API_KEY")?.to_string(),
        env.var("ALPACA_SECRET_KEY")?.to_string(),
        env.var("ENVIRONMENT")?.to_string() == "paper",
    );

    if !alpaca.is_market_open().await? {
        return Ok(());
    }

    let open_trades = db.get_open_trades().await?;
    if open_trades.is_empty() {
        return Ok(());
    }

    let cfg = SpreadConfig::default();

    for trade in &open_trades {
        // Check for pending_fill → mark as open if filled
        if matches!(trade.status, crate::types::TradeStatus::PendingFill) {
            if let Some(ref order_id) = trade.broker_order_id {
                match alpaca.get_order(order_id).await {
                    Ok(order) => {
                        if matches!(order.status, crate::broker::types::OrderStatus::Filled) {
                            let fill_price = order.filled_price.unwrap_or(trade.entry_credit);
                            db.mark_trade_filled(&trade.id, fill_price).await.ok();
                            console_log!("Trade {} filled at {:.2}", trade.id, fill_price);
                        }
                    }
                    Err(e) => console_log!("Could not check order {}: {:?}", order_id, e),
                }
            }
            continue; // Don't check exits for pending trades
        }

        let dte = days_to_expiry(&trade.expiration).unwrap_or(0);
        let current_debit = match get_spread_debit(&alpaca, trade).await {
            Some(d) => d,
            None => {
                console_log!("Could not get current price for trade {}", trade.id);
                continue;
            }
        };

        if let Some(exit_reason) = check_exit_conditions(trade.entry_credit, current_debit, dte, &cfg) {
            console_log!(
                "Exiting {} {}/{} via {:?} (entry: {:.2}, current: {:.2}, DTE: {})",
                trade.underlying,
                trade.short_strike,
                trade.long_strike,
                exit_reason,
                trade.entry_credit,
                current_debit,
                dte,
            );

            // Close the spread: buy back short, sell long
            let net_pnl = (trade.entry_credit - current_debit) * 100.0 * trade.contracts as f64;

            db.close_trade(&trade.id, current_debit, &exit_reason, net_pnl).await?;

            discord.send_position_exit(
                &trade.underlying,
                trade.short_strike,
                trade.long_strike,
                exit_reason.as_str(),
                net_pnl,
            ).await.ok();
        }
    }

    Ok(())
}

/// Get the current debit to close a spread (bid of short + ask of long, as a net debit).
async fn get_spread_debit(alpaca: &AlpacaClient, trade: &crate::types::Trade) -> Option<f64> {
    let chain = alpaca.get_options_chain(&trade.underlying).await.ok()?;

    let option_type = match trade.spread_type {
        crate::types::SpreadType::BullPut => crate::broker::types::OptionType::Put,
        crate::types::SpreadType::BearCall => crate::broker::types::OptionType::Call,
    };

    let short = chain.contracts.iter().find(|c| {
        (c.strike - trade.short_strike).abs() < 0.01
            && c.expiration == trade.expiration
            && c.option_type == option_type
    })?;

    let long = chain.contracts.iter().find(|c| {
        (c.strike - trade.long_strike).abs() < 0.01
            && c.expiration == trade.expiration
            && c.option_type == option_type
    })?;

    // Cost to close: buy back short (pay ask), sell long (receive bid)
    Some(short.ask - long.bid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpreadConfig;

    #[test]
    fn identifies_profit_target_exit() {
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 0.70, 40, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::ProfitTarget)));
    }

    #[test]
    fn identifies_stop_loss_exit() {
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 2.30, 40, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::StopLoss)));
    }

    #[test]
    fn identifies_gamma_explosion_exit() {
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 0.90, 5, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::Expiration)));
    }

    #[test]
    fn no_exit_when_within_normal_range() {
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 0.90, 30, &cfg);
        assert!(result.is_none());
    }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- handlers::position_monitor 2>&1 | tail -10
```

Expected: `test result: ok. 4 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/handlers/position_monitor.rs && \
git commit -m "feat(trader-joe): implement position monitor with profit/stop/gamma exits"
```

---

### Task 18: handlers/eod_summary.rs — EOD Summary Handler

**Group:** G (depends on Group E + F)

**Behavior being verified:** `calculate_win_rate` returns 1.0 for zero closed trades; returns 0.5 for 1 win and 1 loss.

**Interface under test:** `calculate_win_rate(closed_trades: &[ClosedTradeSummary]) -> f64`

**Files:**
- Modify: `trader-joe/src/handlers/eod_summary.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_rate_is_1_for_no_trades() {
        // No closed trades → return 1.0 (not 0/0 = NaN)
        let rate = calculate_win_rate(&[]);
        assert!((rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn win_rate_is_50_pct_for_one_win_one_loss() {
        let trades = vec![
            ClosedTradeSummary { net_pnl: 50.0 },
            ClosedTradeSummary { net_pnl: -30.0 },
        ];
        let rate = calculate_win_rate(&trades);
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn win_rate_is_1_for_all_winners() {
        let trades = vec![
            ClosedTradeSummary { net_pnl: 50.0 },
            ClosedTradeSummary { net_pnl: 25.0 },
        ];
        assert!((calculate_win_rate(&trades) - 1.0).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- handlers::eod_summary 2>&1 | tail -5
```

Expected: FAIL — `ClosedTradeSummary` and `calculate_win_rate` not defined.

- [ ] **Step 3: Implement handlers/eod_summary.rs**

```rust
use worker::{console_log, Env, Result};

use crate::broker::alpaca::AlpacaClient;
use crate::db::d1::D1Client;
use crate::db::kv::KvClient;
use crate::notifications::discord::DiscordClient;

#[derive(Debug, Clone)]
pub struct ClosedTradeSummary {
    pub net_pnl: f64,
}

/// Calculate win rate from closed trades.
/// Returns 1.0 (100%) when there are no closed trades to avoid NaN.
pub fn calculate_win_rate(trades: &[ClosedTradeSummary]) -> f64 {
    if trades.is_empty() {
        return 1.0;
    }
    let wins = trades.iter().filter(|t| t.net_pnl >= 0.0).count();
    wins as f64 / trades.len() as f64
}

pub async fn run(env: &Env) -> Result<()> {
    use chrono::Utc;
    console_log!("EOD summary starting...");

    let db = D1Client::new(env.d1("DB")?);
    let kv = KvClient::new(env.kv("KV")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    let today = Utc::now().format("%Y-%m-%d").to_string();

    // Get today's closed trades from D1
    let trades_today = db.get_trades_closed_on(&today).await?;

    let closed_summaries: Vec<ClosedTradeSummary> = trades_today
        .iter()
        .filter_map(|t| t.net_pnl.map(|pnl| ClosedTradeSummary { net_pnl: pnl }))
        .collect();

    let realized_pnl: f64 = closed_summaries.iter().map(|t| t.net_pnl).sum();
    let win_rate = calculate_win_rate(&closed_summaries);

    // Save IV snapshots for each underlying scanned today
    let alpaca = AlpacaClient::new(
        env.var("ALPACA_API_KEY")?.to_string(),
        env.var("ALPACA_SECRET_KEY")?.to_string(),
        env.var("ENVIRONMENT")?.to_string() == "paper",
    );

    for symbol in &["SPY", "QQQ", "IWM"] {
        match alpaca.get_options_chain(symbol).await {
            Ok(chain) => {
                let atm_iv = chain.contracts.iter()
                    .filter(|c| (c.strike - chain.underlying_price).abs() < chain.underlying_price * 0.02)
                    .filter_map(|c| c.implied_volatility)
                    .fold((0.0f64, 0usize), |(sum, n), iv| (sum + iv, n + 1));

                if atm_iv.1 > 0 {
                    let iv = atm_iv.0 / atm_iv.1 as f64;
                    db.upsert_iv_history(symbol, &today, iv).await.ok();
                    console_log!("{}: IV snapshot saved ({:.2}%)", symbol, iv * 100.0);
                }
            }
            Err(e) => console_log!("Could not fetch chain for {} IV snapshot: {:?}", symbol, e),
        }
    }

    // Update daily stats in KV
    let mut daily_stats = kv.get_daily_stats().await?;
    daily_stats.realized_pnl = realized_pnl;
    daily_stats.trades_closed = closed_summaries.len() as i64;
    kv.set_daily_stats(&daily_stats).await.ok();

    discord.send_eod_summary(&today, realized_pnl, closed_summaries.len() as i64, win_rate).await.ok();

    console_log!(
        "EOD summary complete: {} closed, P&L ${:.0}, win rate {:.0}%",
        closed_summaries.len(),
        realized_pnl,
        win_rate * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_rate_is_1_for_no_trades() {
        let rate = calculate_win_rate(&[]);
        assert!((rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn win_rate_is_50_pct_for_one_win_one_loss() {
        let trades = vec![
            ClosedTradeSummary { net_pnl: 50.0 },
            ClosedTradeSummary { net_pnl: -30.0 },
        ];
        let rate = calculate_win_rate(&trades);
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn win_rate_is_1_for_all_winners() {
        let trades = vec![
            ClosedTradeSummary { net_pnl: 50.0 },
            ClosedTradeSummary { net_pnl: 25.0 },
        ];
        assert!((calculate_win_rate(&trades) - 1.0).abs() < 1e-9);
    }
}
```

Note: `D1Client::get_trades_closed_on` needs to be added to `db/d1.rs` (add after the existing methods):

```rust
/// Get all trades closed on a specific date (YYYY-MM-DD).
pub async fn get_trades_closed_on(&self, date: &str) -> Result<Vec<Trade>> {
    let result = self
        .db
        .prepare(
            "SELECT * FROM trades WHERE status = 'closed'
            AND date(exit_time) = ?",
        )
        .bind(&[date.into()])?
        .all()
        .await?;
    let rows = result.results::<serde_json::Value>()?;
    Ok(rows.iter().filter_map(TradeRow::from_d1_row).collect())
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe -- handlers::eod_summary 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/handlers/eod_summary.rs trader-joe/src/db/d1.rs && \
git commit -m "feat(trader-joe): implement EOD summary handler with IV history and win rate"
```

---

### Task 19: Final Wiring — lib.rs and Full Cargo Build

**Group:** H (depends on Group G)

**Behavior being verified:** `cargo build --package trader-joe` succeeds without errors or warnings; all unit tests pass.

**Interface under test:** Full package build + test suite.

**Files:**
- Modify: `trader-joe/src/lib.rs` (replace stubs with full imports)

- [ ] **Step 1: Write the failing test**

```bash
# Verify full test suite passes before final wiring
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe 2>&1 | tail -10
```

Expected: Some tests may fail due to stub placeholders in lib.rs.

- [ ] **Step 2: Run test — verify which tests fail**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe 2>&1 | grep -E "FAILED|error"
```

Expected: Any compilation errors from incomplete wiring.

- [ ] **Step 3: Update lib.rs with final wiring**

```rust
use worker::*;

mod analysis;
mod broker;
mod config;
mod db;
mod handlers;
mod notifications;
mod risk;
mod types;

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Discord ping/pong for bot verification
    if req.method() == Method::Post {
        if let Ok(body) = req.text().await {
            if body.contains("\"type\":1") {
                return Response::from_json(&serde_json::json!({ "type": 1 }));
            }
        }
    }
    Response::ok("trader-joe")
}

#[event(scheduled)]
async fn scheduled(event: ScheduledEvent, env: Env, _ctx: ScheduleContext) {
    let cron = event.cron();
    console_log!("Scheduled event: {}", cron);

    let result = match cron.as_str() {
        "0 14 * * MON-FRI" => handlers::morning_scan::run(&env).await,
        "*/5 14-20 * * MON-FRI" => handlers::position_monitor::run(&env).await,
        "15 20 * * MON-FRI" => handlers::eod_summary::run(&env).await,
        unknown => {
            console_error!("Unknown cron expression: {}", unknown);
            Ok(())
        }
    };

    if let Err(e) = result {
        console_error!("Handler error for cron {}: {:?}", cron, e);
        // Best-effort Discord error notification
        if let (Ok(token), Ok(channel)) = (
            env.var("DISCORD_BOT_TOKEN"),
            env.var("DISCORD_CHANNEL_ID"),
        ) {
            let discord = notifications::discord::DiscordClient::new(token.to_string(), channel.to_string());
            discord
                .send_error(
                    &format!("Cron Handler Error: {}", cron),
                    &format!("{:?}", e),
                )
                .await
                .ok();
        }
    }
}
```

- [ ] **Step 4: Run full test suite — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package trader-joe 2>&1 | tail -20
```

Expected: All tests pass. Count: at minimum tasks 4–18 each contributed tests (3+5+5+5+2+5+5+5+2+3+2+3+3+4+3 = ~55 tests).

Also run backtest tests to confirm they still pass:

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
cargo test --package mahler-backtest 2>&1 | tail -5
```

Expected: 101 tests pass (per CLAUDE.md).

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
git add trader-joe/src/lib.rs && \
git commit -m "feat(trader-joe): wire cron router and fetch handler in lib.rs"
```

---

## Post-Build Verification

After all tasks complete, run the migration against the actual D1 database:

```bash
cd /Users/jdhiman/Documents/mahler/traderjoe && \
wrangler d1 execute mahler-db --file=src/migrations/0017_trader_joe_tables.sql
```

Verify the three new tables appear alongside the assistant's tables:

```bash
wrangler d1 execute mahler-db --command="SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
```

Expected output includes: `email_triage_log`, `iv_history`, `scan_log`, `trades`, `triage_state`

Deploy to Cloudflare:

```bash
wrangler deploy
```

Trigger a test run:

```bash
wrangler dev --test-scheduled
```

---

## Challenge Review

### CEO Pass

#### 1. Premise Challenge

**Right problem?** Yes. The Python Worker with 8,000 lines of ML complexity that autoresearch confirms doesn't beat fixed parameters is genuinely over-engineered. The Pyodide WASM runtime is memory-hungry and slow. Replacing it with a lean Rust Worker is the correct call.

**Real pain?** Concrete: the system is unreliable because dead code paths (agents, memory, three-perspective) add cognitive overhead with zero measured edge. Regime detector runs stale frozen weights. The pain is real.

**Direct path?** Yes. The plan directly implements the minimal system the autoresearch validates. No proxy problem being solved.

**Existing coverage?** `mahler-backtest/` already uses Rust with the same domain types. The workspace approach correctly extends an existing pattern.

#### 2. Scope Check

The only deferrable item is bear_call spreads — the backtest validates put spreads primarily, and bear calls add screener complexity with unvalidated edge. The spec resolves this as "include both, filter by score." Acceptable.

The hardest problem (worker-rs WASM compatibility with no async runtime for tests, D1 binding API correctness) is addressed via pure-function extraction. The plan does not avoid it.

#### 3. Twelve-Month Alignment

```
CURRENT STATE                     THIS PLAN                    12-MONTH IDEAL
8k-line Python, dead ML code  →   ~1.5k Rust, fixed params  →  Same Rust system
Unreliable, stale ML weights       Pure functions, testable     with better IV
Pyodide WASM, slow startup         WASM Rust, fast startup      data sources
Dead agents/memory code            Deleted entirely             and live VIX feed
```

This plan moves directly toward the ideal. The only tech debt created is the VIX proxy (VIXY ETF instead of spot VIX) — noted below.

#### 4. Alternatives Check

The spec discusses the Rust WASM approach and why it fits (same Pyodide runtime model, worker-rs bindings, shared workspace). The rationale is embedded in the spec's "Chosen Approach" section, which is sufficient.

---

### Engineering Pass

#### 5. Architecture

**Data flow:**
```
cron trigger -> lib.rs router -> handler::run(&env)
                                     |
                           KV: circuit breaker check
                                     |
                         Alpaca: market open / account / VIX
                                     |
                           CircuitBreaker::evaluate()
                                     |
                         Alpaca: get_options_chain(symbol)
                                     |
                        OptionsScreener::screen_chain()
                                     |
                      PositionSizer::calculate_for_underlying()
                                     |
                         Alpaca: place_spread_order()
                                     |
                        D1: create_trade() [ghost trade prevention]
                                     |
                        Discord: send_trade_placed()
```

**Security:** No user input flows to SQL or shell. D1 uses parameterized queries throughout. No injection vectors found.

**Scaling:** Scans 3 underlyings sequentially. Each fetches a 1000-contract options chain. No unbounded loops or fan-out patterns that break under load.

**Deployment:** If wrangler deploy succeeds but migration hasn't run, D1 queries fail loudly. Acceptable failure mode.

#### 6. Module Depth Audit

| Module | Exports | Hides | Verdict |
|--------|---------|-------|---------|
| analysis/greeks.rs | 3 fns | Hart(1968) CDF, BS formula, date parsing | DEEP |
| analysis/iv_rank.rs | 1 fn + IVMetrics | rank vs percentile distinction, clamping | DEEP |
| analysis/screener.rs | screen_chain() | DTE/liquidity/delta/credit filters, scoring | DEEP |
| risk/circuit_breaker.rs | evaluate() | 5 thresholds, size multiplier logic | DEEP |
| risk/position_sizer.rs | calculate(), calculate_for_underlying() | 3 limits, VIX reduction | DEEP |
| broker/alpaca.rs | 8 methods | Alpaca REST, OCC format, auth headers | DEEP |
| db/d1.rs | 7 methods | D1 bind/execute, JS-Rust conversion, deserialization | DEEP |
| db/kv.rs | 6 methods | KV serialization, TTL management, key construction | DEEP |
| notifications/discord.rs | 5 domain methods + send() | Discord embed JSON format, color codes | DEEP |
| types.rs | structs/enums + CreditSpread methods | none | SHALLOW (justified: pure data types) |
| config.rs | SpreadConfig + 2 methods | exit condition math, parameter constants | DEEP |

#### 7. Code Quality

**[BLOCKER] (confidence: 10/10) — Task 16: vix_data use-after-move (compile error)**

`vix_data: Option<VixData>` is an owned value. The plan calls `.map()` on it twice:

```
db.log_scan(..., vix_data.map(|v| v.vix), ...).await.ok();       // moves vix_data
discord.send_scan_summary(..., vix_data.map(|v| v.vix)).await.ok(); // ERROR: moved
```

`Option<VixData>` is not `Copy`. Fix: extract `let vix_value = vix_data.as_ref().map(|v| v.vix);` before both calls.

**[BLOCKER] (confidence: 10/10) — Task 17: position monitor marks trades closed without placing broker closing orders**

`position_monitor.rs` detects exits and calls `db.close_trade()` but never calls `alpaca.place_spread_order()`. The comment "Close the spread: buy back short, sell long" has no broker call following it. The actual options position at Alpaca remains open indefinitely while D1 reports it closed — orphaned positions accumulate.

Fix: before `db.close_trade()`, construct and place a closing `SpreadOrder` with `side: "buy_to_close"` / `side: "sell_to_close"`. Apply the same ghost-trade-prevention pattern: cancel if D1 write fails.

**[RISK] (confidence: 8/10) — Task 1 leaves ~200KB of dead Python files; spec requires deleting all**

The spec says: `traderjoe/src/ (Python tree) | Delete all Python handlers and core modules | Delete`

Task 1's deletion list omits:
- `src/handlers/morning_scan.py` (34.8K), `position_monitor.py` (53.5K), `eod_summary.py` (44.7K), `health.py`
- `src/entry.py`
- `src/core/analysis/greeks.py`, `iv_rank.py`, `screener.py`, `indicators.py`
- `src/core/broker/`, `src/core/db/`, `src/core/notifications/`
- `src/core/risk/circuit_breaker.py`, `src/core/risk/position_sizer.py`
- `src/core/backtesting/`, `src/core/monitoring/`, `src/core/inference/`, `src/core/learning/`, `src/core/reflection/`, `src/core/ai/`

Update Task 1 to delete `src/handlers/`, `src/core/`, and `src/entry.py` entirely, or add a step that does `rm -rf src/handlers/ src/core/ src/entry.py`.

**[RISK] (confidence: 8/10) — VIX source is VIXY (ETF proxy), not spot VIX**

`alpaca.rs` fetches VIXY price as a VIX proxy. VIXY systematically trades below spot VIX due to futures contango and roll costs. During flash crashes where circuit breaker matters most, VIXY < spot VIX. A halt threshold of 50 based on VIXY price may miss a genuine VIX=55 spike. Consider lowering vix_halt to 40 or documenting this limitation explicitly.

**[RISK] (confidence: 8/10) — Worker-rs 0.4 API compatibility unverified**

Three specific usages need verification before execution:
1. `i64.into()` → `JsValue` — JS numbers are f64; i64 may require explicit cast `(val as f64).into()`
2. `result.results::<serde_json::Value>()` — verify generic D1 results API exists in 0.4
3. `worker::js_sys::Math::random()` — verify js_sys is re-exported in worker-rs 0.4

Mitigation: add `cargo build --package trader-joe` to Task 3's verification step.

**[RISK] (confidence: 7/10) — build_occ_symbol panics on malformed expiration strings**

```rust
let parts: Vec<&str> = expiration.split('-').collect();
let yy = &parts[0][2..];  // panics if parts[0] < 2 chars
let mm = parts[1];         // panics if fewer than 2 segments
let dd = parts[2];         // panics if fewer than 3 segments
```

Only valid inputs are tested. Add a guard: if `parts.len() != 3 || parts[0].len() < 4`, return a `Result::Err` or a sentinel string rather than panicking.

**[RISK] (confidence: 8/10) — iv_history fetch uses unwrap_or_default() violating CLAUDE.md**

```rust
let history = db.get_iv_history(symbol, 252).await.unwrap_or_default();
```

CLAUDE.md: "User prefers to use explicit exception handling rather than fallback mechanisms." This silently uses neutral IV metrics when D1 is unavailable. Should propagate the error or send a Discord alert.

**[RISK] (confidence: 7/10) — get_spread_debit() has no test and silently skips without Discord alert**

This is the critical function determining exit P&L. If it returns None, positions are silently skipped with only a console_log. A position that consistently fails price lookup will never be closed. Should Discord-alert when a trade's price is unavailable.

**[OBS] — mut new_state in morning_scan.rs is unnecessary (Rust warning)**

**[OBS] — contracts.max(1) in position_sizer.rs is unreachable dead code**

The `if contracts <= 0 { return SizeResult::zero(...) }` check before it catches all zero cases.

#### 8. Test Philosophy Audit

All tests for pure functions test behavior through public interfaces with no internal mocking. Handler tests correctly extract pure functions for unit testing (async handlers require CF Worker runtime, which is correct to bypass).

**[RISK] (confidence: 7/10) — Screener test higher_iv_percentile_scores_higher has conditional assertion**

```rust
if !low_results.is_empty() && !high_results.is_empty() {
    assert!(high_results[0].score > low_results[0].score, ...);
}
```

If both result sets are empty, the test passes vacuously. Move the assertion outside the `if` block, or assert `!results.is_empty()` first.

#### 9. Vertical Slice Audit

All 19 tasks follow one-test -> one-implementation -> one-commit. No horizontal slicing detected.

**[OBS] — Task 18 modifies two files in one commit (eod_summary.rs + d1.rs)**

`get_trades_closed_on` should have been in Task 12. Both files are required for the same behavior, so the bundled commit is acceptable. Minor spec-coverage gap.

#### 10. Test Coverage Gaps

```
handlers/position_monitor.rs
  check_exit_conditions(): [TESTED] all 4 branches
  get_spread_debit():       [GAP] no test — critical path for exit P&L

broker/alpaca.rs
  build_occ_symbol():  [TESTED] 3 valid cases / [GAP] malformed expiration panic
  get_options_chain(): [GAP] no test (requires HTTP — acceptable)

analysis/screener.rs
  screen_chain():
    delta out of range:      [TESTED]
    valid spread found:      [TESTED]
    IV percentile scoring:   [TESTED with conditional assertion - weak]
    credit < 10% of width:   [GAP]
    no expirations in range: [GAP]
```

#### 11. Failure Modes

**BLOCKER (repeated):** Position monitor marks trades closed in D1 without placing broker closing orders.

**[RISK] — Ghost trade protection only in morning_scan, not position_monitor**

When closing order placement is added to position_monitor (per BLOCKER fix), apply the same pattern: if `db.close_trade()` fails after the closing order is placed, alert Discord and log for manual intervention.

**[RISK] — daily_stats.starting_equity never written to KV**

Morning scan computes `starting_daily` locally but never saves it to KV. `DailyStats.starting_equity` is always 0 in storage. The circuit breaker baseline is computed once per morning scan and lost when the handler exits. Functionally acceptable (position_monitor doesn't run CB evaluation), but the field is effectively dead storage.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|-----------|---------|--------|
| `worker = "0.4"` has stable D1/KV API matching plan code | VALIDATE | Worker-rs releases rapidly; verify before Task 3 |
| `i64.into()` -> JsValue compiles in wasm_bindgen | VALIDATE | JS numbers are f64; i64 may need explicit cast |
| `worker::js_sys::Math::random()` accessible without extra features | VALIDATE | Depends on js_sys re-export in worker-rs 0.4 |
| VIXY ETF price is sufficient proxy for spot VIX circuit breaker | RISKY | VIXY systematically trades below spot VIX via futures discount |
| Alpaca `order_class: "mleg"` is correct format for credit spread orders | VALIDATE | Verify against Alpaca options API docs before Task 15 |
| mahler-backtest workspace member compiles unchanged after adding workspace Cargo.toml | SAFE | Workspace root only adds parent; existing crate untouched |
| Task 1 deletion list covers all files spec requires removed | RISKY | Confirmed: ~200KB of Python files remain after Task 1 |
| chrono wasmbind feature correctly enables Utc::now() in WASM | VALIDATE | Listed in Cargo.toml; confirm it works with worker-rs 0.4 |
| D1 binding name "DB" and KV binding name "KV" match wrangler.toml | SAFE | Confirmed: wrangler.toml plan uses exactly "DB" and "KV" |

---

### Summary

**[BLOCKER] count: 2**
1. `vix_data` use-after-move in `morning_scan.rs` — compile error (Task 16)
2. `position_monitor.rs` marks trades closed in D1 without placing broker closing orders (Task 17)

**[RISK] count: 8**
1. Worker-rs 0.4 API compatibility — verify before Task 3
2. VIXY ETF is not spot VIX — circuit breaker halt threshold will be inaccurate during actual spikes
3. Task 1 leaves ~200KB of dead Python files that spec requires deleted
4. `build_occ_symbol` panics on malformed expiration strings
5. `iv_history` unwrap_or_default() violates CLAUDE.md explicit-exception preference
6. `get_spread_debit()` silently skips positions without Discord alert — no test, no alerting
7. Screener test `higher_iv_percentile_scores_higher` has conditional assertion that can pass vacuously
8. `daily_stats.starting_equity` never written to KV — dead storage

**[QUESTION] count: 2**
1. Should Task 1 delete the entire `src/` Python tree (per spec) rather than a subset?
2. Should Alpaca multi-leg options order format be verified against Alpaca API docs before Task 15?

VERDICT: NEEDS_REWORK — [BLOCKER 1: fix vix_data use-after-move in morning_scan.rs; BLOCKER 2: add broker closing order placement to position_monitor before db.close_trade()]
