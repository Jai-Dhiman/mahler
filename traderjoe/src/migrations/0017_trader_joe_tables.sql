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
