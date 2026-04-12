-- TraderJoe initial schema
-- trades: one row per credit spread position
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL CHECK (spread_type IN ('bull_put', 'bear_call')),
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    entry_credit REAL NOT NULL,
    max_loss REAL NOT NULL,
    broker_order_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending_fill'
        CHECK (status IN ('pending_fill', 'open', 'closed', 'cancelled')),
    fill_price REAL,
    fill_time TEXT,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT CHECK (exit_reason IN ('profit_target', 'stop_loss', 'expiration', 'manual', NULL)),
    net_pnl REAL,
    iv_rank REAL,
    short_delta REAL,
    short_theta REAL
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying);

-- iv_history: daily ATM IV per underlying, used for IV rank/percentile calculation
CREATE TABLE IF NOT EXISTS iv_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    iv REAL NOT NULL,
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_iv_history_symbol_date ON iv_history(symbol, date);

-- scan_log: audit log for every cron invocation
CREATE TABLE IF NOT EXISTS scan_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_time TEXT NOT NULL,
    scan_type TEXT NOT NULL,
    underlyings_scanned INTEGER NOT NULL DEFAULT 0,
    opportunities_found INTEGER NOT NULL DEFAULT 0,
    trades_placed INTEGER NOT NULL DEFAULT 0,
    vix REAL,
    circuit_breaker_active INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);
