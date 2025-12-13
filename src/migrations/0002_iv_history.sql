-- IV History Schema
-- Run with: wrangler d1 execute mahler-db --file=src/migrations/0002_iv_history.sql

-- Daily IV observations for each underlying
-- Used to calculate IV Rank and IV Percentile with real historical data
CREATE TABLE IF NOT EXISTS iv_history (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    underlying TEXT NOT NULL,
    atm_iv REAL NOT NULL,
    underlying_price REAL,
    created_at TEXT DEFAULT (datetime('now')),

    UNIQUE(date, underlying)
);

-- VIX daily observations for position sizing and circuit breaker
CREATE TABLE IF NOT EXISTS vix_history (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL UNIQUE,
    vix_close REAL NOT NULL,
    vix3m_close REAL,  -- For term structure analysis
    term_structure_ratio REAL,  -- VIX/VIX3M, >1 = backwardation
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_iv_history_underlying_date ON iv_history(underlying, date DESC);
CREATE INDEX IF NOT EXISTS idx_iv_history_date ON iv_history(date DESC);
CREATE INDEX IF NOT EXISTS idx_vix_history_date ON vix_history(date DESC);
