-- Screening Results Table
-- Tracks every scan and what opportunities were found/filtered/approved
-- Run with: wrangler d1 execute mahler-db --remote --file=src/migrations/0015_screening_results.sql

CREATE TABLE IF NOT EXISTS screening_results (
    id TEXT PRIMARY KEY,
    scan_date TEXT NOT NULL,
    scan_time TEXT NOT NULL,  -- 'morning', 'midday', 'afternoon'
    scan_timestamp TEXT NOT NULL,  -- ISO timestamp

    -- Opportunity counts
    total_underlyings_scanned INTEGER NOT NULL DEFAULT 0,
    opportunities_found INTEGER NOT NULL DEFAULT 0,
    opportunities_passed_filters INTEGER NOT NULL DEFAULT 0,
    opportunities_sent_to_agents INTEGER NOT NULL DEFAULT 0,
    opportunities_approved INTEGER NOT NULL DEFAULT 0,

    -- Skip reason breakdown (JSON)
    skip_reasons TEXT,  -- {"iv_too_low": 3, "circuit_breaker": 1, "agent_rejected": 2, ...}

    -- Per-underlying results (JSON)
    underlying_results TEXT,  -- {"SPY": {"found": 5, "passed": 2, "reason": "..."}, ...}

    -- Market context (JSON)
    market_context TEXT,  -- {"vix": 25.3, "iv_percentile": {"SPY": 45, ...}, "regime": "bull_low_vol", ...}

    -- Circuit breaker state
    circuit_breaker_active INTEGER NOT NULL DEFAULT 0,
    circuit_breaker_reason TEXT,

    -- Size multipliers
    risk_multiplier REAL DEFAULT 1.0,
    regime_multiplier REAL DEFAULT 1.0,
    combined_multiplier REAL DEFAULT 1.0,

    created_at TEXT DEFAULT (datetime('now'))
);

-- Index for efficient date lookups
CREATE INDEX IF NOT EXISTS idx_screening_results_date ON screening_results(scan_date DESC);
CREATE INDEX IF NOT EXISTS idx_screening_results_timestamp ON screening_results(scan_timestamp DESC);
