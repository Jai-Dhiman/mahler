-- Phase 3: Entry Scoring & Dynamic Betas
-- Migration for dynamic_betas and optimized_weights tables

-- Table for storing dynamically calculated betas
CREATE TABLE IF NOT EXISTS dynamic_betas (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    beta_ewma REAL NOT NULL,
    beta_rolling_20 REAL,
    beta_rolling_60 REAL,
    beta_blended REAL NOT NULL,
    correlation_spy REAL,
    data_days INTEGER NOT NULL,
    calculated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dynamic_betas_symbol_date
    ON dynamic_betas(symbol, calculated_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dynamic_betas_symbol_date_unique
    ON dynamic_betas(symbol, DATE(calculated_at));

-- Table for storing optimized scoring weights by regime
CREATE TABLE IF NOT EXISTS optimized_weights (
    id TEXT PRIMARY KEY,
    regime TEXT NOT NULL,
    weight_iv REAL NOT NULL,
    weight_delta REAL NOT NULL,
    weight_credit REAL NOT NULL,
    weight_ev REAL NOT NULL,
    sharpe_ratio REAL,
    n_trades INTEGER NOT NULL,
    optimized_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_optimized_weights_regime
    ON optimized_weights(regime, optimized_at DESC);
