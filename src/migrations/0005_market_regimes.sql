-- Migration: Add market_regimes table for regime detection history
-- Purpose: Store regime history for analysis and backtesting
-- Phase: 2 (Regime Detection)

CREATE TABLE IF NOT EXISTS market_regimes (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,  -- bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
    probability REAL NOT NULL,
    position_multiplier REAL NOT NULL,
    features TEXT,  -- JSON blob of feature values
    detected_at TEXT NOT NULL
);

-- Index for querying regime history by symbol and time
CREATE INDEX IF NOT EXISTS idx_market_regimes_symbol_detected
    ON market_regimes(symbol, detected_at);

-- Index for filtering by regime type
CREATE INDEX IF NOT EXISTS idx_market_regimes_regime
    ON market_regimes(regime);
