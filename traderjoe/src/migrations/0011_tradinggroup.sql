-- TradingGroup Paper Implementation (arXiv:2508.17565)
-- Adds support for:
-- 1. Chain-of-Thought traces storage
-- 2. Hit-score for forecasting accuracy
-- 3. Per-agent reflections table

-- Add raw_cot_traces column to trade_trajectories
-- Stores mapping of agent_id -> raw thinking content from extended thinking
ALTER TABLE trade_trajectories ADD COLUMN raw_cot_traces TEXT;

-- Add hit_score column to trade_trajectories
-- Formula: hit_score = sign_ok * tanh(|pct| / epsilon) * confidence
ALTER TABLE trade_trajectories ADD COLUMN hit_score REAL;

-- Per-agent reflections table
-- Stores reflections from ForecastingReflector, StyleReflector, DecisionReflector
CREATE TABLE IF NOT EXISTS agent_reflections (
    id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,  -- 'forecasting', 'style', 'decision'
    generated_at TEXT NOT NULL,
    lookback_days INTEGER,
    trade_count INTEGER,
    success_patterns TEXT,  -- JSON array of pattern objects
    failure_patterns TEXT,  -- JSON array of pattern objects
    key_insights TEXT,      -- JSON array of insight strings
    recommended_adjustments TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Index for querying reflections by agent type
CREATE INDEX IF NOT EXISTS idx_agent_reflections_type ON agent_reflections(agent_type);

-- Index for ordering by generation time
CREATE INDEX IF NOT EXISTS idx_agent_reflections_generated ON agent_reflections(generated_at DESC);

-- Index for hit_score analysis on trajectories
CREATE INDEX IF NOT EXISTS idx_trajectories_hit_score ON trade_trajectories(hit_score)
    WHERE hit_score IS NOT NULL;
