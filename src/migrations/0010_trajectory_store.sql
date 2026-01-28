-- V2 Learning Layer: Trajectory Store for Auto-Labeling
-- Stores complete trade trajectories for learning from outcomes

-- Trade Trajectories: Complete decision context and outcomes
CREATE TABLE IF NOT EXISTS trade_trajectories (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,

    -- Trade details
    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL,
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,
    entry_credit REAL NOT NULL,
    contracts INTEGER NOT NULL,

    -- Market context at entry
    market_regime TEXT,
    iv_rank REAL,
    vix_at_entry REAL,

    -- Agent outputs (JSON)
    analyst_outputs TEXT,        -- JSON array of analyst messages
    debate_transcript TEXT,      -- JSON array of debate messages
    synthesis_output TEXT,       -- JSON with synthesis result
    decision_output TEXT,        -- JSON with decision result
    three_perspective_result TEXT,  -- JSON with three-perspective assessment

    -- Outcome (filled after trade closes)
    actual_pnl REAL,
    actual_pnl_percent REAL,
    exit_reason TEXT,
    days_held INTEGER,

    -- Labels (filled by data synthesis)
    reward_label TEXT,           -- strong_positive, positive, negative, strong_negative
    reward_score REAL,

    -- Linked trade ID
    trade_id TEXT REFERENCES trades(id)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_trajectories_underlying ON trade_trajectories(underlying);
CREATE INDEX IF NOT EXISTS idx_trajectories_market_regime ON trade_trajectories(market_regime);
CREATE INDEX IF NOT EXISTS idx_trajectories_reward_label ON trade_trajectories(reward_label);
CREATE INDEX IF NOT EXISTS idx_trajectories_trade_id ON trade_trajectories(trade_id);

-- Index for unlabeled trajectories with outcomes (used by data synthesis)
CREATE INDEX IF NOT EXISTS idx_trajectories_unlabeled_with_outcome
    ON trade_trajectories(reward_label, actual_pnl)
    WHERE reward_label IS NULL AND actual_pnl IS NOT NULL;

-- Index for created_at ordering
CREATE INDEX IF NOT EXISTS idx_trajectories_created_at ON trade_trajectories(created_at);
