-- V2 Memory Layer: Episodic and Semantic Memory Tables
-- Migration for multi-agent system memory storage

-- Episodic Memory: Stores complete agent context for each trade
-- Used for similar trade retrieval and learning from past decisions
CREATE TABLE IF NOT EXISTS episodic_memory (
    id TEXT PRIMARY KEY,
    trade_id TEXT REFERENCES trades(id),
    entry_date TEXT NOT NULL,
    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL,
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,

    -- Agent outputs (JSON)
    analyst_outputs TEXT,      -- JSON array of analyst messages
    debate_transcript TEXT,    -- JSON array of debate messages
    debate_outcome TEXT,       -- JSON with facilitator synthesis

    -- Prediction vs Reality
    predicted_outcome TEXT,    -- JSON with expected outcome
    actual_outcome TEXT,       -- JSON with actual outcome (filled after close)

    -- Learning
    reflection TEXT,           -- AI-generated reflection
    lesson_extracted TEXT,     -- Extracted rule/lesson
    embedding_id TEXT,         -- Reference to Vectorize embedding

    -- Metadata
    market_regime TEXT,
    iv_rank REAL,
    vix_at_entry REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Semantic Rules: Validated trading rules extracted from episodic memory
-- Rules must pass statistical validation before being used
CREATE TABLE IF NOT EXISTS semantic_rules (
    id TEXT PRIMARY KEY,
    rule_text TEXT NOT NULL,
    rule_type TEXT NOT NULL,   -- 'entry', 'exit', 'sizing', 'regime'

    -- Source and validation
    source TEXT NOT NULL,      -- 'initial', 'learned', 'validated'
    applies_to_agent TEXT,     -- 'all' or specific agent_id
    supporting_trades INTEGER DEFAULT 0,
    opposing_trades INTEGER DEFAULT 0,

    -- Statistical validation
    p_value REAL,              -- Statistical significance
    effect_size REAL,          -- Practical significance
    confidence_interval TEXT,  -- JSON with [lower, upper]

    -- Conditions for rule application
    conditions TEXT,           -- JSON with conditions (regime, IV range, etc.)

    -- Status
    is_active INTEGER DEFAULT 1,
    last_validated TEXT,
    validation_count INTEGER DEFAULT 0,

    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Agent Messages: Stores individual agent outputs for analysis
CREATE TABLE IF NOT EXISTS agent_messages (
    id TEXT PRIMARY KEY,
    episodic_memory_id TEXT REFERENCES episodic_memory(id),
    agent_id TEXT NOT NULL,
    message_type TEXT NOT NULL,  -- 'analysis', 'argument', 'synthesis', 'decision'
    content TEXT NOT NULL,
    structured_data TEXT,        -- JSON
    confidence REAL,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Debate Rounds: Tracks debate progression for analysis
CREATE TABLE IF NOT EXISTS debate_rounds (
    id TEXT PRIMARY KEY,
    episodic_memory_id TEXT REFERENCES episodic_memory(id),
    round_number INTEGER NOT NULL,
    bull_message_id TEXT REFERENCES agent_messages(id),
    bear_message_id TEXT REFERENCES agent_messages(id),
    consensus_reached INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_episodic_memory_trade_id ON episodic_memory(trade_id);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_underlying ON episodic_memory(underlying);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_entry_date ON episodic_memory(entry_date);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_market_regime ON episodic_memory(market_regime);

CREATE INDEX IF NOT EXISTS idx_semantic_rules_source ON semantic_rules(source);
CREATE INDEX IF NOT EXISTS idx_semantic_rules_rule_type ON semantic_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_semantic_rules_is_active ON semantic_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_semantic_rules_applies_to ON semantic_rules(applies_to_agent);

CREATE INDEX IF NOT EXISTS idx_agent_messages_episodic ON agent_messages(episodic_memory_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_agent ON agent_messages(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_type ON agent_messages(message_type);

CREATE INDEX IF NOT EXISTS idx_debate_rounds_episodic ON debate_rounds(episodic_memory_id);
