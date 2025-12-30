-- Phase 4: Rule Validation Schema
-- Tracks statistical validation of playbook rules

-- Table for storing rule validation results
CREATE TABLE IF NOT EXISTS rule_validations (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    validated_at TEXT NOT NULL,

    -- Sample sizes
    trades_with_rule INTEGER NOT NULL,
    trades_without_rule INTEGER NOT NULL,

    -- Outcome statistics
    mean_pnl_with REAL NOT NULL,
    mean_pnl_without REAL NOT NULL,
    win_rate_with REAL NOT NULL,
    win_rate_without REAL NOT NULL,

    -- Test results
    u_statistic REAL NOT NULL,
    p_value REAL NOT NULL,
    p_value_adjusted REAL NOT NULL,
    is_significant INTEGER NOT NULL,
    effect_direction TEXT NOT NULL CHECK (effect_direction IN ('positive', 'negative', 'neutral')),

    FOREIGN KEY (rule_id) REFERENCES playbook(id) ON DELETE CASCADE
);

-- Index for querying validation history by rule
CREATE INDEX IF NOT EXISTS idx_rule_validations_rule
    ON rule_validations(rule_id, validated_at DESC);

-- Index for finding significant validations
CREATE INDEX IF NOT EXISTS idx_rule_validations_significant
    ON rule_validations(is_significant, validated_at DESC);

-- Add validation status columns to playbook table
ALTER TABLE playbook ADD COLUMN is_validated INTEGER DEFAULT 0;
ALTER TABLE playbook ADD COLUMN last_validated_at TEXT;
ALTER TABLE playbook ADD COLUMN validation_p_value REAL;

-- Add applied_rule_ids column to trades table for rule tagging
ALTER TABLE trades ADD COLUMN applied_rule_ids TEXT;
