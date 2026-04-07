-- FinMem Memory Enhancement Migration
-- Adds three-tier episodic memory with exponential decay scoring,
-- critical event detection, and memory consolidation support.
--
-- Reference: FinMem paper (https://arxiv.org/abs/2311.13743)

-- Add new columns to episodic_memory for FinMem features
ALTER TABLE episodic_memory ADD COLUMN memory_layer TEXT DEFAULT 'shallow';
ALTER TABLE episodic_memory ADD COLUMN access_count INTEGER DEFAULT 0;
ALTER TABLE episodic_memory ADD COLUMN last_accessed_at TEXT;
ALTER TABLE episodic_memory ADD COLUMN critical_event INTEGER DEFAULT 0;
ALTER TABLE episodic_memory ADD COLUMN critical_event_reason TEXT;
ALTER TABLE episodic_memory ADD COLUMN pnl_dollars REAL;
ALTER TABLE episodic_memory ADD COLUMN pnl_percent REAL;
ALTER TABLE episodic_memory ADD COLUMN promoted_at TEXT;
ALTER TABLE episodic_memory ADD COLUMN promoted_from TEXT;

-- Indexes for layer-based queries
CREATE INDEX IF NOT EXISTS idx_episodic_memory_layer ON episodic_memory(memory_layer);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_critical ON episodic_memory(critical_event);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_access_count ON episodic_memory(access_count);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_layer_critical ON episodic_memory(memory_layer, critical_event);
