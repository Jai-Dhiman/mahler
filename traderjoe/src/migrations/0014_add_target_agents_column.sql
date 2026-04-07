-- Add target_agents column to semantic_rules
-- Required for FINCON-style selective knowledge propagation
-- The code expects this column but it was missing from the original migration

ALTER TABLE semantic_rules ADD COLUMN target_agents TEXT DEFAULT 'all';

-- Create index for target_agents queries
CREATE INDEX IF NOT EXISTS idx_semantic_rules_target_agents ON semantic_rules(target_agents);
