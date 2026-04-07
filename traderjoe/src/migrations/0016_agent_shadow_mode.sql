-- Add agent shadow mode columns to trades table.
-- Agent decisions are recorded but do not gate trade execution.
-- This enables comparing algorithmic vs agent-filtered performance.

ALTER TABLE trades ADD COLUMN agent_decision TEXT;       -- 'approve', 'modify', 'reject', NULL
ALTER TABLE trades ADD COLUMN agent_contracts INTEGER;   -- contracts agent would have set
ALTER TABLE trades ADD COLUMN agent_confidence REAL;     -- agent confidence 0.0-1.0
ALTER TABLE trades ADD COLUMN agent_thesis TEXT;         -- agent's reasoning summary
