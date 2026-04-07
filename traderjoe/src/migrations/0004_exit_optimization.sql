-- Migration: Add exit analytics columns to trades table
-- Purpose: Track exit reason, IV rank at exit, and DTE at exit for Phase 1 Exit Optimization

-- Add exit analytics columns to trades table
ALTER TABLE trades ADD COLUMN exit_reason TEXT;
ALTER TABLE trades ADD COLUMN iv_rank_at_exit REAL;
ALTER TABLE trades ADD COLUMN dte_at_exit INTEGER;
