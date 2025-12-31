-- Migration: Add exit order tracking to trades table
-- Purpose: Track exit order IDs to enable reconciliation if D1 update fails after order placement

-- Add exit_order_id column to trades table
ALTER TABLE trades ADD COLUMN exit_order_id TEXT;

-- Index for finding trades with pending exit orders
CREATE INDEX IF NOT EXISTS idx_trades_exit_order_id ON trades(exit_order_id) WHERE exit_order_id IS NOT NULL;
