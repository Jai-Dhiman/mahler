-- Mahler Initial Schema
-- Run with: wrangler d1 execute mahler --file=src/migrations/0001_initial.sql

-- Recommendations awaiting approval
CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'executed')),

    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL CHECK (spread_type IN ('bull_put', 'bear_call')),
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,

    credit REAL NOT NULL,
    max_loss REAL NOT NULL,

    iv_rank REAL,
    delta REAL,
    theta REAL,

    thesis TEXT,
    confidence TEXT CHECK (confidence IN ('low', 'medium', 'high')),
    suggested_contracts INTEGER,

    analysis_price REAL,
    discord_message_id TEXT
);

-- Executed trades
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    recommendation_id TEXT REFERENCES recommendations(id),

    opened_at TEXT,
    closed_at TEXT,
    status TEXT CHECK (status IN ('open', 'closed')),

    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL CHECK (spread_type IN ('bull_put', 'bear_call')),
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,

    entry_credit REAL NOT NULL,
    exit_debit REAL,
    profit_loss REAL,

    contracts INTEGER DEFAULT 1,
    broker_order_id TEXT,

    reflection TEXT,
    lesson TEXT
);

-- Position snapshots (current state of open trades)
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL REFERENCES trades(id),

    underlying TEXT NOT NULL,
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,
    contracts INTEGER NOT NULL,

    current_value REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,

    updated_at TEXT DEFAULT (datetime('now'))
);

-- Daily performance tracking
CREATE TABLE IF NOT EXISTS daily_performance (
    date TEXT PRIMARY KEY,
    starting_balance REAL NOT NULL,
    ending_balance REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    trades_opened INTEGER DEFAULT 0,
    trades_closed INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0
);

-- Playbook rules (both initial and learned)
CREATE TABLE IF NOT EXISTS playbook (
    id TEXT PRIMARY KEY,
    rule TEXT NOT NULL,
    source TEXT DEFAULT 'initial' CHECK (source IN ('initial', 'learned')),
    supporting_trade_ids TEXT,  -- JSON array of trade IDs
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status);
CREATE INDEX IF NOT EXISTS idx_recommendations_created ON recommendations(created_at);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying);
CREATE INDEX IF NOT EXISTS idx_positions_trade_id ON positions(trade_id);
CREATE INDEX IF NOT EXISTS idx_positions_underlying ON positions(underlying);
CREATE INDEX IF NOT EXISTS idx_daily_performance_date ON daily_performance(date);

-- Insert initial playbook rules based on PRD strategy
INSERT OR IGNORE INTO playbook (id, rule, source) VALUES
    ('rule_001', 'Only trade SPY, QQQ, IWM - high liquidity ETFs', 'initial'),
    ('rule_002', 'Target 30-45 DTE for optimal theta decay', 'initial'),
    ('rule_003', 'Short strike delta between 0.05-0.15 (85-95% OTM probability) - validated by 19-year backtest', 'initial'),
    ('rule_004', 'Trade in all IV environments - IV filter removed based on backtest showing +59% CAGR improvement', 'initial'),
    ('rule_005', 'Take profit at 65% of maximum credit - marginal improvement with slippage', 'initial'),
    ('rule_006', 'Stop loss at 200% of credit received', 'initial'),
    ('rule_007', 'Close all positions at 21 DTE regardless of P/L', 'initial'),
    ('rule_008', 'Maximum 2% account risk per trade', 'initial'),
    ('rule_009', 'Maximum 10% total portfolio heat', 'initial'),
    ('rule_010', 'Halt new trades when VIX > 50', 'initial');
