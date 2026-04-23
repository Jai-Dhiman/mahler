-- Measurement infrastructure for paper->live go-live gate.
-- Adds NBBO/Greeks capture to trades and four new tables for observability.

-- NBBO and expanded Greeks additions to trades.
ALTER TABLE trades ADD COLUMN entry_short_bid REAL;
ALTER TABLE trades ADD COLUMN entry_short_ask REAL;
ALTER TABLE trades ADD COLUMN entry_long_bid REAL;
ALTER TABLE trades ADD COLUMN entry_long_ask REAL;
ALTER TABLE trades ADD COLUMN entry_net_mid REAL;
ALTER TABLE trades ADD COLUMN exit_short_bid REAL;
ALTER TABLE trades ADD COLUMN exit_short_ask REAL;
ALTER TABLE trades ADD COLUMN exit_long_bid REAL;
ALTER TABLE trades ADD COLUMN exit_long_ask REAL;
ALTER TABLE trades ADD COLUMN exit_net_mid REAL;
ALTER TABLE trades ADD COLUMN entry_short_gamma REAL;
ALTER TABLE trades ADD COLUMN entry_short_vega REAL;
ALTER TABLE trades ADD COLUMN entry_long_delta REAL;
ALTER TABLE trades ADD COLUMN entry_long_gamma REAL;
ALTER TABLE trades ADD COLUMN entry_long_vega REAL;
ALTER TABLE trades ADD COLUMN nbbo_displayed_size_short INTEGER;
ALTER TABLE trades ADD COLUMN nbbo_displayed_size_long INTEGER;
ALTER TABLE trades ADD COLUMN nbbo_snapshot_time TEXT;

CREATE TABLE IF NOT EXISTS equity_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('eod','trade_open','trade_close','circuit_breaker')),
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    open_position_mtm REAL NOT NULL,
    realized_pnl_day REAL NOT NULL DEFAULT 0,
    unrealized_pnl_day REAL NOT NULL DEFAULT 0,
    open_position_count INTEGER NOT NULL DEFAULT 0,
    trade_id_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_history(timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_equity_eod_unique
    ON equity_history(date(timestamp)) WHERE event_type = 'eod';

CREATE TABLE IF NOT EXISTS portfolio_greeks_eod (
    date TEXT PRIMARY KEY,
    beta_weighted_delta REAL NOT NULL,
    total_gamma REAL NOT NULL,
    total_vega REAL NOT NULL,
    total_theta REAL NOT NULL,
    delta_by_underlying TEXT NOT NULL,
    max_gamma_single_position REAL NOT NULL,
    max_vega_single_position REAL NOT NULL,
    open_position_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS market_context_daily (
    date TEXT PRIMARY KEY,
    spot_vix REAL,
    spot_vix_source TEXT CHECK (spot_vix_source IN ('fred','stooq','unavailable')),
    spot_vix_source_date TEXT,
    vixy_close REAL,
    spy_20d_realized_vol REAL,
    spy_20d_return REAL,
    spy_drawdown_from_52w_high REAL
);

CREATE TABLE IF NOT EXISTS metrics_weekly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    trade_count INTEGER NOT NULL,
    sharpe REAL,
    sortino REAL,
    profit_factor REAL,
    win_rate REAL,
    pnl_skew REAL,
    max_drawdown_pct REAL,
    mean_slippage_vs_mid REAL,
    max_slippage_vs_mid REAL,
    slippage_vs_orats_ratio REAL,
    fill_size_violation_count INTEGER NOT NULL DEFAULT 0,
    fill_size_violation_pct REAL,
    regime_buckets TEXT,
    greek_ranges TEXT,
    sample_size_tag TEXT NOT NULL CHECK (sample_size_tag IN ('INSUFFICIENT','WEAK','OK'))
);
