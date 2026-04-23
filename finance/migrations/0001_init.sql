CREATE TABLE IF NOT EXISTS finance_plaid_item (
  item_id           TEXT PRIMARY KEY,
  institution_name  TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'ok',
  last_synced_at    TEXT,
  last_error        TEXT,
  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS finance_account (
  account_id          TEXT PRIMARY KEY,
  item_id             TEXT,
  display_name        TEXT NOT NULL,
  account_type        TEXT NOT NULL,
  asset_class         TEXT NOT NULL,
  currency            TEXT NOT NULL DEFAULT 'USD',
  is_liability        INTEGER NOT NULL DEFAULT 0,
  include_in_networth INTEGER NOT NULL DEFAULT 1,
  is_active           INTEGER NOT NULL DEFAULT 1,
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS finance_balance_snapshot (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id        TEXT NOT NULL,
  taken_at          TEXT NOT NULL,
  snapshot_date     TEXT NOT NULL,
  current_balance   REAL NOT NULL,
  available_balance REAL,
  source            TEXT NOT NULL,
  raw_response      TEXT,
  UNIQUE (account_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_account_time
  ON finance_balance_snapshot (account_id, taken_at DESC);

CREATE TABLE IF NOT EXISTS finance_event_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  occurred_at TEXT NOT NULL DEFAULT (datetime('now')),
  event_type  TEXT NOT NULL,
  item_id     TEXT,
  account_id  TEXT,
  payload     TEXT
);
