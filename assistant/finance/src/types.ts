export interface Env {
  DB: D1Database;
  FINANCE_KV: KVNamespace;
  ENVIRONMENT: string;
  LOG_LEVEL: string;
  BEARER_TOKEN: string;
  PLAID_CLIENT_ID: string;
  PLAID_SECRET_DEV: string;
  PLAID_WEBHOOK_SECRET: string;
  DISCORD_WEBHOOK_URL: string;
  ALPACA_PAPER_KEY_ID: string;
  ALPACA_PAPER_SECRET: string;
}

export interface PlaidItemRow {
  item_id: string;
  institution_name: string;
  status: "ok" | "needs_reauth" | "error";
  last_synced_at: string | null;
  last_error: string | null;
  created_at: string;
}

export interface AccountRow {
  account_id: string;
  item_id: string | null;
  display_name: string;
  account_type: string;
  asset_class: string;
  currency: string;
  is_liability: 0 | 1;
  include_in_networth: 0 | 1;
  is_active: 0 | 1;
  created_at: string;
}

export interface SnapshotRow {
  id: number;
  account_id: string;
  taken_at: string;
  snapshot_date: string;
  current_balance: number;
  available_balance: number | null;
  source: "plaid" | "alpaca" | "manual";
  raw_response: string | null;
}

export interface EventRow {
  id: number;
  occurred_at: string;
  event_type: string;
  item_id: string | null;
  account_id: string | null;
  payload: string | null;
}

export interface SyncResult {
  itemsAttempted: number;
  itemsSucceeded: number;
  snapshotsWritten: number;
  errors: Array<{ item_id: string; error: string }>;
}

export interface WeeklyAccountLine {
  account_id: string;
  display_name: string;
  account_type: string;
  current_balance: number;
  prior_week_balance: number | null;
  delta: number | null;
}

export interface WeeklyData {
  asOf: string;
  netWorth: number;
  priorWeekNetWorth: number | null;
  netWorthDelta: number | null;
  accounts: WeeklyAccountLine[];
  strategyPaperEquity: number | null;
  sparkline: Array<{ date: string; net_worth: number }>;
  staleItems: Array<{ item_id: string; institution_name: string; last_synced_at: string | null }>;
}
