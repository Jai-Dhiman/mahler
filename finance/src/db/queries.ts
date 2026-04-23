import type { AccountRow, Env, PlaidItemRow, SnapshotRow } from "../types";

export async function upsertAccount(env: Env, row: Omit<AccountRow, "created_at">): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO finance_account
       (account_id, item_id, display_name, account_type, asset_class, currency,
        is_liability, include_in_networth, is_active)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(account_id) DO UPDATE SET
       item_id = excluded.item_id,
       display_name = excluded.display_name,
       account_type = excluded.account_type,
       asset_class = excluded.asset_class,
       currency = excluded.currency,
       is_liability = excluded.is_liability,
       include_in_networth = excluded.include_in_networth,
       is_active = excluded.is_active`,
  )
    .bind(
      row.account_id,
      row.item_id,
      row.display_name,
      row.account_type,
      row.asset_class,
      row.currency,
      row.is_liability,
      row.include_in_networth,
      row.is_active,
    )
    .run();
}

export interface SnapshotInput {
  account_id: string;
  taken_at: string;
  snapshot_date: string;
  current_balance: number;
  available_balance: number | null;
  source: "plaid" | "alpaca" | "manual";
  raw_response: string | null;
}

export async function insertSnapshot(env: Env, row: SnapshotInput): Promise<void> {
  await env.DB.prepare(
    `INSERT OR IGNORE INTO finance_balance_snapshot
       (account_id, taken_at, snapshot_date, current_balance, available_balance, source, raw_response)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      row.account_id,
      row.taken_at,
      row.snapshot_date,
      row.current_balance,
      row.available_balance,
      row.source,
      row.raw_response,
    )
    .run();
}

export async function getLatestSnapshots(env: Env): Promise<SnapshotRow[]> {
  const result = await env.DB.prepare(
    `SELECT s.* FROM finance_balance_snapshot s
       JOIN (
         SELECT account_id, MAX(taken_at) AS max_taken
         FROM finance_balance_snapshot
         GROUP BY account_id
       ) latest
         ON latest.account_id = s.account_id AND latest.max_taken = s.taken_at`,
  ).all<SnapshotRow>();
  return result.results;
}

export async function getHistory(env: Env, accountId: string, days: number): Promise<SnapshotRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_balance_snapshot
     WHERE account_id = ?
     ORDER BY snapshot_date DESC
     LIMIT ?`,
  )
    .bind(accountId, days)
    .all<SnapshotRow>();
  return result.results;
}

export interface UpsertItemInput {
  item_id: string;
  institution_name: string;
}

export async function upsertItem(env: Env, input: UpsertItemInput): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO finance_plaid_item (item_id, institution_name)
     VALUES (?, ?)
     ON CONFLICT(item_id) DO UPDATE SET
       institution_name = excluded.institution_name`,
  )
    .bind(input.item_id, input.institution_name)
    .run();
}

export async function listItems(env: Env): Promise<PlaidItemRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_plaid_item ORDER BY created_at ASC`,
  ).all<PlaidItemRow>();
  return result.results;
}

export async function listAccounts(env: Env): Promise<AccountRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_account WHERE is_active = 1 ORDER BY created_at ASC`,
  ).all<AccountRow>();
  return result.results;
}

export async function updateItemStatus(
  env: Env,
  itemId: string,
  status: "ok" | "needs_reauth" | "error",
  lastError: string | null = null,
): Promise<void> {
  await env.DB.prepare(
    `UPDATE finance_plaid_item
     SET status = ?, last_error = ?, last_synced_at = datetime('now')
     WHERE item_id = ?`,
  )
    .bind(status, lastError, itemId)
    .run();
}
