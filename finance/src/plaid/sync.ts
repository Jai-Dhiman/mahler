import {
  insertSnapshot,
  listItems,
  logEvent,
  updateItemStatus,
  upsertAccount,
} from "../db/queries";
import type { Env, SyncResult } from "../types";
import { getBalances, type PlaidAccount } from "./client";

function classifyAssetClass(account: PlaidAccount): { asset_class: string; is_liability: 0 | 1 } {
  if (account.type === "credit" || account.type === "loan") {
    return { asset_class: "liability", is_liability: 1 };
  }
  if (account.type === "investment" || account.type === "brokerage") {
    return { asset_class: "mixed", is_liability: 0 };
  }
  return { asset_class: "cash", is_liability: 0 };
}

export async function syncAllItems(env: Env, today: string): Promise<SyncResult> {
  const items = await listItems(env);
  const result: SyncResult = {
    itemsAttempted: items.length,
    itemsSucceeded: 0,
    snapshotsWritten: 0,
    errors: [],
  };

  for (const item of items) {
    try {
      const balances = await getBalances(env, item.item_id);
      const takenAt = `${today}T07:00:00Z`;
      for (const account of balances.accounts) {
        const cls = classifyAssetClass(account);
        await upsertAccount(env, {
          account_id: account.account_id,
          item_id: item.item_id,
          display_name: account.official_name ?? account.name,
          account_type: account.subtype ?? account.type,
          asset_class: cls.asset_class,
          currency: account.balances.iso_currency_code ?? "USD",
          is_liability: cls.is_liability,
          include_in_networth: 1,
          is_active: 1,
        });
        await insertSnapshot(env, {
          account_id: account.account_id,
          taken_at: takenAt,
          snapshot_date: today,
          current_balance: account.balances.current,
          available_balance: account.balances.available,
          source: "plaid",
          raw_response: JSON.stringify(account),
        });
        result.snapshotsWritten += 1;
      }
      await updateItemStatus(env, item.item_id, "ok", null);
      result.itemsSucceeded += 1;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await updateItemStatus(env, item.item_id, "error", message.slice(0, 500));
      await logEvent(env, {
        event_type: "item_error",
        item_id: item.item_id,
        account_id: null,
        payload: { error: message },
      });
      result.errors.push({ item_id: item.item_id, error: message });
    }
  }

  await logEvent(env, {
    event_type: "snapshot_run",
    item_id: null,
    account_id: null,
    payload: result,
  });

  return result;
}
