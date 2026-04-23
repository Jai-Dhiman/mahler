import { getPaperEquity } from "../alpaca/client";
import { insertSnapshot, listAccounts, logEvent, upsertAccount } from "../db/queries";
import { syncAllItems } from "../plaid/sync";
import { computeWeeklySummary } from "../summary/compute";
import { postWeeklySummary } from "../discord/embed";
import type { Env } from "../types";

const PAPER_ACCOUNT_ID = "acc_paper";

function isoDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

async function snapshotAlpaca(env: Env, today: string): Promise<void> {
  let equity: number;
  try {
    equity = await getPaperEquity(env);
  } catch (err) {
    await logEvent(env, {
      event_type: "alpaca_error",
      item_id: null,
      account_id: PAPER_ACCOUNT_ID,
      payload: { error: err instanceof Error ? err.message : String(err) },
    });
    return;
  }
  const accounts = await listAccounts(env);
  if (!accounts.find((a) => a.account_id === PAPER_ACCOUNT_ID)) {
    await upsertAccount(env, {
      account_id: PAPER_ACCOUNT_ID,
      item_id: null,
      display_name: "Alpaca Paper",
      account_type: "brokerage_paper",
      asset_class: "strategy_paper",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 0,
      is_active: 1,
    });
  }
  await insertSnapshot(env, {
    account_id: PAPER_ACCOUNT_ID,
    taken_at: `${today}T07:00:00Z`,
    snapshot_date: today,
    current_balance: equity,
    available_balance: equity,
    source: "alpaca",
    raw_response: null,
  });
}

export async function handleScheduled(event: ScheduledEvent, env: Env): Promise<void> {
  const today = isoDate(event.scheduledTime);
  await syncAllItems(env, today);
  await snapshotAlpaca(env, today);

  if (event.cron === "0 7 * * MON") {
    const data = await computeWeeklySummary(env, new Date(event.scheduledTime));
    await postWeeklySummary(env, data);
    await logEvent(env, {
      event_type: "summary_posted",
      item_id: null,
      account_id: null,
      payload: { netWorth: data.netWorth, asOf: data.asOf },
    });
  }
}
