import { listAccounts, listItems } from "../db/queries";
import type { AccountRow, Env, SnapshotRow, WeeklyAccountLine, WeeklyData } from "../types";

const DAY_MS = 86_400_000;

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

async function snapshotOnOrBefore(
  env: Env,
  accountId: string,
  asOfISO: string,
): Promise<SnapshotRow | null> {
  const row = await env.DB.prepare(
    `SELECT * FROM finance_balance_snapshot
     WHERE account_id = ? AND snapshot_date <= ?
     ORDER BY snapshot_date DESC LIMIT 1`,
  )
    .bind(accountId, asOfISO)
    .first<SnapshotRow>();
  return row ?? null;
}

function networthContribution(account: AccountRow, balance: number): number {
  if (!account.include_in_networth) return 0;
  return account.is_liability ? -balance : balance;
}

export async function computeWeeklySummary(env: Env, asOf: Date): Promise<WeeklyData> {
  const asOfISO = isoDate(asOf);
  const priorISO = isoDate(new Date(asOf.getTime() - 7 * DAY_MS));

  const accounts = await listAccounts(env);

  const accountLines: WeeklyAccountLine[] = [];
  let strategyPaper: number | null = null;
  let netWorth = 0;
  let priorNetWorth = 0;
  let priorAnyExists = false;

  for (const account of accounts) {
    const cur = await snapshotOnOrBefore(env, account.account_id, asOfISO);
    if (!cur) continue;
    const prior = await snapshotOnOrBefore(env, account.account_id, priorISO);
    if (prior) priorAnyExists = true;

    const curBal = cur.current_balance;
    const priorBal = prior ? prior.current_balance : null;

    if (account.asset_class === "strategy_paper") {
      strategyPaper = curBal;
      continue;
    }

    netWorth += networthContribution(account, curBal);
    if (prior) priorNetWorth += networthContribution(account, prior.current_balance);

    accountLines.push({
      account_id: account.account_id,
      display_name: account.display_name,
      account_type: account.account_type,
      current_balance: curBal,
      prior_week_balance: priorBal,
      delta: priorBal === null ? null : curBal - priorBal,
    });
  }

  const sparkline: Array<{ date: string; net_worth: number }> = [];
  for (let i = 3; i >= 0; i--) {
    const dt = new Date(asOf.getTime() - i * 7 * DAY_MS);
    const iso = isoDate(dt);
    let nw = 0;
    let any = false;
    for (const account of accounts) {
      if (account.asset_class === "strategy_paper") continue;
      const snap = await snapshotOnOrBefore(env, account.account_id, iso);
      if (snap) {
        any = true;
        nw += networthContribution(account, snap.current_balance);
      }
    }
    if (any) sparkline.push({ date: iso, net_worth: nw });
  }

  const items = await listItems(env);
  const staleCutoffMs = asOf.getTime() - 36 * 3600 * 1000;
  const staleItems = items
    .filter((i) => {
      if (!i.last_synced_at) return false;
      if (i.status !== "ok") return false;
      return new Date(i.last_synced_at).getTime() < staleCutoffMs;
    })
    .map((i) => ({
      item_id: i.item_id,
      institution_name: i.institution_name,
      last_synced_at: i.last_synced_at,
    }));

  return {
    asOf: asOfISO,
    netWorth,
    priorWeekNetWorth: priorAnyExists ? priorNetWorth : null,
    netWorthDelta: priorAnyExists ? netWorth - priorNetWorth : null,
    accounts: accountLines,
    strategyPaperEquity: strategyPaper,
    sparkline,
    staleItems,
  };
}
