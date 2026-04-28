import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import { insertSnapshot, upsertAccount, upsertItem, updateItemStatus } from "../../src/db/queries";
import { computeWeeklySummary } from "../../src/summary/compute";

async function reset(): Promise<void> {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
}

async function seedAccount(opts: {
  id: string;
  item: string | null;
  name: string;
  type: string;
  asset: string;
  liability?: 0 | 1;
  includeNw?: 0 | 1;
}): Promise<void> {
  await upsertAccount(env, {
    account_id: opts.id,
    item_id: opts.item,
    display_name: opts.name,
    account_type: opts.type,
    asset_class: opts.asset,
    currency: "USD",
    is_liability: opts.liability ?? 0,
    include_in_networth: opts.includeNw ?? 1,
    is_active: 1,
  });
}

async function seedSnap(accountId: string, date: string, balance: number, source: "plaid" | "alpaca" = "plaid"): Promise<void> {
  await insertSnapshot(env, {
    account_id: accountId,
    taken_at: `${date}T07:00:00Z`,
    snapshot_date: date,
    current_balance: balance,
    available_balance: balance,
    source,
    raw_response: null,
  });
}

beforeEach(reset);

describe("computeWeeklySummary", () => {
  it("computes net worth, week-over-week delta, paper P&L, and 4-week sparkline", async () => {
    await seedAccount({ id: "acc_chk", item: "item_wf", name: "WF Checking", type: "checking", asset: "cash" });
    await seedAccount({ id: "acc_wlth", item: "item_wlth", name: "WF Cash", type: "savings", asset: "cash" });
    await seedAccount({ id: "acc_paper", item: null, name: "Alpaca Paper", type: "brokerage_paper", asset: "strategy_paper", includeNw: 0 });

    // 4 weeks of weekly snapshots — Sundays
    const weeks = [
      { date: "2026-03-29", chk: 2800, wlth: 9000, paper: 10000 },
      { date: "2026-04-05", chk: 2900, wlth: 9100, paper: 10100 },
      { date: "2026-04-12", chk: 3000, wlth: 9200, paper: 10200 },
      { date: "2026-04-19", chk: 3200, wlth: 9300, paper: 10250 },
    ];
    for (const w of weeks) {
      await seedSnap("acc_chk", w.date, w.chk);
      await seedSnap("acc_wlth", w.date, w.wlth);
      await seedSnap("acc_paper", w.date, w.paper, "alpaca");
    }

    const data = await computeWeeklySummary(env, new Date("2026-04-19T23:00:00Z"));

    expect(data.asOf).toBe("2026-04-19");
    expect(data.netWorth).toBe(12500);
    expect(data.priorWeekNetWorth).toBe(12200);
    expect(data.netWorthDelta).toBe(300);

    const chk = data.accounts.find((a) => a.account_id === "acc_chk")!;
    expect(chk.current_balance).toBe(3200);
    expect(chk.prior_week_balance).toBe(3000);
    expect(chk.delta).toBe(200);

    expect(data.strategyPaperEquity).toBe(10250);

    expect(data.sparkline).toHaveLength(4);
    expect(data.sparkline.map((p) => p.net_worth)).toEqual([11800, 12000, 12200, 12500]);
    expect(data.sparkline[3]!.date).toBe("2026-04-19");

    expect(data.staleItems).toEqual([]);
  });

  it("flags items as stale when last_synced_at is older than 36 hours", async () => {
    await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
    await seedAccount({ id: "acc_chk", item: "item_wf", name: "WF Checking", type: "checking", asset: "cash" });
    await seedSnap("acc_chk", "2026-04-19", 3200);

    // mark item as last synced 48 hours before asOf
    await env.DB.prepare(
      `UPDATE finance_plaid_item SET last_synced_at = ? WHERE item_id = ?`,
    )
      .bind("2026-04-17T23:00:00Z", "item_wf")
      .run();

    const data = await computeWeeklySummary(env, new Date("2026-04-19T23:00:00Z"));
    expect(data.staleItems).toHaveLength(1);
    expect(data.staleItems[0]!.item_id).toBe("item_wf");
  });

  it("returns null deltas when no prior-week snapshot exists", async () => {
    await seedAccount({ id: "acc_new", item: "item_x", name: "Brand New", type: "checking", asset: "cash" });
    await seedSnap("acc_new", "2026-04-19", 500);

    const data = await computeWeeklySummary(env, new Date("2026-04-19T23:00:00Z"));
    expect(data.priorWeekNetWorth).toBeNull();
    expect(data.netWorthDelta).toBeNull();
    expect(data.accounts[0]!.prior_week_balance).toBeNull();
    expect(data.accounts[0]!.delta).toBeNull();
  });
});
