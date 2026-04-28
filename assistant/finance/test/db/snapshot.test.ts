import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import {
  getHistory,
  getLatestSnapshots,
  insertSnapshot,
  upsertAccount,
} from "../../src/db/queries";

async function reset(): Promise<void> {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
}

describe("snapshot round-trip", () => {
  beforeEach(reset);

  it("persists per-day snapshots idempotently and returns history in reverse-chronological order", async () => {
    await upsertAccount(env, {
      account_id: "acc_checking",
      item_id: "item_wf",
      display_name: "Wells Checking",
      account_type: "checking",
      asset_class: "cash",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 1,
      is_active: 1,
    });

    const dates = ["2026-04-15", "2026-04-16", "2026-04-17", "2026-04-18", "2026-04-19"];
    for (const [i, date] of dates.entries()) {
      await insertSnapshot(env, {
        account_id: "acc_checking",
        taken_at: `${date}T07:00:00Z`,
        snapshot_date: date,
        current_balance: 1000 + i * 10,
        available_balance: 1000 + i * 10,
        source: "plaid",
        raw_response: JSON.stringify({ marker: i }),
      });
    }

    // duplicate-day insert must be ignored
    await insertSnapshot(env, {
      account_id: "acc_checking",
      taken_at: "2026-04-17T23:00:00Z",
      snapshot_date: "2026-04-17",
      current_balance: 9999,
      available_balance: 9999,
      source: "plaid",
      raw_response: null,
    });

    const latest = await getLatestSnapshots(env);
    expect(latest).toHaveLength(1);
    expect(latest[0]!.account_id).toBe("acc_checking");
    expect(latest[0]!.current_balance).toBe(1040);
    expect(latest[0]!.snapshot_date).toBe("2026-04-19");

    const history = await getHistory(env, "acc_checking", 7);
    expect(history.map((r) => r.snapshot_date)).toEqual([
      "2026-04-19",
      "2026-04-18",
      "2026-04-17",
      "2026-04-16",
      "2026-04-15",
    ]);
    const day3 = history.find((r) => r.snapshot_date === "2026-04-17")!;
    expect(day3.current_balance).toBe(1020);
  });
});
