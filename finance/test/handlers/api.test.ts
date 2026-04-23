import { SELF, env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { insertSnapshot, upsertAccount, upsertItem } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-wf");

  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
  await upsertAccount(env, {
    account_id: "acc_chk",
    item_id: "item_wf",
    display_name: "WF Checking",
    account_type: "checking",
    asset_class: "cash",
    currency: "USD",
    is_liability: 0,
    include_in_networth: 1,
    is_active: 1,
  });
  await insertSnapshot(env, {
    account_id: "acc_chk",
    taken_at: "2026-04-19T07:00:00Z",
    snapshot_date: "2026-04-19",
    current_balance: 3200,
    available_balance: 3200,
    source: "plaid",
    raw_response: null,
  });
});

afterEach(() => vi.restoreAllMocks());

describe("read API auth", () => {
  it("returns 401 without bearer", async () => {
    const res = await SELF.fetch("https://finance.test/balances");
    expect(res.status).toBe(401);
  });

  it("returns 401 with wrong bearer", async () => {
    const res = await SELF.fetch("https://finance.test/balances", {
      headers: { authorization: "Bearer wrong" },
    });
    expect(res.status).toBe(401);
  });
});

describe("read API happy path", () => {
  const auth = { authorization: "Bearer test-token" };

  it("/balances returns latest snapshot rows", async () => {
    const res = await SELF.fetch("https://finance.test/balances", { headers: auth });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { snapshots: Array<{ account_id: string; current_balance: number }> };
    expect(body.snapshots).toHaveLength(1);
    expect(body.snapshots[0]!.account_id).toBe("acc_chk");
    expect(body.snapshots[0]!.current_balance).toBe(3200);
  });

  it("/networth returns weekly summary shape", async () => {
    const res = await SELF.fetch("https://finance.test/networth", { headers: auth });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { netWorth: number; accounts: unknown[] };
    expect(body.netWorth).toBe(3200);
    expect(Array.isArray(body.accounts)).toBe(true);
  });

  it("/history returns rows for the given account+window", async () => {
    const res = await SELF.fetch(
      "https://finance.test/history?account_id=acc_chk&days=14",
      { headers: auth },
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { history: Array<{ snapshot_date: string }> };
    expect(body.history.map((r) => r.snapshot_date)).toEqual(["2026-04-19"]);
  });

  it("/refresh triggers sync and returns result", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.includes("plaid.com/accounts/balance/get")) {
        return new Response(
          JSON.stringify({
            accounts: [{
              account_id: "acc_chk",
              name: "WF Checking",
              official_name: "WF Checking",
              type: "depository",
              subtype: "checking",
              balances: { current: 3300, available: 3300, iso_currency_code: "USD" },
            }],
            item: { item_id: "item_wf" },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      // pass-through to SELF for the actual /refresh request
      return new Response("ok", { status: 200 });
    });

    const res = await SELF.fetch("https://finance.test/refresh", {
      method: "POST",
      headers: auth,
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { itemsSucceeded: number; snapshotsWritten: number };
    expect(body.itemsSucceeded).toBe(1);
    expect(body.snapshotsWritten).toBe(1);
  });
});
