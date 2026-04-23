import { env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getLatestSnapshots,
  listEvents,
  listItems,
  upsertItem,
} from "../../src/db/queries";
import { syncAllItems } from "../../src/plaid/sync";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-wf");
  await env.FINANCE_KV.put("plaid_item:item_wlth", "access-wlth");
  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
  await upsertItem(env, { item_id: "item_wlth", institution_name: "Wealthfront" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

function balanceResponse(accounts: Array<{ id: string; name: string; current: number; subtype: string }>) {
  return new Response(
    JSON.stringify({
      accounts: accounts.map((a) => ({
        account_id: a.id,
        name: a.name,
        official_name: a.name,
        type: "depository",
        subtype: a.subtype,
        balances: { current: a.current, available: a.current, iso_currency_code: "USD" },
      })),
      item: { item_id: "ignored" },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

describe("syncAllItems", () => {
  it("polls every item, writes snapshots per account, logs success, and survives one failing item", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const body = JSON.parse((init!.body as string) ?? "{}");
      if (body.access_token === "access-wf") {
        return balanceResponse([{ id: "acc_wf_chk", name: "WF Checking", current: 3210.5, subtype: "checking" }]);
      }
      if (body.access_token === "access-wlth") {
        // simulate plaid hiccup — non-2xx
        return new Response(JSON.stringify({ error_code: "ITEM_LOGIN_REQUIRED" }), {
          status: 400,
          headers: { "content-type": "application/json" },
        });
      }
      throw new Error(`unexpected fetch: ${(input as Request).url ?? input}`);
    });

    const result = await syncAllItems(env, "2026-04-19");

    expect(result.itemsAttempted).toBe(2);
    expect(result.itemsSucceeded).toBe(1);
    expect(result.snapshotsWritten).toBe(1);
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0]!.item_id).toBe("item_wlth");

    const snapshots = await getLatestSnapshots(env);
    expect(snapshots).toHaveLength(1);
    expect(snapshots[0]!.account_id).toBe("acc_wf_chk");
    expect(snapshots[0]!.current_balance).toBe(3210.5);
    expect(snapshots[0]!.snapshot_date).toBe("2026-04-19");
    expect(snapshots[0]!.source).toBe("plaid");

    const items = await listItems(env);
    const wf = items.find((i) => i.item_id === "item_wf")!;
    const wlth = items.find((i) => i.item_id === "item_wlth")!;
    expect(wf.status).toBe("ok");
    expect(wlth.status).toBe("error");
    expect(wlth.last_error).toContain("400");

    const events = await listEvents(env, 10);
    const eventTypes = events.map((e) => e.event_type);
    expect(eventTypes).toContain("snapshot_run");
    expect(eventTypes).toContain("item_error");
  });

  it("re-running for the same day does not double-write snapshots", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const body = JSON.parse((init!.body as string) ?? "{}");
      if (body.access_token === "access-wf") {
        return balanceResponse([{ id: "acc_wf_chk", name: "WF Checking", current: 3210.5, subtype: "checking" }]);
      }
      return new Response(JSON.stringify({ error_code: "X" }), { status: 400 });
    });

    await syncAllItems(env, "2026-04-19");
    await syncAllItems(env, "2026-04-19");

    const snapshots = await getLatestSnapshots(env);
    expect(snapshots).toHaveLength(1);
  });
});
