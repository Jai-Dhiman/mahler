import { env, createScheduledController } from "cloudflare:test";
import worker from "../../src/index";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getLatestSnapshots,
  listEvents,
  upsertAccount,
  upsertItem,
} from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-wf");
  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
  await upsertAccount(env, {
    account_id: "acc_paper",
    item_id: null,
    display_name: "Alpaca Paper",
    account_type: "brokerage_paper",
    asset_class: "strategy_paper",
    currency: "USD",
    is_liability: 0,
    include_in_networth: 0,
    is_active: 1,
  });
});

afterEach(() => vi.restoreAllMocks());

function plaidBalanceFetch(): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
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
            balances: { current: 3200, available: 3200, iso_currency_code: "USD" },
          }],
          item: { item_id: "item_wf" },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (url === "https://paper-api.alpaca.markets/v2/account") {
      return new Response(JSON.stringify({ equity: "10250.42", cash: "5000" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url === "https://discord.test/webhook") {
      return new Response("", { status: 204 });
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;
}

describe("scheduled handler", () => {
  it("daily cron writes plaid + alpaca snapshots and logs snapshot_run", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(plaidBalanceFetch());
    const controller = createScheduledController({
      scheduledTime: new Date("2026-04-19T07:00:00Z").getTime(),
      cron: "0 7 * * *",
    });
    await worker.scheduled!(controller, env, {} as ExecutionContext);

    const snapshots = await getLatestSnapshots(env);
    const ids = snapshots.map((s) => s.account_id).sort();
    expect(ids).toEqual(["acc_chk", "acc_paper"]);
    const paper = snapshots.find((s) => s.account_id === "acc_paper")!;
    expect(paper.current_balance).toBe(10250.42);
    expect(paper.source).toBe("alpaca");

    const events = await listEvents(env, 10);
    expect(events.some((e) => e.event_type === "snapshot_run")).toBe(true);
  });

  it("Sunday cron runs sync, posts weekly summary embed to discord", async () => {
    let discordPostBody: unknown = null;
    vi.spyOn(globalThis, "fetch").mockImplementation((async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url === "https://discord.test/webhook") {
        discordPostBody = JSON.parse(init!.body as string);
        return new Response("", { status: 204 });
      }
      return plaidBalanceFetch()(input, init);
    }) as typeof fetch);

    const controller = createScheduledController({
      scheduledTime: new Date("2026-04-19T07:00:00Z").getTime(),
      cron: "0 7 * * SUN",
    });
    await worker.scheduled!(controller, env, {} as ExecutionContext);

    expect(discordPostBody).not.toBeNull();
    const body = discordPostBody as { embeds: Array<{ title: string }> };
    expect(body.embeds[0]!.title).toContain("Weekly Finance");

    const events = await listEvents(env, 10);
    expect(events.some((e) => e.event_type === "summary_posted")).toBe(true);
  });
});
