import { SELF, env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { listEvents, listItems, upsertItem } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
});

afterEach(() => vi.restoreAllMocks());

describe("plaid webhook", () => {
  it("rejects requests without the verification header", async () => {
    const res = await SELF.fetch("https://finance.test/webhook/plaid", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ webhook_type: "ITEM", webhook_code: "ERROR", item_id: "item_wf" }),
    });
    expect(res.status).toBe(401);
  });

  it("on ITEM_ERROR: marks item needs_reauth, logs event, posts Discord nudge", async () => {
    let discordPosted: { content?: string } | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url === "https://discord.test/webhook") {
        discordPosted = JSON.parse(init!.body as string);
        return new Response("", { status: 204 });
      }
      throw new Error(`unexpected fetch ${url}`);
    });

    const res = await SELF.fetch("https://finance.test/webhook/plaid", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "plaid-verification": "test-webhook-secret",
      },
      body: JSON.stringify({
        webhook_type: "ITEM",
        webhook_code: "ERROR",
        item_id: "item_wf",
        error: { error_code: "ITEM_LOGIN_REQUIRED" },
      }),
    });
    expect(res.status).toBe(200);

    const items = await listItems(env);
    expect(items[0]!.status).toBe("needs_reauth");
    expect(items[0]!.last_error).toBe("ITEM_LOGIN_REQUIRED");

    const events = await listEvents(env, 10);
    expect(events.some((e) => e.event_type === "reauth_needed")).toBe(true);

    expect(discordPosted).not.toBeNull();
    expect(JSON.stringify(discordPosted)).toContain("Wells Fargo");
    expect(JSON.stringify(discordPosted).toLowerCase()).toContain("re-auth");
  });

  it("ignores unrelated webhook codes", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const res = await SELF.fetch("https://finance.test/webhook/plaid", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "plaid-verification": "test-webhook-secret",
      },
      body: JSON.stringify({
        webhook_type: "TRANSACTIONS",
        webhook_code: "DEFAULT_UPDATE",
        item_id: "item_wf",
      }),
    });
    expect(res.status).toBe(200);
    expect(fetchSpy).not.toHaveBeenCalled();
    const items = await listItems(env);
    expect(items[0]!.status).toBe("ok");
  });
});
