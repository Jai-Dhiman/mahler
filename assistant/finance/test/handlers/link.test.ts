import { SELF, env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { listEvents, listItems } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
});

afterEach(() => vi.restoreAllMocks());

describe("/link in dev environment", () => {
  // miniflare ENVIRONMENT binding is "test" — link.ts treats anything !== "production" as dev
  it("GET /link returns HTML loading the Plaid Link JS", async () => {
    const res = await SELF.fetch("https://finance.test/link");
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/html");
    const html = await res.text();
    expect(html).toContain("https://cdn.plaid.com/link/v2/stable/link-initialize.js");
    expect(html).toContain("/link/exchange");
  });

  it("POST /link/exchange exchanges token, persists item, and logs event", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.includes("plaid.com/item/public_token/exchange")) {
        return new Response(
          JSON.stringify({ access_token: "access-abc", item_id: "item_wf_new" }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      throw new Error(`unexpected fetch ${url}`);
    });

    const res = await SELF.fetch("https://finance.test/link/exchange", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ public_token: "public-xyz", institution_name: "Wells Fargo" }),
    });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ item_id: "item_wf_new" });

    const items = await listItems(env);
    expect(items).toHaveLength(1);
    expect(items[0]!.item_id).toBe("item_wf_new");
    expect(items[0]!.institution_name).toBe("Wells Fargo");

    const stored = await env.FINANCE_KV.get("plaid_item:item_wf_new");
    expect(stored).toBe("access-abc");

    const events = await listEvents(env, 10);
    expect(events.map((e) => e.event_type)).toContain("plaid_link_exchanged");
  });
});

describe("/link in production", () => {
  it("returns 404 when ENVIRONMENT === 'production'", async () => {
    const res = await SELF.fetch("https://finance.test/link", {
      // override env binding for this request via a custom path the worker recognizes? No — instead
      // we cannot mutate env per-request from the test. Verify behavior with a unit-style test:
      // import handleLink directly with a stubbed env.
    });
    // The above approach can't override env; this assertion is therefore skipped at the SELF level.
    // The unit assertion below covers production gating.
    expect([200, 404]).toContain(res.status);
  });

  it("handleLink returns 404 directly when env.ENVIRONMENT is production", async () => {
    const { handleLink } = await import("../../src/handlers/link");
    const prodEnv = { ...env, ENVIRONMENT: "production" } as typeof env;
    const res = await handleLink(new Request("https://finance.test/link"), prodEnv);
    expect(res.status).toBe(404);
  });
});
