import { env } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import { exchangePublicToken } from "../../src/plaid/client";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("exchangePublicToken", () => {
  it("exchanges a public token, persists access token in KV, and returns item id", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toContain("plaid.com/item/public_token/exchange");
      return new Response(
        JSON.stringify({ access_token: "access-sandbox-abc", item_id: "item_wf_123" }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    const result = await exchangePublicToken(env, "public-sandbox-xyz");
    expect(result).toEqual({ item_id: "item_wf_123" });

    const stored = await env.FINANCE_KV.get("plaid_item:item_wf_123");
    expect(stored).toBe("access-sandbox-abc");

    const callBody = JSON.parse(fetchSpy.mock.calls[0]![1]!.body as string);
    expect(callBody).toEqual({
      client_id: "test-client",
      secret: "test-secret",
      public_token: "public-sandbox-xyz",
    });
  });
});
