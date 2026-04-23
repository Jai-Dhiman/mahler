import { env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getBalances, verifyWebhook } from "../../src/plaid/client";

beforeEach(async () => {
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-sandbox-abc");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getBalances", () => {
  it("loads access token from KV, calls plaid balance endpoint, returns accounts", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toContain("plaid.com/accounts/balance/get");
      const body = JSON.parse((init!.body as string) ?? "{}");
      expect(body.access_token).toBe("access-sandbox-abc");
      return new Response(
        JSON.stringify({
          accounts: [
            {
              account_id: "acc_checking",
              name: "Checking",
              official_name: "WF Everyday Checking",
              type: "depository",
              subtype: "checking",
              balances: { current: 1234.56, available: 1200.0, iso_currency_code: "USD" },
            },
          ],
          item: { item_id: "item_wf" },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    const balances = await getBalances(env, "item_wf");
    expect(balances.accounts).toHaveLength(1);
    expect(balances.accounts[0]!.account_id).toBe("acc_checking");
    expect(balances.accounts[0]!.balances.current).toBe(1234.56);
  });

  it("throws a typed error when KV has no token for the item", async () => {
    await expect(getBalances(env, "item_missing")).rejects.toThrow(/no access token/i);
  });
});

describe("verifyWebhook", () => {
  it("accepts a request with matching shared-secret header and rejects others", async () => {
    const body = JSON.stringify({ webhook_type: "ITEM", webhook_code: "ERROR" });
    await expect(verifyWebhook(env, body, new Headers({ "plaid-verification": "test-webhook-secret" }))).resolves.toBe(true);
    await expect(verifyWebhook(env, body, new Headers({ "plaid-verification": "wrong" }))).resolves.toBe(false);
    await expect(verifyWebhook(env, body, new Headers())).resolves.toBe(false);
  });
});
