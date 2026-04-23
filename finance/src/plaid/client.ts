import type { Env } from "../types";

const PLAID_BASE: Record<string, string> = {
  dev: "https://development.plaid.com",
  test: "https://development.plaid.com",
  sandbox: "https://sandbox.plaid.com",
  production: "https://production.plaid.com",
};

function plaidUrl(env: Env, path: string): string {
  const base = PLAID_BASE[env.ENVIRONMENT] ?? PLAID_BASE.dev!;
  return `${base}${path}`;
}

interface PlaidExchangeResponse {
  access_token: string;
  item_id: string;
}

export async function exchangePublicToken(
  env: Env,
  publicToken: string,
): Promise<{ item_id: string }> {
  const res = await fetch(plaidUrl(env, "/item/public_token/exchange"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      public_token: publicToken,
    }),
  });
  if (!res.ok) {
    throw new Error(`plaid exchange failed: ${res.status} ${await res.text()}`);
  }
  const json = (await res.json()) as PlaidExchangeResponse;
  await env.FINANCE_KV.put(`plaid_item:${json.item_id}`, json.access_token);
  return { item_id: json.item_id };
}

export interface PlaidAccount {
  account_id: string;
  name: string;
  official_name: string | null;
  type: string;
  subtype: string | null;
  balances: {
    current: number;
    available: number | null;
    iso_currency_code: string | null;
  };
}

export interface PlaidBalanceResponse {
  accounts: PlaidAccount[];
  item: { item_id: string };
}

export async function getBalances(env: Env, itemId: string): Promise<PlaidBalanceResponse> {
  const accessToken = await env.FINANCE_KV.get(`plaid_item:${itemId}`);
  if (!accessToken) {
    throw new Error(`no access token in KV for item ${itemId}`);
  }
  const res = await fetch(plaidUrl(env, "/accounts/balance/get"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      access_token: accessToken,
    }),
  });
  if (!res.ok) {
    throw new Error(`plaid balance fetch failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as PlaidBalanceResponse;
}

export function verifyWebhook(env: Env, _body: string, headers: Headers): boolean {
  const provided = headers.get("plaid-verification");
  if (!provided) return false;
  return provided === env.PLAID_WEBHOOK_SECRET;
}

export async function createLinkToken(env: Env): Promise<{ link_token: string }> {
  const res = await fetch(plaidUrl(env, "/link/token/create"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      client_name: "Mahler Finance",
      country_codes: ["US"],
      language: "en",
      user: { client_user_id: "mahler-single-user" },
      products: ["balance"],
    }),
  });
  if (!res.ok) {
    throw new Error(`plaid link token create failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as { link_token: string };
}
