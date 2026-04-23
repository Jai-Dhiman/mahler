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
