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

function base64urlDecode(str: string): Uint8Array {
  const pad = (4 - (str.length % 4)) % 4;
  const b64 = str.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat(pad);
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function verifyPlaidJwt(env: Env, body: string, token: string): Promise<boolean> {
  const parts = token.split(".");
  if (parts.length !== 3) return false;

  let jwtHeader: { alg?: string; kid?: string };
  let jwtPayload: { iat?: number; request_body_sha256?: string };
  try {
    jwtHeader = JSON.parse(new TextDecoder().decode(base64urlDecode(parts[0]!)));
    jwtPayload = JSON.parse(new TextDecoder().decode(base64urlDecode(parts[1]!)));
  } catch (_err) {
    return false;
  }

  if (jwtHeader.alg !== "ES256" || !jwtHeader.kid) return false;

  const keyRes = await fetch(plaidUrl(env, "/webhook_verification_key/get"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      key_id: jwtHeader.kid,
    }),
  });
  if (!keyRes.ok) return false;
  const keyData = (await keyRes.json()) as { key: JsonWebKey & { expired_at: string | null } };
  if (keyData.key.expired_at) return false;

  let publicKey: CryptoKey;
  try {
    publicKey = await crypto.subtle.importKey(
      "jwk",
      keyData.key,
      { name: "ECDSA", namedCurve: "P-256" },
      false,
      ["verify"],
    );
  } catch (_err) {
    return false;
  }

  const signedInput = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
  const valid = await crypto.subtle.verify(
    { name: "ECDSA", hash: "SHA-256" },
    publicKey,
    base64urlDecode(parts[2]!),
    signedInput,
  );
  if (!valid) return false;

  const now = Math.floor(Date.now() / 1000);
  if (!jwtPayload.iat || Math.abs(now - jwtPayload.iat) > 300) return false;

  const bodyHashBuf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(body));
  const bodyHashHex = Array.from(new Uint8Array(bodyHashBuf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return bodyHashHex === jwtPayload.request_body_sha256;
}

export async function verifyWebhook(env: Env, body: string, headers: Headers): Promise<boolean> {
  const token = headers.get("plaid-verification");
  if (!token) return false;
  // Non-production: static secret comparison keeps dev/test working without JWK fetch
  if (env.ENVIRONMENT !== "production") {
    return token === env.PLAID_WEBHOOK_SECRET;
  }
  return verifyPlaidJwt(env, body, token);
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
