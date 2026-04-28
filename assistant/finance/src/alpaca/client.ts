import type { Env } from "../types";

interface AlpacaAccount {
  equity: string;
  cash: string;
}

export async function getPaperEquity(env: Env): Promise<number> {
  const res = await fetch("https://paper-api.alpaca.markets/v2/account", {
    headers: {
      "APCA-API-KEY-ID": env.ALPACA_PAPER_KEY_ID,
      "APCA-API-SECRET-KEY": env.ALPACA_PAPER_SECRET,
    },
  });
  if (!res.ok) {
    throw new Error(`alpaca paper account fetch failed: ${res.status} ${await res.text()}`);
  }
  const json = (await res.json()) as AlpacaAccount;
  const value = Number(json.equity);
  if (!Number.isFinite(value)) {
    throw new Error(`alpaca returned non-numeric equity: ${json.equity}`);
  }
  return value;
}
