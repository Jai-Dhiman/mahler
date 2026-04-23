import type { Env, WeeklyAccountLine, WeeklyData } from "../types";

const USD = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

function fmtSigned(n: number | null): string {
  if (n === null) return "n/a";
  const sign = n >= 0 ? "+" : "-";
  return `${sign}${USD.format(Math.abs(n))}`;
}

function accountField(line: WeeklyAccountLine): { name: string; value: string; inline: boolean } {
  const delta = line.delta === null ? "(no prior)" : fmtSigned(line.delta);
  return {
    name: `${line.display_name} (${line.account_type})`,
    value: `${USD.format(line.current_balance)}  ${delta}`,
    inline: true,
  };
}

export async function postWeeklySummary(env: Env, data: WeeklyData): Promise<void> {
  const fields: Array<{ name: string; value: string; inline: boolean }> = data.accounts.map(
    accountField,
  );
  if (data.strategyPaperEquity !== null) {
    fields.push({
      name: "Strategy Paper (Alpaca)",
      value: USD.format(data.strategyPaperEquity),
      inline: false,
    });
  }

  const description =
    `Net worth: ${USD.format(data.netWorth)}  ${fmtSigned(data.netWorthDelta)} ` +
    `vs prior week.`;

  const footerText = data.staleItems.length
    ? `Stale: ${data.staleItems
        .map((s) => `${s.institution_name} — re-auth via /link`)
        .join("; ")}`
    : "All items fresh";

  const color = data.netWorthDelta === null ? 0x808080 : data.netWorthDelta >= 0 ? 0x2ecc71 : 0xe74c3c;

  const embed = {
    title: `Weekly Finance — ${data.asOf}`,
    description,
    color,
    fields,
    footer: { text: footerText },
  };

  const res = await fetch(env.DISCORD_WEBHOOK_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ embeds: [embed] }),
  });
  if (!res.ok && res.status !== 204) {
    throw new Error(`discord webhook failed: ${res.status} ${await res.text()}`);
  }
}
