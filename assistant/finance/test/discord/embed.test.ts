import { env } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import { postWeeklySummary } from "../../src/discord/embed";
import type { WeeklyData } from "../../src/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const sample: WeeklyData = {
  asOf: "2026-04-19",
  netWorth: 12500.0,
  priorWeekNetWorth: 12000.0,
  netWorthDelta: 500.0,
  accounts: [
    {
      account_id: "acc_checking",
      display_name: "Wells Checking",
      account_type: "checking",
      current_balance: 3200.0,
      prior_week_balance: 3000.0,
      delta: 200.0,
    },
    {
      account_id: "acc_wlth_cash",
      display_name: "Wealthfront Cash",
      account_type: "savings",
      current_balance: 9300.0,
      prior_week_balance: 9000.0,
      delta: 300.0,
    },
  ],
  strategyPaperEquity: 10250.42,
  sparkline: [
    { date: "2026-03-29", net_worth: 11000 },
    { date: "2026-04-05", net_worth: 11500 },
    { date: "2026-04-12", net_worth: 12000 },
    { date: "2026-04-19", net_worth: 12500 },
  ],
  staleItems: [
    { item_id: "item_wf", institution_name: "Wells Fargo", last_synced_at: "2026-04-15T07:00:00Z" },
  ],
};

describe("postWeeklySummary", () => {
  it("posts a discord webhook embed with net worth, accounts, paper P&L, and stale-item footer", async () => {
    let capturedBody: unknown = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toBe("https://discord.test/webhook");
      capturedBody = JSON.parse(init!.body as string);
      return new Response("", { status: 204 });
    });

    await postWeeklySummary(env, sample);

    const body = capturedBody as { embeds: Array<Record<string, unknown>> };
    expect(body.embeds).toHaveLength(1);
    const embed = body.embeds[0]!;
    expect(embed.title).toContain("Weekly Finance");
    expect(embed.title).toContain("2026-04-19");
    expect(String(embed.description)).toContain("$12,500");
    expect(String(embed.description)).toContain("+$500");

    const fields = embed.fields as Array<{ name: string; value: string }>;
    const checking = fields.find((f) => f.name.includes("Wells Checking"));
    expect(checking).toBeTruthy();
    expect(checking!.value).toContain("$3,200");
    expect(checking!.value).toContain("+$200");

    const paper = fields.find((f) => f.name.includes("Strategy"));
    expect(paper).toBeTruthy();
    expect(paper!.value).toContain("$10,250.42");

    const footer = embed.footer as { text: string };
    expect(footer.text).toContain("Wells Fargo");
    expect(footer.text.toLowerCase()).toContain("re-auth");
  });
});
