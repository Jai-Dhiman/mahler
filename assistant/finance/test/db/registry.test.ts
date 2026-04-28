import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import {
  listAccounts,
  listItems,
  updateItemStatus,
  upsertAccount,
  upsertItem,
} from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
});

describe("plaid item + account registry", () => {
  it("upserts items and accounts, lists them, and reflects status updates", async () => {
    await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
    await upsertItem(env, { item_id: "item_wlth", institution_name: "Wealthfront" });
    await upsertItem(env, { item_id: "item_wlth", institution_name: "Wealthfront Inc" });

    await upsertAccount(env, {
      account_id: "acc_checking",
      item_id: "item_wf",
      display_name: "Wells Checking",
      account_type: "checking",
      asset_class: "cash",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 1,
      is_active: 1,
    });
    await upsertAccount(env, {
      account_id: "acc_paper",
      item_id: null,
      display_name: "Alpaca Paper",
      account_type: "brokerage_paper",
      asset_class: "strategy_paper",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 0,
      is_active: 1,
    });

    const items = await listItems(env);
    expect(items.map((i) => i.item_id).sort()).toEqual(["item_wf", "item_wlth"]);
    const wlth = items.find((i) => i.item_id === "item_wlth")!;
    expect(wlth.institution_name).toBe("Wealthfront Inc");
    expect(wlth.status).toBe("ok");
    expect(wlth.last_error).toBeNull();

    const accounts = await listAccounts(env);
    expect(accounts.map((a) => a.account_id).sort()).toEqual(["acc_checking", "acc_paper"]);
    expect(accounts.find((a) => a.account_id === "acc_paper")!.include_in_networth).toBe(0);

    await updateItemStatus(env, "item_wf", "needs_reauth", "ITEM_LOGIN_REQUIRED");
    const after = await listItems(env);
    const wf = after.find((i) => i.item_id === "item_wf")!;
    expect(wf.status).toBe("needs_reauth");
    expect(wf.last_error).toBe("ITEM_LOGIN_REQUIRED");
  });
});
