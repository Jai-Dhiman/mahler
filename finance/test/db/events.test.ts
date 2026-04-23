import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import { listEvents, logEvent } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_event_log").run();
});

describe("event log", () => {
  it("appends events with type/ids/payload and reads back in order", async () => {
    await logEvent(env, {
      event_type: "snapshot_run",
      item_id: null,
      account_id: null,
      payload: { itemsSucceeded: 2, snapshotsWritten: 4 },
    });
    await logEvent(env, {
      event_type: "item_error",
      item_id: "item_wf",
      account_id: null,
      payload: { code: "ITEM_LOGIN_REQUIRED" },
    });
    await logEvent(env, {
      event_type: "summary_posted",
      item_id: null,
      account_id: null,
      payload: { netWorth: 12345.67 },
    });

    const events = await listEvents(env, 10);
    expect(events).toHaveLength(3);
    expect(events.map((e) => e.event_type)).toEqual([
      "snapshot_run",
      "item_error",
      "summary_posted",
    ]);
    expect(JSON.parse(events[1]!.payload!)).toEqual({ code: "ITEM_LOGIN_REQUIRED" });
    expect(events[1]!.item_id).toBe("item_wf");
  });
});
