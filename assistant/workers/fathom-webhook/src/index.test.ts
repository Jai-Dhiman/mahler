import { describe, it, expect } from "vitest";
import { verifySignature } from "./index";

const TEST_SECRET = "whsec_dGVzdC1zZWNyZXQ="; // whsec_ + base64("test-secret")

async function makeSignature(
  webhookId: string,
  timestamp: string,
  body: string,
  secret: string
): Promise<string> {
  const rawSecret = atob(secret.replace(/^whsec_/, ""));
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(rawSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signed = `${webhookId}.${timestamp}.${body}`;
  const sigBytes = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signed));
  return "v1," + btoa(String.fromCharCode(...new Uint8Array(sigBytes)));
}

describe("verifySignature", () => {
  it("returns true for a valid signature", async () => {
    const id = "msg_test123";
    const ts = String(Math.floor(Date.now() / 1000));
    const body = '{"title":"Test Meeting","recording_id":1}';
    const sig = await makeSignature(id, ts, body, TEST_SECRET);
    const result = await verifySignature(id, ts, body, sig, TEST_SECRET);
    expect(result).toBe(true);
  });

  it("returns false when body is tampered", async () => {
    const id = "msg_test123";
    const ts = String(Math.floor(Date.now() / 1000));
    const body = '{"title":"Test Meeting","recording_id":1}';
    const sig = await makeSignature(id, ts, body, TEST_SECRET);
    const result = await verifySignature(id, ts, '{"title":"Tampered"}', sig, TEST_SECRET);
    expect(result).toBe(false);
  });

  it("returns false when secret is wrong", async () => {
    const id = "msg_test123";
    const ts = String(Math.floor(Date.now() / 1000));
    const body = '{"title":"Test Meeting","recording_id":1}';
    const sig = await makeSignature(id, ts, body, TEST_SECRET);
    const result = await verifySignature(id, ts, body, sig, "whsec_d3JvbmctcGFzc3dvcmQ=");
    expect(result).toBe(false);
  });

  it("returns false when timestamp is more than 5 minutes old", async () => {
    const id = "msg_test123";
    const staleTs = String(Math.floor(Date.now() / 1000) - 400);
    const body = '{"title":"Test Meeting","recording_id":1}';
    const sig = await makeSignature(id, staleTs, body, TEST_SECRET);
    const result = await verifySignature(id, staleTs, body, sig, TEST_SECRET);
    expect(result).toBe(false);
  });

  it("returns false when signature header has no v1, prefix", async () => {
    const id = "msg_test123";
    const ts = String(Math.floor(Date.now() / 1000));
    const body = '{"title":"Test Meeting","recording_id":1}';
    const result = await verifySignature(id, ts, body, "invalidsig", TEST_SECRET);
    expect(result).toBe(false);
  });
});

import { vi, afterEach } from "vitest";
import { extractSummary } from "./index";

describe("extractSummary", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns default_summary.markdown_formatted from payload when present", async () => {
    const meeting = {
      recording_id: 1,
      default_summary: { markdown_formatted: "## Discussion\n- Topic A\n- Topic B" },
    };
    const result = await extractSummary(meeting, "any-key");
    expect(result).toBe("## Discussion\n- Topic A\n- Topic B");
  });

  it("calls Fathom API when default_summary is null and returns markdown_formatted", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ markdown_formatted: "## Fetched\n- Item 1" }),
    }));
    const meeting = { recording_id: 42, default_summary: null };
    const result = await extractSummary(meeting, "test-api-key");
    expect(result).toBe("## Fetched\n- Item 1");
    expect(fetch).toHaveBeenCalledWith(
      "https://api.fathom.video/recordings/42/summary",
      { headers: { "X-Api-Key": "test-api-key" } }
    );
  });

  it("throws when Fathom API returns non-OK status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    const meeting = { recording_id: 7, default_summary: undefined };
    await expect(extractSummary(meeting, "key")).rejects.toThrow(
      "Fathom API error 503 fetching summary for recording 7"
    );
  });
});

import { env } from "cloudflare:test";
import { checkAndSetDedup } from "./index";

describe("checkAndSetDedup", () => {
  it("returns false on first call and true on second call for the same recording_id", async () => {
    const recordingId = 88881;
    const first = await checkAndSetDedup(env.KV, recordingId);
    expect(first).toBe(false);
    const second = await checkAndSetDedup(env.KV, recordingId);
    expect(second).toBe(true);
  });

  it("treats different recording_ids as independent", async () => {
    const dup1 = await checkAndSetDedup(env.KV, 88882);
    const dup2 = await checkAndSetDedup(env.KV, 88883);
    expect(dup1).toBe(false);
    expect(dup2).toBe(false);
  });
});

import { buildDiscordMessage } from "./index";

describe("buildDiscordMessage", () => {
  it("contains @mention, [FATHOM_MEETING] tag, title, attendee, and summary", () => {
    const msg = buildDiscordMessage(
      "1:1 with Alice Chen",
      [{ name: "Alice Chen", email: "alice@example.com", is_external: true }],
      "## Summary\n- Discussed roadmap",
      "123456789012345678"
    );
    expect(msg).toContain("<@123456789012345678>");
    expect(msg).toContain("[FATHOM_MEETING]");
    expect(msg).toContain("1:1 with Alice Chen");
    expect(msg).toContain("Alice Chen <alice@example.com>");
    expect(msg).toContain("## Summary\n- Discussed roadmap");
  });

  it("omits attendees with no email", () => {
    const msg = buildDiscordMessage(
      "Team Sync",
      [
        { name: "Alice Chen", email: "alice@example.com", is_external: true },
        { name: null, email: null, is_external: false },
      ],
      "Short summary.",
      "999"
    );
    expect(msg).toContain("Alice Chen <alice@example.com>");
    expect(msg).not.toContain("null");
  });

  it("renders attendee email only when name is null", () => {
    const msg = buildDiscordMessage(
      "Intro Call",
      [{ name: null, email: "unknown@corp.com", is_external: true }],
      "Notes here.",
      "111"
    );
    expect(msg).toContain("unknown@corp.com");
    expect(msg).not.toContain("null");
  });

  it("truncates to 2000 chars when message exceeds Discord limit", () => {
    const longSummary = "x".repeat(2000);
    const msg = buildDiscordMessage("Title", [], longSummary, "bot123");
    expect(msg.length).toBeLessThanOrEqual(2000);
    expect(msg).toContain("…(truncated)");
  });
});

import { SELF, env } from "cloudflare:test";

async function signedRequest(body: string, recordingId: number): Promise<Request> {
  const id = `msg_integration_${recordingId}`;
  const ts = String(Math.floor(Date.now() / 1000));
  const rawSecret = atob(TEST_SECRET.replace(/^whsec_/, ""));
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(rawSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signed = `${id}.${ts}.${body}`;
  const sigBytes = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signed));
  const sig = "v1," + btoa(String.fromCharCode(...new Uint8Array(sigBytes)));
  return new Request("https://fathom-webhook.workers.dev/", {
    method: "POST",
    headers: {
      "webhook-id": id,
      "webhook-timestamp": ts,
      "webhook-signature": sig,
      "content-type": "application/json",
    },
    body,
  });
}

const FIXTURE_MEETING = {
  title: "1:1 with Alice Chen",
  recording_id: 77701,
  default_summary: { markdown_formatted: "## Meeting Notes\n- Discussed Q3 roadmap" },
  calendar_invitees: [
    { name: "Alice Chen", email: "alice@example.com", is_external: true },
  ],
};

describe("fetch handler", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns 401 for an invalid signature", async () => {
    const req = new Request("https://fathom-webhook.workers.dev/", {
      method: "POST",
      headers: {
        "webhook-id": "msg_bad",
        "webhook-timestamp": String(Math.floor(Date.now() / 1000)),
        "webhook-signature": "v1,invalidsignaturevalue",
        "content-type": "application/json",
      },
      body: JSON.stringify(FIXTURE_MEETING),
    });
    const resp = await SELF.fetch(req);
    expect(resp.status).toBe(401);
  });

  it("returns 200 silently for a duplicate recording_id without calling Discord", async () => {
    const body = JSON.stringify({ ...FIXTURE_MEETING, recording_id: 77702 });
    const req1 = await signedRequest(body, 77702);
    const req2 = await signedRequest(body, 77702);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));

    const resp1 = await SELF.fetch(req1);
    expect(resp1.status).toBe(200);
    const callsAfterFirst = (fetch as ReturnType<typeof vi.fn>).mock.calls.length;

    const resp2 = await SELF.fetch(req2);
    expect(resp2.status).toBe(200);
    const callsAfterSecond = (fetch as ReturnType<typeof vi.fn>).mock.calls.length;

    expect(callsAfterSecond).toBe(callsAfterFirst);
  });

  it("returns 200 and posts to Discord for a valid new webhook", async () => {
    const discordBodies: string[] = [];
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
      if (typeof url === "string" && url.includes("discord-test.invalid")) {
        discordBodies.push(typeof init?.body === "string" ? init.body : await new Response(init?.body).text());
        return Promise.resolve({ ok: true });
      }
      return Promise.resolve({ ok: true, json: async () => ({ markdown_formatted: "fallback" }) });
    }));

    const body = JSON.stringify({ ...FIXTURE_MEETING, recording_id: 77703 });
    const req = await signedRequest(body, 77703);
    const resp = await SELF.fetch(req);

    expect(resp.status).toBe(200);
    expect(discordBodies.length).toBe(1);
    const posted = JSON.parse(discordBodies[0]) as { content: string };
    expect(posted.content).toContain("[FATHOM_MEETING]");
    expect(posted.content).toContain("1:1 with Alice Chen");
    expect(posted.content).toContain("Alice Chen <alice@example.com>");
  });

  it("returns 500 when Discord webhook POST fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    const body = JSON.stringify({ ...FIXTURE_MEETING, recording_id: 77704 });
    const req = await signedRequest(body, 77704);
    const resp = await SELF.fetch(req);
    expect(resp.status).toBe(500);
  });
});
