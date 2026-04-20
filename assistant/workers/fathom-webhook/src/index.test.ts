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
