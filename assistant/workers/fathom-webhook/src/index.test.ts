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
});
