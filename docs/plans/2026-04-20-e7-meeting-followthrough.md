# E7 Meeting Follow-Through Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** After any recorded meeting, Mahler automatically generates context-aware action items, creates Notion tasks, and updates CRM contacts — closing the loop that meeting-prep opens.
**Spec:** docs/specs/2026-04-20-e7-meeting-followthrough-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

Group A (parallel): Task 1 (HMAC verification + scaffold), Task 6 (SKILL.md)
Group B (sequential, depends on A): Task 2 (KV dedup)
Group C (sequential, depends on B): Task 3 (summary extraction)
Group D (sequential, depends on C): Task 4 (Discord message formatting)
Group E (sequential, depends on D): Task 5 (full handler integration)

---

### Task 1: CF Worker scaffold + HMAC signature verification
**Group:** A (parallel with Task 6)

**Behavior being verified:** `verifySignature` returns `true` for a valid Fathom HMAC-SHA256 signature and `false` when the body has been tampered with or the secret is wrong.

**Interface under test:** `verifySignature(webhookId, webhookTimestamp, rawBody, signatureHeader, secret): Promise<boolean>`

**Files:**
- Create: `assistant/workers/fathom-webhook/wrangler.toml`
- Create: `assistant/workers/fathom-webhook/package.json`
- Create: `assistant/workers/fathom-webhook/vitest.config.ts`
- Create: `assistant/workers/fathom-webhook/src/index.ts`
- Create: `assistant/workers/fathom-webhook/src/index.test.ts`

- [ ] **Step 1: Write the failing test**

Create `assistant/workers/fathom-webhook/src/index.test.ts`:

```typescript
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: FAIL — `Cannot find module './index'`

If the test PASSES without the implementation, the test is wrong: it is testing shape, not behavior. Rewrite.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `assistant/workers/fathom-webhook/wrangler.toml`:

```toml
name = "fathom-webhook"
main = "src/index.ts"
compatibility_date = "2025-01-01"

[[kv_namespaces]]
binding = "KV"
id = "0a93ac9040324708a8b9f00eed8715e9"

[vars]
DISCORD_BOT_USER_ID = ""
```

Create `assistant/workers/fathom-webhook/package.json`:

```json
{
  "name": "fathom-webhook",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "deploy": "wrangler deploy",
    "dev": "wrangler dev",
    "test": "vitest run"
  },
  "devDependencies": {
    "@cloudflare/vitest-pool-workers": "^0.5.0",
    "@cloudflare/workers-types": "^4.0.0",
    "typescript": "^5.0.0",
    "vitest": "^1.0.0",
    "wrangler": "^3.0.0"
  }
}
```

Create `assistant/workers/fathom-webhook/vitest.config.ts`:

```typescript
import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    poolOptions: {
      workers: {
        wrangler: { configPath: "./wrangler.toml" },
        miniflare: {
          bindings: {
            FATHOM_WEBHOOK_SECRET: "whsec_dGVzdC1zZWNyZXQ=",
            FATHOM_API_KEY: "test-fathom-api-key",
            DISCORD_TRIAGE_WEBHOOK: "https://discord-test.invalid/webhook",
            DISCORD_BOT_USER_ID: "123456789012345678",
          },
        },
      },
    },
  },
});
```

Create `assistant/workers/fathom-webhook/src/index.ts`:

```typescript
export interface Env {
  KV: KVNamespace;
  FATHOM_WEBHOOK_SECRET: string;
  FATHOM_API_KEY: string;
  DISCORD_TRIAGE_WEBHOOK: string;
  DISCORD_BOT_USER_ID: string;
}

export interface Invitee {
  name: string | null;
  email: string | null;
  is_external: boolean;
}

export interface Meeting {
  title: string;
  recording_id: number;
  default_summary?: { markdown_formatted?: string } | null;
  calendar_invitees: Invitee[];
}

export async function verifySignature(
  webhookId: string,
  webhookTimestamp: string,
  rawBody: string,
  signatureHeader: string,
  secret: string
): Promise<boolean> {
  const now = Math.floor(Date.now() / 1000);
  const ts = parseInt(webhookTimestamp, 10);
  if (isNaN(ts) || Math.abs(now - ts) > 300) return false;

  const rawSecret = atob(secret.replace(/^whsec_/, ""));
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(rawSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signed = `${webhookId}.${webhookTimestamp}.${rawBody}`;
  const sigBytes = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signed));
  const expected = btoa(String.fromCharCode(...new Uint8Array(sigBytes)));

  return signatureHeader.split(" ").some(sig => sig.replace(/^v1,/, "") === expected);
}

export async function checkAndSetDedup(_kv: KVNamespace, _recordingId: number): Promise<boolean> {
  throw new Error("Not implemented");
}

export async function extractSummary(
  _meeting: Pick<Meeting, "recording_id" | "default_summary">,
  _fathomApiKey: string
): Promise<string> {
  throw new Error("Not implemented");
}

export function buildDiscordMessage(
  _title: string,
  _attendees: Invitee[],
  _summary: string,
  _botUserId: string
): string {
  throw new Error("Not implemented");
}

export default {
  async fetch(_req: Request, _env: Env): Promise<Response> {
    return new Response("Not implemented", { status: 501 });
  },
};
```

Install dependencies:

```bash
cd assistant/workers/fathom-webhook && bun install
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: PASS — 4 passing tests in `verifySignature` describe block

- [ ] **Step 5: Commit**

```bash
git add assistant/workers/fathom-webhook/ && git commit -m "feat(e7): add fathom-webhook scaffold + HMAC signature verification"
```

---

### Task 2: KV-based dedup
**Group:** B (depends on Group A)

**Behavior being verified:** `checkAndSetDedup` returns `false` and writes to KV on first call for a `recording_id`, returns `true` on every subsequent call for the same id.

**Interface under test:** `checkAndSetDedup(kv: KVNamespace, recordingId: number): Promise<boolean>`

**Files:**
- Modify: `assistant/workers/fathom-webhook/src/index.ts`
- Modify: `assistant/workers/fathom-webhook/src/index.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `assistant/workers/fathom-webhook/src/index.test.ts`:

```typescript
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: FAIL — `Error: Not implemented` thrown by `checkAndSetDedup`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/workers/fathom-webhook/src/index.ts`, replace the `checkAndSetDedup` stub:

```typescript
export async function checkAndSetDedup(kv: KVNamespace, recordingId: number): Promise<boolean> {
  const key = `fathom:${recordingId}`;
  const existing = await kv.get(key);
  if (existing !== null) return true;
  await kv.put(key, "1", { expirationTtl: 86400 });
  return false;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: PASS — all 6 tests passing (4 from Task 1 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add assistant/workers/fathom-webhook/src/ && git commit -m "feat(e7): add KV-based recording dedup to fathom-webhook"
```

---

### Task 3: Summary extraction with Fathom API fallback
**Group:** C (depends on Group B)

**Behavior being verified:** `extractSummary` returns the summary from the webhook payload when present; calls `GET /recordings/{id}/summary` and returns its result when `default_summary` is absent.

**Interface under test:** `extractSummary(meeting, fathomApiKey): Promise<string>`

**Files:**
- Modify: `assistant/workers/fathom-webhook/src/index.ts`
- Modify: `assistant/workers/fathom-webhook/src/index.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `assistant/workers/fathom-webhook/src/index.test.ts`:

```typescript
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: FAIL — `Error: Not implemented` from `extractSummary` stub

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/workers/fathom-webhook/src/index.ts`, replace the `extractSummary` stub:

```typescript
export async function extractSummary(
  meeting: Pick<Meeting, "recording_id" | "default_summary">,
  fathomApiKey: string
): Promise<string> {
  if (meeting.default_summary?.markdown_formatted) {
    return meeting.default_summary.markdown_formatted;
  }
  const resp = await fetch(
    `https://api.fathom.video/recordings/${meeting.recording_id}/summary`,
    { headers: { "X-Api-Key": fathomApiKey } }
  );
  if (!resp.ok) {
    throw new Error(
      `Fathom API error ${resp.status} fetching summary for recording ${meeting.recording_id}`
    );
  }
  const data = await resp.json() as { markdown_formatted: string };
  return data.markdown_formatted;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: PASS — all 9 tests passing (6 from Tasks 1–2 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add assistant/workers/fathom-webhook/src/ && git commit -m "feat(e7): add summary extraction with Fathom API fallback"
```

---

### Task 4: Discord message formatting
**Group:** D (depends on Group C)

**Behavior being verified:** `buildDiscordMessage` produces a string containing the @mention, the `[FATHOM_MEETING]` trigger tag, the meeting title, formatted attendee names, and the summary body.

**Interface under test:** `buildDiscordMessage(title, attendees, summary, botUserId): string`

**Files:**
- Modify: `assistant/workers/fathom-webhook/src/index.ts`
- Modify: `assistant/workers/fathom-webhook/src/index.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `assistant/workers/fathom-webhook/src/index.test.ts`:

```typescript
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
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: FAIL — `Error: Not implemented` from `buildDiscordMessage` stub

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/workers/fathom-webhook/src/index.ts`, replace the `buildDiscordMessage` stub:

```typescript
export function buildDiscordMessage(
  title: string,
  attendees: Invitee[],
  summary: string,
  botUserId: string
): string {
  const attendeeStr = attendees
    .filter(a => a.email !== null)
    .map(a => (a.name !== null ? `${a.name} <${a.email}>` : a.email!))
    .join(", ");
  return [
    `<@${botUserId}> [FATHOM_MEETING]`,
    `Meeting: ${title}`,
    `Attendees: ${attendeeStr || "none"}`,
    "",
    "Summary:",
    summary,
  ].join("\n");
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: PASS — all 12 tests passing (9 from Tasks 1–3 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add assistant/workers/fathom-webhook/src/ && git commit -m "feat(e7): add Discord message formatting"
```

---

### Task 5: Full handler integration
**Group:** E (depends on Group D)

**Behavior being verified:** The `fetch` handler returns 401 for an invalid signature, 200 silently for a duplicate `recording_id`, and 200 with a Discord POST for a valid new webhook; a Discord failure propagates as a 500.

**Interface under test:** `export default { fetch }` — the CF Worker entry point via `SELF.fetch`

**Files:**
- Modify: `assistant/workers/fathom-webhook/src/index.ts`
- Modify: `assistant/workers/fathom-webhook/src/index.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `assistant/workers/fathom-webhook/src/index.test.ts`:

```typescript
import { SELF, env } from "cloudflare:test";

const TEST_SECRET = "whsec_dGVzdC1zZWNyZXQ=";

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
    const discordCalls: Request[] = [];
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (typeof url === "string" && url.includes("discord-test.invalid")) {
        discordCalls.push(new Request(url, init));
        return Promise.resolve({ ok: true });
      }
      return Promise.resolve({ ok: true, json: async () => ({ markdown_formatted: "fallback" }) });
    }));

    const body = JSON.stringify({ ...FIXTURE_MEETING, recording_id: 77703 });
    const req = await signedRequest(body, 77703);
    const resp = await SELF.fetch(req);

    expect(resp.status).toBe(200);
    expect(discordCalls.length).toBe(1);
    const posted = await discordCalls[0].json() as { content: string };
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: FAIL — `501 Not implemented` from the handler stub on valid requests

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/workers/fathom-webhook/src/index.ts`, replace the `export default { fetch }` stub:

```typescript
export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });

    const rawBody = await req.text();
    const webhookId = req.headers.get("webhook-id") ?? "";
    const webhookTimestamp = req.headers.get("webhook-timestamp") ?? "";
    const webhookSignature = req.headers.get("webhook-signature") ?? "";

    const valid = await verifySignature(
      webhookId, webhookTimestamp, rawBody, webhookSignature, env.FATHOM_WEBHOOK_SECRET
    );
    if (!valid) return new Response("Unauthorized", { status: 401 });

    const meeting: Meeting = JSON.parse(rawBody);
    const isDup = await checkAndSetDedup(env.KV, meeting.recording_id);
    if (isDup) return new Response("OK", { status: 200 });

    const summary = await extractSummary(meeting, env.FATHOM_API_KEY);
    const message = buildDiscordMessage(
      meeting.title,
      meeting.calendar_invitees,
      summary,
      env.DISCORD_BOT_USER_ID
    );

    const discordResp = await fetch(env.DISCORD_TRIAGE_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: message }),
    });
    if (!discordResp.ok) {
      throw new Error(`Discord webhook failed with ${discordResp.status}`);
    }

    return new Response("OK", { status: 200 });
  },
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/workers/fathom-webhook && bun run test
```

Expected: PASS — all 16 tests passing (12 from Tasks 1–4 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add assistant/workers/fathom-webhook/src/ && git commit -m "feat(e7): wire full fathom-webhook handler"
```

---

### Task 6: meeting-followthrough Hermes skill
**Group:** A (parallel with Task 1)

**Behavior being verified:** Mahler recognizes the `[FATHOM_MEETING]` trigger, follows the step-by-step procedure to gather CRM context, generate action items, create Notion tasks, update contacts, and post a summary. Verified manually via end-to-end test with a real Fathom recording.

**Interface under test:** `SKILL.md` trigger + procedure (no unit-testable code)

**Files:**
- Create: `assistant/config/skills/meeting-followthrough/SKILL.md`

- [ ] **Step 1: Write the skill**

Create `assistant/config/skills/meeting-followthrough/SKILL.md`:

```markdown
---
name: meeting-followthrough
description: >
  Process a completed meeting forwarded from Fathom. Gather CRM context for
  each attendee, generate smart context-aware action items using current tasks
  and priorities, push to Notion, and update last_contact in the CRM.
triggers:
  - "[FATHOM_MEETING]"
  - process fathom meeting
env:
  - CF_ACCOUNT_ID
  - CF_D1_DATABASE_ID
  - CF_API_TOKEN
  - NOTION_API_TOKEN
  - NOTION_DATABASE_ID
---

# Meeting Follow-Through

Closes the loop after any recorded meeting. Triggered automatically when the
`fathom-webhook` Cloudflare Worker posts a structured @Mahler message after a
Fathom recording completes.

## Message format (posted by CF Worker)

```
@Mahler [FATHOM_MEETING]
Meeting: {title}
Attendees: {name <email>, ...}

Summary:
{chronological_summary_markdown}
```

## Procedure

### Step 1 — Parse the message

Extract from the triggering Discord message:
- `MEETING_TITLE`: the value after `Meeting:`
- `ATTENDEES`: list of `name <email>` pairs from the `Attendees:` line
- `SUMMARY`: everything after `Summary:` to end of message

### Step 2 — Fetch CRM context for each external attendee

For each attendee in `ATTENDEES`, attempt:

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py summarize \
  --name "ATTENDEE_NAME"
```

- If the command succeeds: record the output (last contact date, context, open tasks).
- If the command fails (contact not in CRM): note "not in CRM" for that attendee and continue. Do not stop.

### Step 3 — Fetch current open tasks

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  --status "Not started"
```

Record the full output. Use it to avoid creating duplicate tasks in Step 4.

### Step 4 — Generate action items

Using all gathered context — meeting summary, CRM outputs from Step 2, open tasks from Step 3, and the injected context from project-log, kaizen priorities, and Honcho memory — reason about what action items arise from this meeting.

Rules for generating action items:
- Only create tasks for concrete commitments from the meeting (things said, agreed, or promised).
- Do not create a task if an equivalent open task already exists from Step 3.
- If the action item relates to a specific attendee who is in the CRM, prefix the task title with `[Attendee Name]` (e.g., `[Alice Chen] Send Q2 IC memo`).
- If the action item is general (not tied to a specific attendee), use no prefix.
- Default priority: Medium. Use High only for explicit deadlines or blockers mentioned in the meeting.

### Step 5 — Create Notion tasks

For each action item determined in Step 4:

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py create \
  --title "TASK_TITLE" \
  --priority PRIORITY
```

- If `tasks.py create` fails for any task: surface the error immediately in Discord and stop creating further tasks. Do not silently skip.
- Record each created task's title for the summary in Step 7.

### Step 6 — Update CRM last_contact

For each attendee from Step 2 whose `contacts.py summarize` succeeded:

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py talked-to \
  --name "ATTENDEE_NAME"
```

- If `contacts.py talked-to` fails: surface the error in Discord but continue to Step 7 (CRM update failure must not block the summary).

### Step 7 — Post summary to Discord

Post a single Discord message with:
- **Meeting:** `MEETING_TITLE`
- **Action items created:** bulleted list of task titles from Step 5, or "None" if no action items were generated.
- **CRM updated:** comma-separated names of contacts whose `last_contact` was updated, or "No CRM matches" if none.

Example output:
```
Post-meeting: 1:1 with Alice Chen
Action items created:
  · [Alice Chen] Send Q2 IC memo
  · [Alice Chen] Intro to Marcus at Benchmark
  · Follow up on Series A timeline
CRM updated: Alice Chen
```

If no action items were generated, say so explicitly rather than posting nothing.

## Failure modes

- `contacts.py summarize` fails → note "not in CRM", continue (non-fatal)
- `tasks.py list` fails → surface error and stop (cannot safely generate without knowing existing tasks)
- `tasks.py create` fails → surface error and stop (partial task list is worse than none)
- `contacts.py talked-to` fails → surface error, continue to Step 7
```

- [ ] **Step 2: Verify it FAILS (manual check)**

Confirm the skill directory does not yet exist in the deployed Hermes container. This is verified during the end-to-end test after deployment.

- [ ] **Step 3: No additional implementation** — the SKILL.md is the implementation.

- [ ] **Step 4: Verify it PASSES — manual end-to-end test**

After deploying (see post-plan deployment steps), record a 2-minute Fathom test meeting. Confirm:
1. CF Worker receives the Fathom webhook and posts a `[FATHOM_MEETING]` @Mahler message to Discord.
2. Mahler responds with action items created in Notion and CRM updated.
3. Notion tasks exist with correct titles and prefixes.
4. `contacts.py summarize` for the attendee shows today's `last_contact` date.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/meeting-followthrough/ && git commit -m "feat(e7): add meeting-followthrough Hermes skill"
```

---

## Post-plan deployment steps (run after all tasks pass)

These steps are not part of the build but must be completed before the system is live:

**1. Deploy the CF Worker:**
```bash
cd assistant/workers/fathom-webhook
wrangler secret put FATHOM_WEBHOOK_SECRET   # paste value from Fathom dashboard
wrangler secret put FATHOM_API_KEY           # paste FATHOM_API_KEY
wrangler secret put DISCORD_TRIAGE_WEBHOOK   # paste from Fly secrets
wrangler secret put DISCORD_BOT_USER_ID      # Mahler bot's Discord user ID
wrangler deploy
```

**2. Get the Worker URL** from the deploy output (e.g., `https://fathom-webhook.<account>.workers.dev`).

**3. Register the webhook with Fathom:**
```bash
curl -X POST https://api.fathom.video/webhooks \
  -H "X-Api-Key: $FATHOM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "destination_url": "https://fathom-webhook.<account>.workers.dev/",
    "include_summary": true,
    "include_action_items": false,
    "triggered_for": ["my_recordings"]
  }'
```

Save the returned `secret` value — it must match `FATHOM_WEBHOOK_SECRET`.

**4. Deploy Hermes with the new skill:**
```bash
cd assistant && flyctl deploy --remote-only
```

**5. Add `DISCORD_BOT_USER_ID` to `.env.example`:**
```
DISCORD_BOT_USER_ID=   # Mahler Discord bot user ID (used by fathom-webhook CF Worker)
```
