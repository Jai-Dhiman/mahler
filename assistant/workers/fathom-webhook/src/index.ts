export interface Env {
  KV: KVNamespace;
  DB: D1Database;
  FATHOM_WEBHOOK_SECRET: string;
  FATHOM_API_KEY: string;
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

  const secretBytes = Uint8Array.from(atob(secret.replace(/^whsec_/, "")), c => c.charCodeAt(0));
  const key = await crypto.subtle.importKey(
    "raw",
    secretBytes,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"]
  );
  const signed = `${webhookId}.${webhookTimestamp}.${rawBody}`;
  for (const sigStr of signatureHeader.split(" ")) {
    const b64 = sigStr.replace(/^v1,/, "");
    if (!b64) continue;
    try {
      const incomingBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const valid = await crypto.subtle.verify("HMAC", key, incomingBytes, new TextEncoder().encode(signed));
      if (valid) return true;
    } catch {
      continue;
    }
  }
  return false;
}

export async function checkAndSetDedup(kv: KVNamespace, recordingId: number): Promise<boolean> {
  const key = `fathom:${recordingId}`;
  const existing = await kv.get(key);
  if (existing !== null) return true;
  await kv.put(key, "1", { expirationTtl: 86400 });
  return false;
}

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

export async function enqueueToD1(
  db: D1Database,
  meeting: Meeting,
  summary: string
): Promise<boolean> {
  await db.prepare(`CREATE TABLE IF NOT EXISTS fathom_meeting_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL UNIQUE,
    title TEXT NOT NULL,
    attendees TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT
  )`).run();

  const attendees = JSON.stringify(meeting.calendar_invitees ?? []);
  const result = await db.prepare(
    "INSERT OR IGNORE INTO fathom_meeting_queue (recording_id, title, attendees, summary) VALUES (?, ?, ?, ?)"
  ).bind(meeting.recording_id, meeting.title, attendees, summary).run();

  return (result.meta?.changes ?? 0) > 0;
}

function sanitize(s: string): string {
  return s.replace(/[\r\n\t]/g, " ").slice(0, 200);
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });

    const rawBody = await req.text();
    const webhookId = req.headers.get("webhook-id") ?? "";
    const webhookTimestamp = req.headers.get("webhook-timestamp") ?? "";
    const webhookSignature = req.headers.get("webhook-signature") ?? "";

    console.log(`[fathom-webhook] POST received, id=${sanitize(webhookId)}, ts=${sanitize(webhookTimestamp)}`);

    const valid = await verifySignature(
      webhookId, webhookTimestamp, rawBody, webhookSignature, env.FATHOM_WEBHOOK_SECRET
    );
    if (!valid) {
      console.log("[fathom-webhook] Signature verification FAILED");
      return new Response("Unauthorized", { status: 401 });
    }
    console.log("[fathom-webhook] Signature OK");

    let meeting: Meeting;
    try {
      meeting = JSON.parse(rawBody) as Meeting;
    } catch {
      console.log("[fathom-webhook] JSON parse failed");
      return new Response("Bad Request", { status: 400 });
    }
    console.log(`[fathom-webhook] recording_id=${meeting.recording_id} title="${sanitize(meeting.title)}"`);

    const isDup = await checkAndSetDedup(env.KV, meeting.recording_id);
    if (isDup) {
      console.log(`[fathom-webhook] KV dedup hit for recording_id=${meeting.recording_id}`);
      return new Response("OK", { status: 200 });
    }

    let summary: string;
    try {
      summary = await extractSummary(meeting, env.FATHOM_API_KEY);
      console.log(`[fathom-webhook] summary fetched, length=${summary.length}`);
    } catch (err) {
      console.log(`[fathom-webhook] summary fetch FAILED: ${(err as Error).message}`);
      return new Response(`Failed to fetch summary: ${(err as Error).message}`, { status: 500 });
    }

    try {
      await enqueueToD1(env.DB, meeting, summary);
      console.log(`[fathom-webhook] enqueued recording_id=${meeting.recording_id} to D1`);
    } catch (err) {
      console.log(`[fathom-webhook] D1 enqueue FAILED: ${(err as Error).message}`);
      return new Response(`D1 enqueue failed: ${(err as Error).message}`, { status: 500 });
    }

    return new Response("OK", { status: 200 });
  },
};
