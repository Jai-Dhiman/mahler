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
      meeting.calendar_invitees ?? [],
      summary,
      env.DISCORD_BOT_USER_ID
    );

    const discordResp = await fetch(env.DISCORD_TRIAGE_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: message }),
    });
    if (!discordResp.ok) {
      return new Response(`Discord webhook failed with ${discordResp.status}`, { status: 500 });
    }

    return new Response("OK", { status: 200 });
  },
};
