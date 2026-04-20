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
