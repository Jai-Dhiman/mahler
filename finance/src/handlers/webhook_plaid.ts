import { listItems, logEvent, updateItemStatus } from "../db/queries";
import { verifyWebhook } from "../plaid/client";
import type { Env } from "../types";

interface PlaidWebhookBody {
  webhook_type: string;
  webhook_code: string;
  item_id: string;
  error?: { error_code?: string };
}

const REAUTH_CODES = new Set(["ERROR", "PENDING_EXPIRATION", "USER_PERMISSION_REVOKED"]);

export async function handlePlaidWebhook(req: Request, env: Env): Promise<Response> {
  const raw = await req.text();
  if (!await verifyWebhook(env, raw, req.headers)) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  const body = JSON.parse(raw) as PlaidWebhookBody;
  if (body.webhook_type !== "ITEM" || !REAUTH_CODES.has(body.webhook_code)) {
    return Response.json({ ok: true, ignored: true });
  }

  const code = body.error?.error_code ?? body.webhook_code;
  await updateItemStatus(env, body.item_id, "needs_reauth", code);
  await logEvent(env, {
    event_type: "reauth_needed",
    item_id: body.item_id,
    account_id: null,
    payload: body,
  });

  const items = await listItems(env);
  const item = items.find((i) => i.item_id === body.item_id);
  const institution = item?.institution_name ?? body.item_id;
  const discordRes = await fetch(env.DISCORD_WEBHOOK_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      content: `Plaid item for **${institution}** needs re-auth (${code}). Run \`wrangler dev\` and redo \`/link\`.`,
    }),
  });
  if (!discordRes.ok && discordRes.status !== 204) {
    throw new Error(`discord webhook failed: ${discordRes.status} ${await discordRes.text()}`);
  }

  return Response.json({ ok: true });
}
