import { logEvent, upsertItem } from "../db/queries";
import { exchangePublicToken } from "../plaid/client";
import type { Env } from "../types";

const LINK_HTML = `<!doctype html>
<html><head><meta charset="utf-8"><title>Mahler Finance Link</title></head>
<body style="font-family: system-ui; max-width: 600px; margin: 4rem auto;">
  <h1>Plaid Link</h1>
  <p>Pick an institution to link. After completing Plaid Link, the access token is stored in KV automatically.</p>
  <input id="institution" placeholder="Institution name (e.g. Wells Fargo)" style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;" />
  <button id="open" style="padding: 0.5rem 1rem;">Open Plaid Link</button>
  <pre id="out" style="margin-top: 1rem; background: #f5f5f5; padding: 1rem;"></pre>
  <script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
  <script>
    async function start() {
      const tokenRes = await fetch('/link/token', { method: 'POST' });
      const { link_token } = await tokenRes.json();
      const handler = Plaid.create({
        token: link_token,
        onSuccess: async (public_token) => {
          const institution = document.getElementById('institution').value || 'Unknown';
          const r = await fetch('/link/exchange', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ public_token, institution_name: institution }),
          });
          document.getElementById('out').textContent = await r.text();
        },
      });
      handler.open();
    }
    document.getElementById('open').addEventListener('click', start);
  </script>
</body></html>`;

export async function handleLink(req: Request, env: Env): Promise<Response> {
  if (env.ENVIRONMENT === "production") {
    return new Response("not found", { status: 404 });
  }

  const url = new URL(req.url);
  if (req.method === "GET" && url.pathname === "/link") {
    return new Response(LINK_HTML, {
      status: 200,
      headers: { "content-type": "text/html; charset=utf-8" },
    });
  }
  if (req.method === "POST" && url.pathname === "/link/token") {
    const { createLinkToken } = await import("../plaid/client");
    const tok = await createLinkToken(env);
    return Response.json(tok);
  }
  if (req.method === "POST" && url.pathname === "/link/exchange") {
    const body = (await req.json()) as { public_token: string; institution_name: string };
    if (!body.public_token || !body.institution_name) {
      return Response.json({ error: "public_token and institution_name required" }, { status: 400 });
    }
    const { item_id } = await exchangePublicToken(env, body.public_token);
    await upsertItem(env, { item_id, institution_name: body.institution_name });
    await logEvent(env, {
      event_type: "plaid_link_exchanged",
      item_id,
      account_id: null,
      payload: { institution_name: body.institution_name },
    });
    return Response.json({ item_id });
  }
  return new Response("not found", { status: 404 });
}
