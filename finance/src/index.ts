import { handleApi } from "./handlers/api";
import type { Env } from "./types";

const API_PATHS = new Set(["/balances", "/networth", "/history", "/refresh"]);

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "finance-state" });
    }
    if (API_PATHS.has(url.pathname)) {
      return handleApi(req, env);
    }
    return new Response("not found", { status: 404 });
  },

  async scheduled(_event: ScheduledEvent, _env: Env, _ctx: ExecutionContext): Promise<void> {
    // wired in Task 14
  },
} satisfies ExportedHandler<Env>;
