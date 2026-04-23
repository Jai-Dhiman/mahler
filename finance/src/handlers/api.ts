import { requireBearer, UnauthorizedError } from "../auth";
import { getHistory, getLatestSnapshots } from "../db/queries";
import { syncAllItems } from "../plaid/sync";
import { computeWeeklySummary } from "../summary/compute";
import type { Env } from "../types";

function unauthorized(): Response {
  return Response.json({ error: "unauthorized" }, { status: 401 });
}

function isoToday(): string {
  return new Date().toISOString().slice(0, 10);
}

export async function handleApi(req: Request, env: Env): Promise<Response> {
  try {
    requireBearer(req, env);
  } catch (err) {
    if (err instanceof UnauthorizedError) return unauthorized();
    throw err;
  }

  const url = new URL(req.url);
  if (req.method === "GET" && url.pathname === "/balances") {
    const snapshots = await getLatestSnapshots(env);
    return Response.json({ snapshots });
  }
  if (req.method === "GET" && url.pathname === "/networth") {
    const data = await computeWeeklySummary(env, new Date());
    return Response.json(data);
  }
  if (req.method === "GET" && url.pathname === "/history") {
    const accountId = url.searchParams.get("account_id");
    const daysRaw = url.searchParams.get("days") ?? "30";
    if (!accountId) {
      return Response.json({ error: "account_id required" }, { status: 400 });
    }
    const days = Number.parseInt(daysRaw, 10);
    if (!Number.isFinite(days) || days <= 0 || days > 365) {
      return Response.json({ error: "days must be 1..365" }, { status: 400 });
    }
    const history = await getHistory(env, accountId, days);
    return Response.json({ history });
  }
  if (req.method === "POST" && url.pathname === "/refresh") {
    const result = await syncAllItems(env, isoToday());
    return Response.json(result);
  }
  return new Response("not found", { status: 404 });
}
