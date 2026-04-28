import type { Env } from "./types";

export class UnauthorizedError extends Error {
  constructor() {
    super("unauthorized");
  }
}

export function requireBearer(req: Request, env: Env): void {
  const header = req.headers.get("authorization");
  if (!header || !header.startsWith("Bearer ")) throw new UnauthorizedError();
  const token = header.slice("Bearer ".length).trim();
  if (token !== env.BEARER_TOKEN) throw new UnauthorizedError();
}
