import { SELF } from "cloudflare:test";
import { describe, expect, it } from "vitest";

describe("GET /health", () => {
  it("returns ok payload identifying the service", async () => {
    const res = await SELF.fetch("https://finance.test/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, service: "finance-state" });
  });
});
