import { env } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getPaperEquity } from "../../src/alpaca/client";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getPaperEquity", () => {
  it("calls alpaca paper account endpoint and returns equity as number", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toBe("https://paper-api.alpaca.markets/v2/account");
      const headers = new Headers(init?.headers);
      expect(headers.get("APCA-API-KEY-ID")).toBe("test-alpaca-key");
      expect(headers.get("APCA-API-SECRET-KEY")).toBe("test-alpaca-secret");
      return new Response(JSON.stringify({ equity: "10250.42", cash: "5000.00" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });

    const equity = await getPaperEquity(env);
    expect(equity).toBe(10250.42);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("throws on non-2xx response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("forbidden", { status: 403 }));
    await expect(getPaperEquity(env)).rejects.toThrow(/alpaca/i);
  });
});
