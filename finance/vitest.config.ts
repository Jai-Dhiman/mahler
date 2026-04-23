import { readD1Migrations } from "@cloudflare/vitest-pool-workers/config";
import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig(async () => {
  const migrations = await readD1Migrations("migrations");
  return {
    test: {
      setupFiles: ["./test/setup.ts"],
      poolOptions: {
        workers: {
          wrangler: { configPath: "./wrangler.toml" },
          miniflare: {
            d1Databases: ["DB"],
            kvNamespaces: ["FINANCE_KV"],
            bindings: {
              ENVIRONMENT: "test",
              BEARER_TOKEN: "test-token",
              PLAID_CLIENT_ID: "test-client",
              PLAID_SECRET_DEV: "test-secret",
              PLAID_WEBHOOK_SECRET: "test-webhook-secret",
              DISCORD_WEBHOOK_URL: "https://discord.test/webhook",
              ALPACA_PAPER_KEY_ID: "test-alpaca-key",
              ALPACA_PAPER_SECRET: "test-alpaca-secret",
              DB_MIGRATIONS: JSON.stringify(migrations),
            },
          },
        },
      },
    },
  };
});
