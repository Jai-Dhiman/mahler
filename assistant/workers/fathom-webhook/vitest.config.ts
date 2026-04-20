import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    poolOptions: {
      workers: {
        wrangler: { configPath: "./wrangler.toml" },
        miniflare: {
          bindings: {
            FATHOM_WEBHOOK_SECRET: "whsec_dGVzdC1zZWNyZXQ=",
            FATHOM_API_KEY: "test-fathom-api-key",
            DISCORD_TRIAGE_WEBHOOK: "https://discord-test.invalid/webhook",
            DISCORD_BOT_USER_ID: "123456789012345678",
          },
        },
      },
    },
  },
});
