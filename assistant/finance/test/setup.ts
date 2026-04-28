import { applyD1Migrations, env } from "cloudflare:test";
import { beforeAll } from "vitest";

beforeAll(async () => {
  const migrations = JSON.parse((env as unknown as { DB_MIGRATIONS: string }).DB_MIGRATIONS);
  await applyD1Migrations(env.DB, migrations);
});
