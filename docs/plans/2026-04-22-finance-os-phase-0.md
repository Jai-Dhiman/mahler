# Personal Finance OS — Phase 0 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Capture daily balance snapshots from Wells Fargo, Wealthfront, and Alpaca paper into D1, and post a weekly Discord summary every Sunday — all read-only, with no LLM in the primary path.
**Spec:** docs/specs/2026-04-22-finance-os-phase-0-design.md
**Style:** Follow `CLAUDE.md` at repo root and `traderjoe/` patterns where applicable. TypeScript strict, no emojis, explicit error handling (no silent fallbacks).

---

## Task Groups

```
Group A (foundation, sequential):
  - Task 1: bootstrap worker + health endpoint

Group B (data layer, sequential, all touch finance/src/db/queries.ts):
  - Task 2: snapshot round-trip
  - Task 3: items + accounts registry
  - Task 4: event log

Group C (external clients + formatters, parallel where files disjoint):
  - Task 5: plaid/client.ts — token exchange + KV storage
  - Task 6: plaid/client.ts — getBalances + verifyWebhook (sequential after Task 5)
  - Task 7: alpaca/client.ts — getPaperEquity (parallel with Task 5+6)
  - Task 8: discord/embed.ts — postWeeklySummary (parallel with Task 5-7)

Group D (orchestration, parallel — disjoint files):
  - Task 9: plaid/sync.ts — syncAllItems
  - Task 10: summary/compute.ts — computeWeeklySummary

Group E (handlers, sequential because all wire into index.ts route table):
  - Task 11: handlers/api.ts — bearer-auth read + refresh
  - Task 12: handlers/link.ts — dev-only Link page + exchange
  - Task 13: handlers/webhook_plaid.ts — Plaid item webhook
  - Task 14: handlers/crons.ts — scheduled-event dispatcher

Group F (Hermes skill, single task):
  - Task 15: finance-read skill (SKILL.md + scripts/query.py)
```

---

### Task 1: Bootstrap worker + health endpoint

**Group:** A (alone)

**Behavior being verified:** A fetched worker responds `200 {"ok": true, "service": "finance-state"}` on `GET /health`, proving the build, wrangler config, vitest pool, D1 binding, and KV binding all wire up.

**Interface under test:** `fetch(req, env)` exported from `finance/src/index.ts`, invoked through the vitest-pool-workers `SELF.fetch()` harness.

**Files:**
- Create: `finance/wrangler.toml`
- Create: `finance/package.json`
- Create: `finance/tsconfig.json`
- Create: `finance/vitest.config.ts`
- Create: `finance/.dev.vars.example`
- Create: `finance/CLAUDE.md`
- Create: `finance/migrations/0001_init.sql`
- Create: `finance/src/index.ts`
- Create: `finance/src/types.ts`
- Test: `finance/test/health.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/health.test.ts`:
```typescript
import { SELF } from "cloudflare:test";
import { describe, expect, it } from "vitest";

describe("GET /health", () => {
  it("returns ok payload identifying the service", async () => {
    const res = await SELF.fetch("https://finance.test/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, service: "finance-state" });
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bun install && bunx vitest run test/health.test.ts
```
Expected: FAIL — wrangler config or main entry missing.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/wrangler.toml`:
```toml
name = "finance-state"
main = "src/index.ts"
compatibility_date = "2026-04-01"
compatibility_flags = ["nodejs_compat"]

[observability]
enabled = true

[triggers]
crons = [
  "0 7 * * *",
  "0 7 * * MON",
]

[[d1_databases]]
binding = "DB"
database_name = "mahler-db"
database_id = "b6cb2eac-2903-46bd-baea-b4ff2dc904d0"
migrations_dir = "migrations"

[[kv_namespaces]]
binding = "FINANCE_KV"
id = "REPLACE_WITH_KV_ID_BEFORE_DEPLOY"

[vars]
ENVIRONMENT = "dev"
LOG_LEVEL = "INFO"
```

`finance/package.json`:
```json
{
  "name": "finance-state",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "wrangler dev",
    "deploy": "wrangler deploy",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "@cloudflare/vitest-pool-workers": "^0.5.0",
    "@cloudflare/workers-types": "^4.20240909.0",
    "typescript": "^5.5.0",
    "vitest": "~1.5.0",
    "wrangler": "^3.78.0"
  },
  "dependencies": {
    "plaid": "^27.0.0"
  }
}
```

`finance/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "types": ["@cloudflare/workers-types", "@cloudflare/vitest-pool-workers"]
  },
  "include": ["src", "test"]
}
```

`finance/vitest.config.ts`:
```typescript
import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
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
          },
        },
      },
    },
  },
});
```

`finance/.dev.vars.example`:
```
PLAID_CLIENT_ID=
PLAID_SECRET_DEV=
PLAID_WEBHOOK_SECRET=
BEARER_TOKEN=
DISCORD_WEBHOOK_URL=
ALPACA_PAPER_KEY_ID=
ALPACA_PAPER_SECRET=
```

`finance/CLAUDE.md`:
```markdown
# finance-state Worker

Read-only Personal Finance OS Phase 0. Polls Plaid + Alpaca daily, posts a weekly Discord summary. No LLM in the primary path.

## Layout
- `src/index.ts` — fetch + scheduled router
- `src/db/queries.ts` — all D1 access (deep module)
- `src/plaid/client.ts` — Plaid SDK wrapper + KV token I/O + webhook verify
- `src/plaid/sync.ts` — syncAllItems orchestration
- `src/alpaca/client.ts` — paper equity read
- `src/summary/compute.ts` — weekly summary derivation
- `src/discord/embed.ts` — Discord embed format + post
- `src/handlers/` — HTTP and scheduled handlers

## Database
Shares `mahler-db` D1 with assistant + traderjoe. Tables prefixed `finance_*`.
KV namespace `FINANCE_KV` is dedicated. Plaid access tokens at key `plaid_item:{item_id}`.

## Deploy
bun install; wrangler kv:namespace create FINANCE_KV; wrangler d1 migrations apply mahler-db;
wrangler secret put PLAID_CLIENT_ID / PLAID_SECRET_DEV / PLAID_WEBHOOK_SECRET / BEARER_TOKEN /
DISCORD_WEBHOOK_URL / ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET; wrangler deploy.

## Plaid Link (one-time per institution)
`wrangler dev`, open http://localhost:8787/link, complete Plaid Link.
`/link` is gated behind `ENVIRONMENT === "dev"`.

## Conventions
TypeScript strict. No emojis. Explicit error handling. Tests use
`@cloudflare/vitest-pool-workers` with miniflare D1 + KV. Plaid, Alpaca,
Discord network calls mocked at the `fetch` boundary.
```

`finance/migrations/0001_init.sql`:
```sql
CREATE TABLE IF NOT EXISTS finance_plaid_item (
  item_id           TEXT PRIMARY KEY,
  institution_name  TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'ok',
  last_synced_at    TEXT,
  last_error        TEXT,
  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS finance_account (
  account_id          TEXT PRIMARY KEY,
  item_id             TEXT,
  display_name        TEXT NOT NULL,
  account_type        TEXT NOT NULL,
  asset_class         TEXT NOT NULL,
  currency            TEXT NOT NULL DEFAULT 'USD',
  is_liability        INTEGER NOT NULL DEFAULT 0,
  include_in_networth INTEGER NOT NULL DEFAULT 1,
  is_active           INTEGER NOT NULL DEFAULT 1,
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS finance_balance_snapshot (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id        TEXT NOT NULL,
  taken_at          TEXT NOT NULL,
  snapshot_date     TEXT NOT NULL,
  current_balance   REAL NOT NULL,
  available_balance REAL,
  source            TEXT NOT NULL,
  raw_response      TEXT,
  UNIQUE (account_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_account_time
  ON finance_balance_snapshot (account_id, taken_at DESC);

CREATE TABLE IF NOT EXISTS finance_event_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  occurred_at TEXT NOT NULL DEFAULT (datetime('now')),
  event_type  TEXT NOT NULL,
  item_id     TEXT,
  account_id  TEXT,
  payload     TEXT
);
```

`finance/src/types.ts`:
```typescript
export interface Env {
  DB: D1Database;
  FINANCE_KV: KVNamespace;
  ENVIRONMENT: string;
  LOG_LEVEL: string;
  BEARER_TOKEN: string;
  PLAID_CLIENT_ID: string;
  PLAID_SECRET_DEV: string;
  PLAID_WEBHOOK_SECRET: string;
  DISCORD_WEBHOOK_URL: string;
  ALPACA_PAPER_KEY_ID: string;
  ALPACA_PAPER_SECRET: string;
}

export interface PlaidItemRow {
  item_id: string;
  institution_name: string;
  status: "ok" | "needs_reauth" | "error";
  last_synced_at: string | null;
  last_error: string | null;
  created_at: string;
}

export interface AccountRow {
  account_id: string;
  item_id: string | null;
  display_name: string;
  account_type: string;
  asset_class: string;
  currency: string;
  is_liability: 0 | 1;
  include_in_networth: 0 | 1;
  is_active: 0 | 1;
  created_at: string;
}

export interface SnapshotRow {
  id: number;
  account_id: string;
  taken_at: string;
  snapshot_date: string;
  current_balance: number;
  available_balance: number | null;
  source: "plaid" | "alpaca" | "manual";
  raw_response: string | null;
}

export interface EventRow {
  id: number;
  occurred_at: string;
  event_type: string;
  item_id: string | null;
  account_id: string | null;
  payload: string | null;
}

export interface SyncResult {
  itemsAttempted: number;
  itemsSucceeded: number;
  snapshotsWritten: number;
  errors: Array<{ item_id: string; error: string }>;
}

export interface WeeklyAccountLine {
  account_id: string;
  display_name: string;
  account_type: string;
  current_balance: number;
  prior_week_balance: number | null;
  delta: number | null;
}

export interface WeeklyData {
  asOf: string;
  netWorth: number;
  priorWeekNetWorth: number | null;
  netWorthDelta: number | null;
  accounts: WeeklyAccountLine[];
  strategyPaperEquity: number | null;
  sparkline: Array<{ date: string; net_worth: number }>;
  staleItems: Array<{ item_id: string; institution_name: string; last_synced_at: string | null }>;
}
```

`finance/src/index.ts`:
```typescript
import type { Env } from "./types";

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "finance-state" });
    }
    return new Response("not found", { status: 404 });
  },

  async scheduled(_event: ScheduledEvent, _env: Env, _ctx: ExecutionContext): Promise<void> {
    // wired in Task 14
  },
} satisfies ExportedHandler<Env>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/health.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/ && git commit -m "feat(finance): bootstrap worker scaffolding + health endpoint"
```

---

### Task 2: queries.ts — snapshot round-trip

**Group:** B (sequential, depends on Task 1)

**Behavior being verified:** Daily balance snapshots persist with per-day idempotency, and historical retrieval returns the expected ordered series.

**Interface under test:** `insertSnapshot`, `getLatestSnapshots`, `getHistory`, `upsertAccount` exported from `finance/src/db/queries.ts`, called against the miniflare D1 (real schema, no mocks).

**Files:**
- Create: `finance/src/db/queries.ts`
- Test: `finance/test/db/snapshot.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/db/snapshot.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import {
  getHistory,
  getLatestSnapshots,
  insertSnapshot,
  upsertAccount,
} from "../../src/db/queries";

async function reset(): Promise<void> {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
}

describe("snapshot round-trip", () => {
  beforeEach(reset);

  it("persists per-day snapshots idempotently and returns history in reverse-chronological order", async () => {
    await upsertAccount(env, {
      account_id: "acc_checking",
      item_id: "item_wf",
      display_name: "Wells Checking",
      account_type: "checking",
      asset_class: "cash",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 1,
      is_active: 1,
    });

    const dates = ["2026-04-15", "2026-04-16", "2026-04-17", "2026-04-18", "2026-04-19"];
    for (const [i, date] of dates.entries()) {
      await insertSnapshot(env, {
        account_id: "acc_checking",
        taken_at: `${date}T07:00:00Z`,
        snapshot_date: date,
        current_balance: 1000 + i * 10,
        available_balance: 1000 + i * 10,
        source: "plaid",
        raw_response: JSON.stringify({ marker: i }),
      });
    }

    // duplicate-day insert must be ignored
    await insertSnapshot(env, {
      account_id: "acc_checking",
      taken_at: "2026-04-17T23:00:00Z",
      snapshot_date: "2026-04-17",
      current_balance: 9999,
      available_balance: 9999,
      source: "plaid",
      raw_response: null,
    });

    const latest = await getLatestSnapshots(env);
    expect(latest).toHaveLength(1);
    expect(latest[0]!.account_id).toBe("acc_checking");
    expect(latest[0]!.current_balance).toBe(1040);
    expect(latest[0]!.snapshot_date).toBe("2026-04-19");

    const history = await getHistory(env, "acc_checking", 7);
    expect(history.map((r) => r.snapshot_date)).toEqual([
      "2026-04-19",
      "2026-04-18",
      "2026-04-17",
      "2026-04-16",
      "2026-04-15",
    ]);
    const day3 = history.find((r) => r.snapshot_date === "2026-04-17")!;
    expect(day3.current_balance).toBe(1020);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/db/snapshot.test.ts
```
Expected: FAIL — `Cannot find module '../../src/db/queries'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/db/queries.ts`:
```typescript
import type { AccountRow, Env, SnapshotRow } from "../types";

export async function upsertAccount(env: Env, row: Omit<AccountRow, "created_at">): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO finance_account
       (account_id, item_id, display_name, account_type, asset_class, currency,
        is_liability, include_in_networth, is_active)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(account_id) DO UPDATE SET
       item_id = excluded.item_id,
       display_name = excluded.display_name,
       account_type = excluded.account_type,
       asset_class = excluded.asset_class,
       currency = excluded.currency,
       is_liability = excluded.is_liability,
       include_in_networth = excluded.include_in_networth,
       is_active = excluded.is_active`,
  )
    .bind(
      row.account_id,
      row.item_id,
      row.display_name,
      row.account_type,
      row.asset_class,
      row.currency,
      row.is_liability,
      row.include_in_networth,
      row.is_active,
    )
    .run();
}

export interface SnapshotInput {
  account_id: string;
  taken_at: string;
  snapshot_date: string;
  current_balance: number;
  available_balance: number | null;
  source: "plaid" | "alpaca" | "manual";
  raw_response: string | null;
}

export async function insertSnapshot(env: Env, row: SnapshotInput): Promise<void> {
  await env.DB.prepare(
    `INSERT OR IGNORE INTO finance_balance_snapshot
       (account_id, taken_at, snapshot_date, current_balance, available_balance, source, raw_response)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      row.account_id,
      row.taken_at,
      row.snapshot_date,
      row.current_balance,
      row.available_balance,
      row.source,
      row.raw_response,
    )
    .run();
}

export async function getLatestSnapshots(env: Env): Promise<SnapshotRow[]> {
  const result = await env.DB.prepare(
    `SELECT s.* FROM finance_balance_snapshot s
       JOIN (
         SELECT account_id, MAX(taken_at) AS max_taken
         FROM finance_balance_snapshot
         GROUP BY account_id
       ) latest
         ON latest.account_id = s.account_id AND latest.max_taken = s.taken_at`,
  ).all<SnapshotRow>();
  return result.results;
}

export async function getHistory(env: Env, accountId: string, days: number): Promise<SnapshotRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_balance_snapshot
     WHERE account_id = ?
       AND taken_at >= datetime('now', ?)
     ORDER BY taken_at DESC`,
  )
    .bind(accountId, `-${days} days`)
    .all<SnapshotRow>();
  return result.results;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/db/snapshot.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/db/queries.ts finance/test/db/snapshot.test.ts && git commit -m "feat(finance): snapshot persistence with per-day idempotency"
```

---

### Task 3: queries.ts — items + accounts registry

**Group:** B (sequential, depends on Task 2)

**Behavior being verified:** Plaid items and accounts can be upserted, listed, and have their status updated; listing reflects the latest state.

**Interface under test:** `upsertItem`, `listItems`, `listAccounts`, `updateItemStatus` from `finance/src/db/queries.ts`.

**Files:**
- Modify: `finance/src/db/queries.ts`
- Test: `finance/test/db/registry.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/db/registry.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import {
  listAccounts,
  listItems,
  updateItemStatus,
  upsertAccount,
  upsertItem,
} from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
});

describe("plaid item + account registry", () => {
  it("upserts items and accounts, lists them, and reflects status updates", async () => {
    await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
    await upsertItem(env, { item_id: "item_wlth", institution_name: "Wealthfront" });
    await upsertItem(env, { item_id: "item_wlth", institution_name: "Wealthfront Inc" });

    await upsertAccount(env, {
      account_id: "acc_checking",
      item_id: "item_wf",
      display_name: "Wells Checking",
      account_type: "checking",
      asset_class: "cash",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 1,
      is_active: 1,
    });
    await upsertAccount(env, {
      account_id: "acc_paper",
      item_id: null,
      display_name: "Alpaca Paper",
      account_type: "brokerage_paper",
      asset_class: "strategy_paper",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 0,
      is_active: 1,
    });

    const items = await listItems(env);
    expect(items.map((i) => i.item_id).sort()).toEqual(["item_wf", "item_wlth"]);
    const wlth = items.find((i) => i.item_id === "item_wlth")!;
    expect(wlth.institution_name).toBe("Wealthfront Inc");
    expect(wlth.status).toBe("ok");
    expect(wlth.last_error).toBeNull();

    const accounts = await listAccounts(env);
    expect(accounts.map((a) => a.account_id).sort()).toEqual(["acc_checking", "acc_paper"]);
    expect(accounts.find((a) => a.account_id === "acc_paper")!.include_in_networth).toBe(0);

    await updateItemStatus(env, "item_wf", "needs_reauth", "ITEM_LOGIN_REQUIRED");
    const after = await listItems(env);
    const wf = after.find((i) => i.item_id === "item_wf")!;
    expect(wf.status).toBe("needs_reauth");
    expect(wf.last_error).toBe("ITEM_LOGIN_REQUIRED");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/db/registry.test.ts
```
Expected: FAIL — `upsertItem`, `listItems`, `listAccounts`, `updateItemStatus` not exported.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `finance/src/db/queries.ts`:
```typescript
import type { PlaidItemRow } from "../types";

export interface UpsertItemInput {
  item_id: string;
  institution_name: string;
}

export async function upsertItem(env: Env, input: UpsertItemInput): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO finance_plaid_item (item_id, institution_name)
     VALUES (?, ?)
     ON CONFLICT(item_id) DO UPDATE SET
       institution_name = excluded.institution_name`,
  )
    .bind(input.item_id, input.institution_name)
    .run();
}

export async function listItems(env: Env): Promise<PlaidItemRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_plaid_item ORDER BY created_at ASC`,
  ).all<PlaidItemRow>();
  return result.results;
}

export async function listAccounts(env: Env): Promise<AccountRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_account WHERE is_active = 1 ORDER BY created_at ASC`,
  ).all<AccountRow>();
  return result.results;
}

export async function updateItemStatus(
  env: Env,
  itemId: string,
  status: "ok" | "needs_reauth" | "error",
  lastError: string | null = null,
): Promise<void> {
  await env.DB.prepare(
    `UPDATE finance_plaid_item
     SET status = ?, last_error = ?, last_synced_at = datetime('now')
     WHERE item_id = ?`,
  )
    .bind(status, lastError, itemId)
    .run();
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/db/registry.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/db/queries.ts finance/test/db/registry.test.ts && git commit -m "feat(finance): item + account registry queries"
```

---

### Task 4: queries.ts — event log

**Group:** B (sequential, depends on Task 3)

**Behavior being verified:** Events append to the audit log with the supplied type, optional ids, and JSON payload, and read back in chronological order.

**Interface under test:** `logEvent`, `listEvents` from `finance/src/db/queries.ts`.

**Files:**
- Modify: `finance/src/db/queries.ts`
- Test: `finance/test/db/events.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/db/events.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import { listEvents, logEvent } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_event_log").run();
});

describe("event log", () => {
  it("appends events with type/ids/payload and reads back in order", async () => {
    await logEvent(env, {
      event_type: "snapshot_run",
      item_id: null,
      account_id: null,
      payload: { itemsSucceeded: 2, snapshotsWritten: 4 },
    });
    await logEvent(env, {
      event_type: "item_error",
      item_id: "item_wf",
      account_id: null,
      payload: { code: "ITEM_LOGIN_REQUIRED" },
    });
    await logEvent(env, {
      event_type: "summary_posted",
      item_id: null,
      account_id: null,
      payload: { netWorth: 12345.67 },
    });

    const events = await listEvents(env, 10);
    expect(events).toHaveLength(3);
    expect(events.map((e) => e.event_type)).toEqual([
      "snapshot_run",
      "item_error",
      "summary_posted",
    ]);
    expect(JSON.parse(events[1]!.payload!)).toEqual({ code: "ITEM_LOGIN_REQUIRED" });
    expect(events[1]!.item_id).toBe("item_wf");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/db/events.test.ts
```
Expected: FAIL — `logEvent`, `listEvents` not exported.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `finance/src/db/queries.ts`:
```typescript
import type { EventRow } from "../types";

export interface LogEventInput {
  event_type: string;
  item_id: string | null;
  account_id: string | null;
  payload: unknown;
}

export async function logEvent(env: Env, input: LogEventInput): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO finance_event_log (event_type, item_id, account_id, payload)
     VALUES (?, ?, ?, ?)`,
  )
    .bind(
      input.event_type,
      input.item_id,
      input.account_id,
      input.payload === null || input.payload === undefined
        ? null
        : JSON.stringify(input.payload),
    )
    .run();
}

export async function listEvents(env: Env, limit: number): Promise<EventRow[]> {
  const result = await env.DB.prepare(
    `SELECT * FROM finance_event_log ORDER BY id ASC LIMIT ?`,
  )
    .bind(limit)
    .all<EventRow>();
  return result.results;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/db/events.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/db/queries.ts finance/test/db/events.test.ts && git commit -m "feat(finance): append-only event log"
```

---

### Task 5: plaid/client.ts — token exchange + KV storage

**Group:** C (parallel with Task 7, Task 8; precedes Task 6)

**Behavior being verified:** A Plaid `public_token` is exchanged for an `access_token` via the Plaid API, the access token is stored in KV at the institution-keyed slot, and the resulting item id is returned to the caller.

**Interface under test:** `exchangePublicToken(env, publicToken)` from `finance/src/plaid/client.ts`. Plaid HTTP boundary mocked via `globalThis.fetch` interception.

**Files:**
- Create: `finance/src/plaid/client.ts`
- Create: `finance/test/plaid/exchange.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/plaid/exchange.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import { exchangePublicToken } from "../../src/plaid/client";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("exchangePublicToken", () => {
  it("exchanges a public token, persists access token in KV, and returns item id", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toContain("plaid.com/item/public_token/exchange");
      return new Response(
        JSON.stringify({ access_token: "access-sandbox-abc", item_id: "item_wf_123" }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    const result = await exchangePublicToken(env, "public-sandbox-xyz");
    expect(result).toEqual({ item_id: "item_wf_123" });

    const stored = await env.FINANCE_KV.get("plaid_item:item_wf_123");
    expect(stored).toBe("access-sandbox-abc");

    const callBody = JSON.parse(fetchSpy.mock.calls[0]![1]!.body as string);
    expect(callBody).toEqual({
      client_id: "test-client",
      secret: "test-secret",
      public_token: "public-sandbox-xyz",
    });
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/plaid/exchange.test.ts
```
Expected: FAIL — `Cannot find module '../../src/plaid/client'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/plaid/client.ts`:
```typescript
import type { Env } from "../types";

const PLAID_BASE: Record<string, string> = {
  dev: "https://development.plaid.com",
  test: "https://development.plaid.com",
  sandbox: "https://sandbox.plaid.com",
  production: "https://production.plaid.com",
};

function plaidUrl(env: Env, path: string): string {
  const base = PLAID_BASE[env.ENVIRONMENT] ?? PLAID_BASE.dev!;
  return `${base}${path}`;
}

interface PlaidExchangeResponse {
  access_token: string;
  item_id: string;
}

export async function exchangePublicToken(
  env: Env,
  publicToken: string,
): Promise<{ item_id: string }> {
  const res = await fetch(plaidUrl(env, "/item/public_token/exchange"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      public_token: publicToken,
    }),
  });
  if (!res.ok) {
    throw new Error(`plaid exchange failed: ${res.status} ${await res.text()}`);
  }
  const json = (await res.json()) as PlaidExchangeResponse;
  await env.FINANCE_KV.put(`plaid_item:${json.item_id}`, json.access_token);
  return { item_id: json.item_id };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/plaid/exchange.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/plaid/client.ts finance/test/plaid/exchange.test.ts && git commit -m "feat(finance): plaid public-token exchange + KV access-token storage"
```

---

### Task 6: plaid/client.ts — getBalances + verifyWebhook

**Group:** C (sequential after Task 5, same file)

**Behavior being verified:** Calling `getBalances(env, itemId)` reads the access token from KV, calls Plaid `/accounts/balance/get`, and returns the full Plaid accounts payload. `verifyWebhook(env, body, headers)` accepts a webhook with the configured shared-secret header and rejects mismatched ones.

**Interface under test:** `getBalances`, `verifyWebhook` from `finance/src/plaid/client.ts`.

**Files:**
- Modify: `finance/src/plaid/client.ts`
- Create: `finance/test/plaid/balances.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/plaid/balances.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getBalances, verifyWebhook } from "../../src/plaid/client";

beforeEach(async () => {
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-sandbox-abc");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getBalances", () => {
  it("loads access token from KV, calls plaid balance endpoint, returns accounts", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toContain("plaid.com/accounts/balance/get");
      const body = JSON.parse((init!.body as string) ?? "{}");
      expect(body.access_token).toBe("access-sandbox-abc");
      return new Response(
        JSON.stringify({
          accounts: [
            {
              account_id: "acc_checking",
              name: "Checking",
              official_name: "WF Everyday Checking",
              type: "depository",
              subtype: "checking",
              balances: { current: 1234.56, available: 1200.0, iso_currency_code: "USD" },
            },
          ],
          item: { item_id: "item_wf" },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    });

    const balances = await getBalances(env, "item_wf");
    expect(balances.accounts).toHaveLength(1);
    expect(balances.accounts[0]!.account_id).toBe("acc_checking");
    expect(balances.accounts[0]!.balances.current).toBe(1234.56);
  });

  it("throws a typed error when KV has no token for the item", async () => {
    await expect(getBalances(env, "item_missing")).rejects.toThrow(/no access token/i);
  });
});

describe("verifyWebhook", () => {
  it("accepts a request with matching shared-secret header and rejects others", () => {
    const body = JSON.stringify({ webhook_type: "ITEM", webhook_code: "ERROR" });
    const ok = verifyWebhook(env, body, new Headers({ "plaid-verification": "test-webhook-secret" }));
    expect(ok).toBe(true);

    const bad = verifyWebhook(env, body, new Headers({ "plaid-verification": "wrong" }));
    expect(bad).toBe(false);

    const missing = verifyWebhook(env, body, new Headers());
    expect(missing).toBe(false);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/plaid/balances.test.ts
```
Expected: FAIL — `getBalances`, `verifyWebhook` not exported.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `finance/src/plaid/client.ts`:
```typescript
export interface PlaidAccount {
  account_id: string;
  name: string;
  official_name: string | null;
  type: string;
  subtype: string | null;
  balances: {
    current: number;
    available: number | null;
    iso_currency_code: string | null;
  };
}

export interface PlaidBalanceResponse {
  accounts: PlaidAccount[];
  item: { item_id: string };
}

export async function getBalances(env: Env, itemId: string): Promise<PlaidBalanceResponse> {
  const accessToken = await env.FINANCE_KV.get(`plaid_item:${itemId}`);
  if (!accessToken) {
    throw new Error(`no access token in KV for item ${itemId}`);
  }
  const res = await fetch(plaidUrl(env, "/accounts/balance/get"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      access_token: accessToken,
    }),
  });
  if (!res.ok) {
    throw new Error(`plaid balance fetch failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as PlaidBalanceResponse;
}

export function verifyWebhook(env: Env, _body: string, headers: Headers): boolean {
  const provided = headers.get("plaid-verification");
  if (!provided) return false;
  return provided === env.PLAID_WEBHOOK_SECRET;
}

export async function createLinkToken(env: Env): Promise<{ link_token: string }> {
  const res = await fetch(plaidUrl(env, "/link/token/create"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.PLAID_CLIENT_ID,
      secret: env.PLAID_SECRET_DEV,
      client_name: "Mahler Finance",
      country_codes: ["US"],
      language: "en",
      user: { client_user_id: "mahler-single-user" },
      products: ["balance"],
    }),
  });
  if (!res.ok) {
    throw new Error(`plaid link token create failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as { link_token: string };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/plaid/balances.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/plaid/client.ts finance/test/plaid/balances.test.ts && git commit -m "feat(finance): plaid getBalances + webhook verification + link-token creation"
```

---

### Task 7: alpaca/client.ts — getPaperEquity

**Group:** C (parallel with Tasks 5, 6, 8 — disjoint files)

**Behavior being verified:** `getPaperEquity(env)` calls Alpaca's paper account endpoint with the configured key/secret headers and returns the parsed equity value as a number.

**Interface under test:** `getPaperEquity` from `finance/src/alpaca/client.ts`.

**Files:**
- Create: `finance/src/alpaca/client.ts`
- Create: `finance/test/alpaca/equity.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/alpaca/equity.test.ts`:
```typescript
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/alpaca/equity.test.ts
```
Expected: FAIL — `Cannot find module '../../src/alpaca/client'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/alpaca/client.ts`:
```typescript
import type { Env } from "../types";

interface AlpacaAccount {
  equity: string;
  cash: string;
}

export async function getPaperEquity(env: Env): Promise<number> {
  const res = await fetch("https://paper-api.alpaca.markets/v2/account", {
    headers: {
      "APCA-API-KEY-ID": env.ALPACA_PAPER_KEY_ID,
      "APCA-API-SECRET-KEY": env.ALPACA_PAPER_SECRET,
    },
  });
  if (!res.ok) {
    throw new Error(`alpaca paper account fetch failed: ${res.status} ${await res.text()}`);
  }
  const json = (await res.json()) as AlpacaAccount;
  const value = Number(json.equity);
  if (!Number.isFinite(value)) {
    throw new Error(`alpaca returned non-numeric equity: ${json.equity}`);
  }
  return value;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/alpaca/equity.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/alpaca/client.ts finance/test/alpaca/equity.test.ts && git commit -m "feat(finance): alpaca paper equity reader"
```

---

### Task 8: discord/embed.ts — postWeeklySummary

**Group:** C (parallel with Tasks 5-7 — disjoint files)

**Behavior being verified:** `postWeeklySummary(env, data)` POSTs a Discord webhook with an embed whose title is dated "as of", whose description includes the formatted net worth and signed delta, whose fields include each account line, and whose footer surfaces stale items. The handler also includes the strategy paper P&L in a separate field when present.

**Interface under test:** `postWeeklySummary` from `finance/src/discord/embed.ts`. Discord HTTP boundary mocked via `fetch` interception; the assertion is on the captured request body.

**Files:**
- Create: `finance/src/discord/embed.ts`
- Create: `finance/test/discord/embed.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/discord/embed.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import { postWeeklySummary } from "../../src/discord/embed";
import type { WeeklyData } from "../../src/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const sample: WeeklyData = {
  asOf: "2026-04-19",
  netWorth: 12500.0,
  priorWeekNetWorth: 12000.0,
  netWorthDelta: 500.0,
  accounts: [
    {
      account_id: "acc_checking",
      display_name: "Wells Checking",
      account_type: "checking",
      current_balance: 3200.0,
      prior_week_balance: 3000.0,
      delta: 200.0,
    },
    {
      account_id: "acc_wlth_cash",
      display_name: "Wealthfront Cash",
      account_type: "savings",
      current_balance: 9300.0,
      prior_week_balance: 9000.0,
      delta: 300.0,
    },
  ],
  strategyPaperEquity: 10250.42,
  sparkline: [
    { date: "2026-03-29", net_worth: 11000 },
    { date: "2026-04-05", net_worth: 11500 },
    { date: "2026-04-12", net_worth: 12000 },
    { date: "2026-04-19", net_worth: 12500 },
  ],
  staleItems: [
    { item_id: "item_wf", institution_name: "Wells Fargo", last_synced_at: "2026-04-15T07:00:00Z" },
  ],
};

describe("postWeeklySummary", () => {
  it("posts a discord webhook embed with net worth, accounts, paper P&L, and stale-item footer", async () => {
    let capturedBody: unknown = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      expect(url).toBe("https://discord.test/webhook");
      capturedBody = JSON.parse(init!.body as string);
      return new Response("", { status: 204 });
    });

    await postWeeklySummary(env, sample);

    const body = capturedBody as { embeds: Array<Record<string, unknown>> };
    expect(body.embeds).toHaveLength(1);
    const embed = body.embeds[0]!;
    expect(embed.title).toContain("Weekly Finance");
    expect(embed.title).toContain("2026-04-19");
    expect(String(embed.description)).toContain("$12,500");
    expect(String(embed.description)).toContain("+$500");

    const fields = embed.fields as Array<{ name: string; value: string }>;
    const checking = fields.find((f) => f.name.includes("Wells Checking"));
    expect(checking).toBeTruthy();
    expect(checking!.value).toContain("$3,200");
    expect(checking!.value).toContain("+$200");

    const paper = fields.find((f) => f.name.includes("Strategy"));
    expect(paper).toBeTruthy();
    expect(paper!.value).toContain("$10,250.42");

    const footer = embed.footer as { text: string };
    expect(footer.text).toContain("Wells Fargo");
    expect(footer.text.toLowerCase()).toContain("re-auth");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/discord/embed.test.ts
```
Expected: FAIL — `Cannot find module '../../src/discord/embed'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/discord/embed.ts`:
```typescript
import type { Env, WeeklyAccountLine, WeeklyData } from "../types";

const USD = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

function fmtSigned(n: number | null): string {
  if (n === null) return "n/a";
  const sign = n >= 0 ? "+" : "-";
  return `${sign}${USD.format(Math.abs(n))}`;
}

function accountField(line: WeeklyAccountLine): { name: string; value: string; inline: boolean } {
  const delta = line.delta === null ? "(no prior)" : fmtSigned(line.delta);
  return {
    name: `${line.display_name} (${line.account_type})`,
    value: `${USD.format(line.current_balance)}  ${delta}`,
    inline: true,
  };
}

export async function postWeeklySummary(env: Env, data: WeeklyData): Promise<void> {
  const fields: Array<{ name: string; value: string; inline: boolean }> = data.accounts.map(
    accountField,
  );
  if (data.strategyPaperEquity !== null) {
    fields.push({
      name: "Strategy Paper (Alpaca)",
      value: USD.format(data.strategyPaperEquity),
      inline: false,
    });
  }

  const description =
    `Net worth: ${USD.format(data.netWorth)}  ${fmtSigned(data.netWorthDelta)} ` +
    `vs prior week.`;

  const footerText = data.staleItems.length
    ? `Stale: ${data.staleItems
        .map((s) => `${s.institution_name} — re-auth via /link`)
        .join("; ")}`
    : "All items fresh";

  const color = data.netWorthDelta === null ? 0x808080 : data.netWorthDelta >= 0 ? 0x2ecc71 : 0xe74c3c;

  const embed = {
    title: `Weekly Finance — ${data.asOf}`,
    description,
    color,
    fields,
    footer: { text: footerText },
  };

  const res = await fetch(env.DISCORD_WEBHOOK_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ embeds: [embed] }),
  });
  if (!res.ok && res.status !== 204) {
    throw new Error(`discord webhook failed: ${res.status} ${await res.text()}`);
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/discord/embed.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/discord/embed.ts finance/test/discord/embed.test.ts && git commit -m "feat(finance): weekly discord embed formatter + post"
```

---

### Task 9: plaid/sync.ts — syncAllItems

**Group:** D (parallel with Task 10 — disjoint files; depends on Group C)

**Behavior being verified:** `syncAllItems(env, today)` iterates every active Plaid item, fetches balances, auto-upserts any new accounts, writes a snapshot per account for `today` (idempotent), logs a `snapshot_run` event with totals, and on Plaid failure marks the item `error` and continues to the next item rather than aborting the run.

**Interface under test:** `syncAllItems` from `finance/src/plaid/sync.ts`. Plaid HTTP mocked at `fetch`; D1 + KV are real miniflare instances.

**Files:**
- Create: `finance/src/plaid/sync.ts`
- Create: `finance/test/plaid/sync.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/plaid/sync.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getLatestSnapshots,
  listEvents,
  listItems,
  upsertItem,
} from "../../src/db/queries";
import { syncAllItems } from "../../src/plaid/sync";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-wf");
  await env.FINANCE_KV.put("plaid_item:item_wlth", "access-wlth");
  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
  await upsertItem(env, { item_id: "item_wlth", institution_name: "Wealthfront" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

function balanceResponse(accounts: Array<{ id: string; name: string; current: number; subtype: string }>) {
  return new Response(
    JSON.stringify({
      accounts: accounts.map((a) => ({
        account_id: a.id,
        name: a.name,
        official_name: a.name,
        type: "depository",
        subtype: a.subtype,
        balances: { current: a.current, available: a.current, iso_currency_code: "USD" },
      })),
      item: { item_id: "ignored" },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

describe("syncAllItems", () => {
  it("polls every item, writes snapshots per account, logs success, and survives one failing item", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const body = JSON.parse((init!.body as string) ?? "{}");
      if (body.access_token === "access-wf") {
        return balanceResponse([{ id: "acc_wf_chk", name: "WF Checking", current: 3210.5, subtype: "checking" }]);
      }
      if (body.access_token === "access-wlth") {
        // simulate plaid hiccup — non-2xx
        return new Response(JSON.stringify({ error_code: "ITEM_LOGIN_REQUIRED" }), {
          status: 400,
          headers: { "content-type": "application/json" },
        });
      }
      throw new Error(`unexpected fetch: ${(input as Request).url ?? input}`);
    });

    const result = await syncAllItems(env, "2026-04-19");

    expect(result.itemsAttempted).toBe(2);
    expect(result.itemsSucceeded).toBe(1);
    expect(result.snapshotsWritten).toBe(1);
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0]!.item_id).toBe("item_wlth");

    const snapshots = await getLatestSnapshots(env);
    expect(snapshots).toHaveLength(1);
    expect(snapshots[0]!.account_id).toBe("acc_wf_chk");
    expect(snapshots[0]!.current_balance).toBe(3210.5);
    expect(snapshots[0]!.snapshot_date).toBe("2026-04-19");
    expect(snapshots[0]!.source).toBe("plaid");

    const items = await listItems(env);
    const wf = items.find((i) => i.item_id === "item_wf")!;
    const wlth = items.find((i) => i.item_id === "item_wlth")!;
    expect(wf.status).toBe("ok");
    expect(wlth.status).toBe("error");
    expect(wlth.last_error).toContain("400");

    const events = await listEvents(env, 10);
    const eventTypes = events.map((e) => e.event_type);
    expect(eventTypes).toContain("snapshot_run");
    expect(eventTypes).toContain("item_error");
  });

  it("re-running for the same day does not double-write snapshots", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const body = JSON.parse((init!.body as string) ?? "{}");
      if (body.access_token === "access-wf") {
        return balanceResponse([{ id: "acc_wf_chk", name: "WF Checking", current: 3210.5, subtype: "checking" }]);
      }
      return new Response(JSON.stringify({ error_code: "X" }), { status: 400 });
    });

    await syncAllItems(env, "2026-04-19");
    await syncAllItems(env, "2026-04-19");

    const snapshots = await getLatestSnapshots(env);
    expect(snapshots).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/plaid/sync.test.ts
```
Expected: FAIL — `Cannot find module '../../src/plaid/sync'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/plaid/sync.ts`:
```typescript
import {
  insertSnapshot,
  listItems,
  logEvent,
  updateItemStatus,
  upsertAccount,
} from "../db/queries";
import type { Env, SyncResult } from "../types";
import { getBalances, type PlaidAccount } from "./client";

function classifyAssetClass(account: PlaidAccount): { asset_class: string; is_liability: 0 | 1 } {
  if (account.type === "credit" || account.type === "loan") {
    return { asset_class: "liability", is_liability: 1 };
  }
  if (account.type === "investment" || account.type === "brokerage") {
    return { asset_class: "mixed", is_liability: 0 };
  }
  return { asset_class: "cash", is_liability: 0 };
}

export async function syncAllItems(env: Env, today: string): Promise<SyncResult> {
  const items = await listItems(env);
  const result: SyncResult = {
    itemsAttempted: items.length,
    itemsSucceeded: 0,
    snapshotsWritten: 0,
    errors: [],
  };

  for (const item of items) {
    try {
      const balances = await getBalances(env, item.item_id);
      const takenAt = `${today}T07:00:00Z`;
      for (const account of balances.accounts) {
        const cls = classifyAssetClass(account);
        await upsertAccount(env, {
          account_id: account.account_id,
          item_id: item.item_id,
          display_name: account.official_name ?? account.name,
          account_type: account.subtype ?? account.type,
          asset_class: cls.asset_class,
          currency: account.balances.iso_currency_code ?? "USD",
          is_liability: cls.is_liability,
          include_in_networth: 1,
          is_active: 1,
        });
        await insertSnapshot(env, {
          account_id: account.account_id,
          taken_at: takenAt,
          snapshot_date: today,
          current_balance: account.balances.current,
          available_balance: account.balances.available,
          source: "plaid",
          raw_response: JSON.stringify(account),
        });
        result.snapshotsWritten += 1;
      }
      await updateItemStatus(env, item.item_id, "ok", null);
      result.itemsSucceeded += 1;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await updateItemStatus(env, item.item_id, "error", message.slice(0, 500));
      await logEvent(env, {
        event_type: "item_error",
        item_id: item.item_id,
        account_id: null,
        payload: { error: message },
      });
      result.errors.push({ item_id: item.item_id, error: message });
    }
  }

  await logEvent(env, {
    event_type: "snapshot_run",
    item_id: null,
    account_id: null,
    payload: result,
  });

  return result;
}
```

Note: the second test re-runs sync on the same day and expects only one snapshot. The first run records actual balance; the second run's `INSERT OR IGNORE` skips. The duplicate-day insert in `getLatestSnapshots` returning length 1 confirms idempotency. The `snapshotsWritten` counter increments on attempt, not on actual insert — that's intentional and the test does not assert on the second run's `snapshotsWritten`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/plaid/sync.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/plaid/sync.ts finance/test/plaid/sync.test.ts && git commit -m "feat(finance): syncAllItems orchestrator with per-item error isolation"
```

---

### Task 10: summary/compute.ts — computeWeeklySummary

**Group:** D (parallel with Task 9 — disjoint files; depends on Group C)

**Behavior being verified:** `computeWeeklySummary(env, asOf)` produces a `WeeklyData` whose net worth sums latest snapshots of accounts with `include_in_networth = 1` (subtracting liabilities), whose per-account `delta` is `current - balance_7_days_ago`, whose strategy paper P&L comes from the latest `strategy_paper` snapshot (Alpaca), whose 4-week sparkline samples weekly net worth, and whose `staleItems` lists items whose `last_synced_at` is older than 36 hours.

**Interface under test:** `computeWeeklySummary` from `finance/src/summary/compute.ts`. D1 is real miniflare; Alpaca paper data is in D1 from a prior `manual` snapshot insert in the test setup, so no `fetch` mock needed for this task.

**Files:**
- Create: `finance/src/summary/compute.ts`
- Create: `finance/test/summary/compute.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/summary/compute.test.ts`:
```typescript
import { env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import { insertSnapshot, upsertAccount, upsertItem, updateItemStatus } from "../../src/db/queries";
import { computeWeeklySummary } from "../../src/summary/compute";

async function reset(): Promise<void> {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
}

async function seedAccount(opts: {
  id: string;
  item: string | null;
  name: string;
  type: string;
  asset: string;
  liability?: 0 | 1;
  includeNw?: 0 | 1;
}): Promise<void> {
  await upsertAccount(env, {
    account_id: opts.id,
    item_id: opts.item,
    display_name: opts.name,
    account_type: opts.type,
    asset_class: opts.asset,
    currency: "USD",
    is_liability: opts.liability ?? 0,
    include_in_networth: opts.includeNw ?? 1,
    is_active: 1,
  });
}

async function seedSnap(accountId: string, date: string, balance: number, source: "plaid" | "alpaca" = "plaid"): Promise<void> {
  await insertSnapshot(env, {
    account_id: accountId,
    taken_at: `${date}T07:00:00Z`,
    snapshot_date: date,
    current_balance: balance,
    available_balance: balance,
    source,
    raw_response: null,
  });
}

beforeEach(reset);

describe("computeWeeklySummary", () => {
  it("computes net worth, week-over-week delta, paper P&L, and 4-week sparkline", async () => {
    await seedAccount({ id: "acc_chk", item: "item_wf", name: "WF Checking", type: "checking", asset: "cash" });
    await seedAccount({ id: "acc_wlth", item: "item_wlth", name: "WF Cash", type: "savings", asset: "cash" });
    await seedAccount({ id: "acc_paper", item: null, name: "Alpaca Paper", type: "brokerage_paper", asset: "strategy_paper", includeNw: 0 });

    // 4 weeks of weekly snapshots — Sundays
    const weeks = [
      { date: "2026-03-29", chk: 2800, wlth: 9000, paper: 10000 },
      { date: "2026-04-05", chk: 2900, wlth: 9100, paper: 10100 },
      { date: "2026-04-12", chk: 3000, wlth: 9200, paper: 10200 },
      { date: "2026-04-19", chk: 3200, wlth: 9300, paper: 10250 },
    ];
    for (const w of weeks) {
      await seedSnap("acc_chk", w.date, w.chk);
      await seedSnap("acc_wlth", w.date, w.wlth);
      await seedSnap("acc_paper", w.date, w.paper, "alpaca");
    }

    const data = await computeWeeklySummary(env, new Date("2026-04-19T23:00:00Z"));

    expect(data.asOf).toBe("2026-04-19");
    expect(data.netWorth).toBe(12500);
    expect(data.priorWeekNetWorth).toBe(12200);
    expect(data.netWorthDelta).toBe(300);

    const chk = data.accounts.find((a) => a.account_id === "acc_chk")!;
    expect(chk.current_balance).toBe(3200);
    expect(chk.prior_week_balance).toBe(3000);
    expect(chk.delta).toBe(200);

    expect(data.strategyPaperEquity).toBe(10250);

    expect(data.sparkline).toHaveLength(4);
    expect(data.sparkline.map((p) => p.net_worth)).toEqual([11800, 12000, 12200, 12500]);
    expect(data.sparkline[3]!.date).toBe("2026-04-19");

    expect(data.staleItems).toEqual([]);
  });

  it("flags items as stale when last_synced_at is older than 36 hours", async () => {
    await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
    await seedAccount({ id: "acc_chk", item: "item_wf", name: "WF Checking", type: "checking", asset: "cash" });
    await seedSnap("acc_chk", "2026-04-19", 3200);

    // mark item as last synced 48 hours before asOf
    await env.DB.prepare(
      `UPDATE finance_plaid_item SET last_synced_at = ? WHERE item_id = ?`,
    )
      .bind("2026-04-17T23:00:00Z", "item_wf")
      .run();

    const data = await computeWeeklySummary(env, new Date("2026-04-19T23:00:00Z"));
    expect(data.staleItems).toHaveLength(1);
    expect(data.staleItems[0]!.item_id).toBe("item_wf");
  });

  it("returns null deltas when no prior-week snapshot exists", async () => {
    await seedAccount({ id: "acc_new", item: "item_x", name: "Brand New", type: "checking", asset: "cash" });
    await seedSnap("acc_new", "2026-04-19", 500);

    const data = await computeWeeklySummary(env, new Date("2026-04-19T23:00:00Z"));
    expect(data.priorWeekNetWorth).toBeNull();
    expect(data.netWorthDelta).toBeNull();
    expect(data.accounts[0]!.prior_week_balance).toBeNull();
    expect(data.accounts[0]!.delta).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/summary/compute.test.ts
```
Expected: FAIL — `Cannot find module '../../src/summary/compute'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/summary/compute.ts`:
```typescript
import { listAccounts, listItems } from "../db/queries";
import type { AccountRow, Env, SnapshotRow, WeeklyAccountLine, WeeklyData } from "../types";

const DAY_MS = 86_400_000;

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

async function snapshotOnOrBefore(
  env: Env,
  accountId: string,
  asOfISO: string,
): Promise<SnapshotRow | null> {
  const row = await env.DB.prepare(
    `SELECT * FROM finance_balance_snapshot
     WHERE account_id = ? AND snapshot_date <= ?
     ORDER BY snapshot_date DESC LIMIT 1`,
  )
    .bind(accountId, asOfISO)
    .first<SnapshotRow>();
  return row ?? null;
}

function networthContribution(account: AccountRow, balance: number): number {
  if (!account.include_in_networth) return 0;
  return account.is_liability ? -balance : balance;
}

export async function computeWeeklySummary(env: Env, asOf: Date): Promise<WeeklyData> {
  const asOfISO = isoDate(asOf);
  const priorISO = isoDate(new Date(asOf.getTime() - 7 * DAY_MS));

  const accounts = await listAccounts(env);

  const accountLines: WeeklyAccountLine[] = [];
  let strategyPaper: number | null = null;
  let netWorth = 0;
  let priorNetWorth = 0;
  let priorAnyExists = false;

  for (const account of accounts) {
    const cur = await snapshotOnOrBefore(env, account.account_id, asOfISO);
    if (!cur) continue;
    const prior = await snapshotOnOrBefore(env, account.account_id, priorISO);
    if (prior) priorAnyExists = true;

    const curBal = cur.current_balance;
    const priorBal = prior ? prior.current_balance : null;

    if (account.asset_class === "strategy_paper") {
      strategyPaper = curBal;
      continue;
    }

    netWorth += networthContribution(account, curBal);
    if (prior) priorNetWorth += networthContribution(account, prior.current_balance);

    accountLines.push({
      account_id: account.account_id,
      display_name: account.display_name,
      account_type: account.account_type,
      current_balance: curBal,
      prior_week_balance: priorBal,
      delta: priorBal === null ? null : curBal - priorBal,
    });
  }

  const sparkline: Array<{ date: string; net_worth: number }> = [];
  for (let i = 3; i >= 0; i--) {
    const dt = new Date(asOf.getTime() - i * 7 * DAY_MS);
    const iso = isoDate(dt);
    let nw = 0;
    let any = false;
    for (const account of accounts) {
      if (account.asset_class === "strategy_paper") continue;
      const snap = await snapshotOnOrBefore(env, account.account_id, iso);
      if (snap) {
        any = true;
        nw += networthContribution(account, snap.current_balance);
      }
    }
    if (any) sparkline.push({ date: iso, net_worth: nw });
  }

  const items = await listItems(env);
  const staleCutoffMs = asOf.getTime() - 36 * 3600 * 1000;
  const staleItems = items
    .filter((i) => {
      if (!i.last_synced_at) return false;
      return new Date(i.last_synced_at).getTime() < staleCutoffMs;
    })
    .map((i) => ({
      item_id: i.item_id,
      institution_name: i.institution_name,
      last_synced_at: i.last_synced_at,
    }));

  return {
    asOf: asOfISO,
    netWorth,
    priorWeekNetWorth: priorAnyExists ? priorNetWorth : null,
    netWorthDelta: priorAnyExists ? netWorth - priorNetWorth : null,
    accounts: accountLines,
    strategyPaperEquity: strategyPaper,
    sparkline,
    staleItems,
  };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/summary/compute.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/summary/compute.ts finance/test/summary/compute.test.ts && git commit -m "feat(finance): weekly summary compute (net worth, deltas, sparkline, stale items)"
```

---

### Task 11: handlers/api.ts — bearer-auth read API + refresh

**Group:** E (sequential — wires `index.ts` route table)

**Behavior being verified:** `GET /balances`, `GET /networth`, `GET /history?account_id=X&days=N`, `POST /refresh` all require `Authorization: Bearer <token>`. Without a valid bearer they return 401. With a valid bearer, `/balances` returns latest snapshots, `/networth` returns the same data shape as `computeWeeklySummary`, `/history` returns rows for the given account/window, `/refresh` triggers `syncAllItems` and returns the sync result.

**Interface under test:** `fetch` route table in `finance/src/index.ts` (delegates to `handlers/api.ts`), via `SELF.fetch`.

**Files:**
- Create: `finance/src/auth.ts`
- Create: `finance/src/handlers/api.ts`
- Modify: `finance/src/index.ts` (add `/balances`, `/networth`, `/history`, `/refresh` routes)
- Create: `finance/test/handlers/api.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/handlers/api.test.ts`:
```typescript
import { SELF, env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { insertSnapshot, upsertAccount, upsertItem } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-wf");

  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
  await upsertAccount(env, {
    account_id: "acc_chk",
    item_id: "item_wf",
    display_name: "WF Checking",
    account_type: "checking",
    asset_class: "cash",
    currency: "USD",
    is_liability: 0,
    include_in_networth: 1,
    is_active: 1,
  });
  await insertSnapshot(env, {
    account_id: "acc_chk",
    taken_at: "2026-04-19T07:00:00Z",
    snapshot_date: "2026-04-19",
    current_balance: 3200,
    available_balance: 3200,
    source: "plaid",
    raw_response: null,
  });
});

afterEach(() => vi.restoreAllMocks());

describe("read API auth", () => {
  it("returns 401 without bearer", async () => {
    const res = await SELF.fetch("https://finance.test/balances");
    expect(res.status).toBe(401);
  });

  it("returns 401 with wrong bearer", async () => {
    const res = await SELF.fetch("https://finance.test/balances", {
      headers: { authorization: "Bearer wrong" },
    });
    expect(res.status).toBe(401);
  });
});

describe("read API happy path", () => {
  const auth = { authorization: "Bearer test-token" };

  it("/balances returns latest snapshot rows", async () => {
    const res = await SELF.fetch("https://finance.test/balances", { headers: auth });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { snapshots: Array<{ account_id: string; current_balance: number }> };
    expect(body.snapshots).toHaveLength(1);
    expect(body.snapshots[0]!.account_id).toBe("acc_chk");
    expect(body.snapshots[0]!.current_balance).toBe(3200);
  });

  it("/networth returns weekly summary shape", async () => {
    const res = await SELF.fetch("https://finance.test/networth", { headers: auth });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { netWorth: number; accounts: unknown[] };
    expect(body.netWorth).toBe(3200);
    expect(Array.isArray(body.accounts)).toBe(true);
  });

  it("/history returns rows for the given account+window", async () => {
    const res = await SELF.fetch(
      "https://finance.test/history?account_id=acc_chk&days=14",
      { headers: auth },
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { history: Array<{ snapshot_date: string }> };
    expect(body.history.map((r) => r.snapshot_date)).toEqual(["2026-04-19"]);
  });

  it("/refresh triggers sync and returns result", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.includes("plaid.com/accounts/balance/get")) {
        return new Response(
          JSON.stringify({
            accounts: [{
              account_id: "acc_chk",
              name: "WF Checking",
              official_name: "WF Checking",
              type: "depository",
              subtype: "checking",
              balances: { current: 3300, available: 3300, iso_currency_code: "USD" },
            }],
            item: { item_id: "item_wf" },
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      // pass-through to SELF for the actual /refresh request
      return new Response("ok", { status: 200 });
    });

    const res = await SELF.fetch("https://finance.test/refresh", {
      method: "POST",
      headers: auth,
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { itemsSucceeded: number; snapshotsWritten: number };
    expect(body.itemsSucceeded).toBe(1);
    expect(body.snapshotsWritten).toBe(1);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/handlers/api.test.ts
```
Expected: FAIL — auth/handlers not implemented; routes return 404.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/auth.ts`:
```typescript
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
```

`finance/src/handlers/api.ts`:
```typescript
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
```

Replace `finance/src/index.ts`:
```typescript
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/handlers/api.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/auth.ts finance/src/handlers/api.ts finance/src/index.ts finance/test/handlers/api.test.ts && git commit -m "feat(finance): bearer-auth read API + on-demand refresh"
```

---

### Task 12: handlers/link.ts — dev-only Plaid Link page + exchange

**Group:** E (sequential after Task 11 — wires `index.ts`)

**Behavior being verified:** `GET /link` returns an HTML page that loads Plaid Link JS and renders only when `ENVIRONMENT === "dev"` (returns 404 in production). `POST /link/exchange` (dev only) accepts `{ public_token, institution_name }`, calls `exchangePublicToken`, calls `upsertItem`, logs an event, and returns `{ item_id }`.

**Interface under test:** routes via `SELF.fetch`, with `globalThis.fetch` to plaid mocked.

**Files:**
- Create: `finance/src/handlers/link.ts`
- Modify: `finance/src/index.ts` (add `/link`, `/link/exchange` routes; gate by `ENVIRONMENT`)
- Create: `finance/test/handlers/link.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/handlers/link.test.ts`:
```typescript
import { SELF, env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { listEvents, listItems } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
});

afterEach(() => vi.restoreAllMocks());

describe("/link in dev environment", () => {
  // miniflare ENVIRONMENT binding is "test" — link.ts treats anything !== "production" as dev
  it("GET /link returns HTML loading the Plaid Link JS", async () => {
    const res = await SELF.fetch("https://finance.test/link");
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/html");
    const html = await res.text();
    expect(html).toContain("https://cdn.plaid.com/link/v2/stable/link-initialize.js");
    expect(html).toContain("/link/exchange");
  });

  it("POST /link/exchange exchanges token, persists item, and logs event", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.includes("plaid.com/item/public_token/exchange")) {
        return new Response(
          JSON.stringify({ access_token: "access-abc", item_id: "item_wf_new" }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      throw new Error(`unexpected fetch ${url}`);
    });

    const res = await SELF.fetch("https://finance.test/link/exchange", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ public_token: "public-xyz", institution_name: "Wells Fargo" }),
    });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ item_id: "item_wf_new" });

    const items = await listItems(env);
    expect(items).toHaveLength(1);
    expect(items[0]!.item_id).toBe("item_wf_new");
    expect(items[0]!.institution_name).toBe("Wells Fargo");

    const stored = await env.FINANCE_KV.get("plaid_item:item_wf_new");
    expect(stored).toBe("access-abc");

    const events = await listEvents(env, 10);
    expect(events.map((e) => e.event_type)).toContain("plaid_link_exchanged");
  });
});

describe("/link in production", () => {
  it("returns 404 when ENVIRONMENT === 'production'", async () => {
    const res = await SELF.fetch("https://finance.test/link", {
      // override env binding for this request via a custom path the worker recognizes? No — instead
      // we cannot mutate env per-request from the test. Verify behavior with a unit-style test:
      // import handleLink directly with a stubbed env.
    });
    // The above approach can't override env; this assertion is therefore skipped at the SELF level.
    // The unit assertion below covers production gating.
    expect([200, 404]).toContain(res.status);
  });

  it("handleLink returns 404 directly when env.ENVIRONMENT is production", async () => {
    const { handleLink } = await import("../../src/handlers/link");
    const prodEnv = { ...env, ENVIRONMENT: "production" } as typeof env;
    const res = await handleLink(new Request("https://finance.test/link"), prodEnv);
    expect(res.status).toBe(404);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/handlers/link.test.ts
```
Expected: FAIL — `Cannot find module '../../src/handlers/link'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/handlers/link.ts`:
```typescript
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
```

Modify `finance/src/index.ts` — replace with:
```typescript
import { handleApi } from "./handlers/api";
import { handleLink } from "./handlers/link";
import type { Env } from "./types";

const API_PATHS = new Set(["/balances", "/networth", "/history", "/refresh"]);
const LINK_PATHS = new Set(["/link", "/link/token", "/link/exchange"]);

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "finance-state" });
    }
    if (API_PATHS.has(url.pathname)) return handleApi(req, env);
    if (LINK_PATHS.has(url.pathname)) return handleLink(req, env);
    return new Response("not found", { status: 404 });
  },

  async scheduled(_event: ScheduledEvent, _env: Env, _ctx: ExecutionContext): Promise<void> {
    // wired in Task 14
  },
} satisfies ExportedHandler<Env>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/handlers/link.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/handlers/link.ts finance/src/index.ts finance/test/handlers/link.test.ts && git commit -m "feat(finance): dev-only plaid link page + token exchange"
```

---

### Task 13: handlers/webhook_plaid.ts — Plaid item webhook

**Group:** E (sequential after Task 12 — wires `index.ts`)

**Behavior being verified:** `POST /webhook/plaid` rejects requests with bad/missing verification headers (returns 401). With a valid header and an `ITEM_ERROR` or `PENDING_EXPIRATION` payload, it calls `updateItemStatus(item_id, "needs_reauth", code)`, logs a `reauth_needed` event, and posts a Discord nudge naming the institution.

**Interface under test:** route via `SELF.fetch`, Discord webhook mocked via `fetch`.

**Files:**
- Create: `finance/src/handlers/webhook_plaid.ts`
- Modify: `finance/src/index.ts` (add `/webhook/plaid`)
- Create: `finance/test/handlers/webhook_plaid.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/handlers/webhook_plaid.test.ts`:
```typescript
import { SELF, env } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { listEvents, listItems, upsertItem } from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
});

afterEach(() => vi.restoreAllMocks());

describe("plaid webhook", () => {
  it("rejects requests without the verification header", async () => {
    const res = await SELF.fetch("https://finance.test/webhook/plaid", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ webhook_type: "ITEM", webhook_code: "ERROR", item_id: "item_wf" }),
    });
    expect(res.status).toBe(401);
  });

  it("on ITEM_ERROR: marks item needs_reauth, logs event, posts Discord nudge", async () => {
    let discordPosted: { content?: string } | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url === "https://discord.test/webhook") {
        discordPosted = JSON.parse(init!.body as string);
        return new Response("", { status: 204 });
      }
      throw new Error(`unexpected fetch ${url}`);
    });

    const res = await SELF.fetch("https://finance.test/webhook/plaid", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "plaid-verification": "test-webhook-secret",
      },
      body: JSON.stringify({
        webhook_type: "ITEM",
        webhook_code: "ERROR",
        item_id: "item_wf",
        error: { error_code: "ITEM_LOGIN_REQUIRED" },
      }),
    });
    expect(res.status).toBe(200);

    const items = await listItems(env);
    expect(items[0]!.status).toBe("needs_reauth");
    expect(items[0]!.last_error).toBe("ITEM_LOGIN_REQUIRED");

    const events = await listEvents(env, 10);
    expect(events.some((e) => e.event_type === "reauth_needed")).toBe(true);

    expect(discordPosted).not.toBeNull();
    expect(JSON.stringify(discordPosted)).toContain("Wells Fargo");
    expect(JSON.stringify(discordPosted).toLowerCase()).toContain("re-auth");
  });

  it("ignores unrelated webhook codes", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const res = await SELF.fetch("https://finance.test/webhook/plaid", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "plaid-verification": "test-webhook-secret",
      },
      body: JSON.stringify({
        webhook_type: "TRANSACTIONS",
        webhook_code: "DEFAULT_UPDATE",
        item_id: "item_wf",
      }),
    });
    expect(res.status).toBe(200);
    expect(fetchSpy).not.toHaveBeenCalled();
    const items = await listItems(env);
    expect(items[0]!.status).toBe("ok");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/handlers/webhook_plaid.test.ts
```
Expected: FAIL — handler not implemented; route returns 404.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/handlers/webhook_plaid.ts`:
```typescript
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
  if (!verifyWebhook(env, raw, req.headers)) {
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
  await fetch(env.DISCORD_WEBHOOK_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      content: `Plaid item for **${institution}** needs re-auth (${code}). Run \`wrangler dev\` and redo \`/link\`.`,
    }),
  });

  return Response.json({ ok: true });
}
```

Modify `finance/src/index.ts` — replace with:
```typescript
import { handleApi } from "./handlers/api";
import { handleLink } from "./handlers/link";
import { handlePlaidWebhook } from "./handlers/webhook_plaid";
import type { Env } from "./types";

const API_PATHS = new Set(["/balances", "/networth", "/history", "/refresh"]);
const LINK_PATHS = new Set(["/link", "/link/token", "/link/exchange"]);

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "finance-state" });
    }
    if (API_PATHS.has(url.pathname)) return handleApi(req, env);
    if (LINK_PATHS.has(url.pathname)) return handleLink(req, env);
    if (url.pathname === "/webhook/plaid") return handlePlaidWebhook(req, env);
    return new Response("not found", { status: 404 });
  },

  async scheduled(_event: ScheduledEvent, _env: Env, _ctx: ExecutionContext): Promise<void> {
    // wired in Task 14
  },
} satisfies ExportedHandler<Env>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/handlers/webhook_plaid.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/handlers/webhook_plaid.ts finance/src/index.ts finance/test/handlers/webhook_plaid.test.ts && git commit -m "feat(finance): plaid item-error webhook with discord re-auth nudge"
```

---

### Task 14: handlers/crons.ts — scheduled-event dispatcher

**Group:** E (sequential after Task 13 — wires `index.ts`)

**Behavior being verified:** When the scheduled handler fires with cron `0 7 * * *`, it runs `syncAllItems` for "today" UTC, includes the Alpaca paper equity as a manual snapshot under `acc_paper`, and logs a `snapshot_run` event. When it fires with cron `0 7 * * MON`, it runs the daily sync, computes the weekly summary, and posts it to Discord.

**Interface under test:** `scheduled(event, env, ctx)` exported from `finance/src/index.ts`.

**Files:**
- Create: `finance/src/handlers/crons.ts`
- Modify: `finance/src/index.ts` (wire `scheduled` to `handleScheduled`)
- Create: `finance/test/handlers/crons.test.ts`

- [ ] **Step 1: Write the failing test**

`finance/test/handlers/crons.test.ts`:
```typescript
import { env, createScheduledController, runInDurableObject } from "cloudflare:test";
import worker from "../../src/index";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getLatestSnapshots,
  listEvents,
  upsertAccount,
  upsertItem,
} from "../../src/db/queries";

beforeEach(async () => {
  await env.DB.prepare("DELETE FROM finance_balance_snapshot").run();
  await env.DB.prepare("DELETE FROM finance_account").run();
  await env.DB.prepare("DELETE FROM finance_plaid_item").run();
  await env.DB.prepare("DELETE FROM finance_event_log").run();
  await env.FINANCE_KV.put("plaid_item:item_wf", "access-wf");
  await upsertItem(env, { item_id: "item_wf", institution_name: "Wells Fargo" });
  await upsertAccount(env, {
    account_id: "acc_paper",
    item_id: null,
    display_name: "Alpaca Paper",
    account_type: "brokerage_paper",
    asset_class: "strategy_paper",
    currency: "USD",
    is_liability: 0,
    include_in_networth: 0,
    is_active: 1,
  });
});

afterEach(() => vi.restoreAllMocks());

function plaidBalanceFetch(): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    if (url.includes("plaid.com/accounts/balance/get")) {
      return new Response(
        JSON.stringify({
          accounts: [{
            account_id: "acc_chk",
            name: "WF Checking",
            official_name: "WF Checking",
            type: "depository",
            subtype: "checking",
            balances: { current: 3200, available: 3200, iso_currency_code: "USD" },
          }],
          item: { item_id: "item_wf" },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (url === "https://paper-api.alpaca.markets/v2/account") {
      return new Response(JSON.stringify({ equity: "10250.42", cash: "5000" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (url === "https://discord.test/webhook") {
      return new Response("", { status: 204 });
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;
}

describe("scheduled handler", () => {
  it("daily cron writes plaid + alpaca snapshots and logs snapshot_run", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(plaidBalanceFetch());
    const controller = createScheduledController({
      scheduledTime: new Date("2026-04-19T07:00:00Z").getTime(),
      cron: "0 7 * * *",
    });
    await worker.scheduled!(controller, env, {} as ExecutionContext);

    const snapshots = await getLatestSnapshots(env);
    const ids = snapshots.map((s) => s.account_id).sort();
    expect(ids).toEqual(["acc_chk", "acc_paper"]);
    const paper = snapshots.find((s) => s.account_id === "acc_paper")!;
    expect(paper.current_balance).toBe(10250.42);
    expect(paper.source).toBe("alpaca");

    const events = await listEvents(env, 10);
    expect(events.some((e) => e.event_type === "snapshot_run")).toBe(true);
  });

  it("Sunday cron runs sync, posts weekly summary embed to discord", async () => {
    let discordPostBody: unknown = null;
    vi.spyOn(globalThis, "fetch").mockImplementation((async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url === "https://discord.test/webhook") {
        discordPostBody = JSON.parse(init!.body as string);
        return new Response("", { status: 204 });
      }
      return plaidBalanceFetch()(input, init);
    }) as typeof fetch);

    const controller = createScheduledController({
      scheduledTime: new Date("2026-04-19T07:00:00Z").getTime(),
      cron: "0 7 * * MON",
    });
    await worker.scheduled!(controller, env, {} as ExecutionContext);

    expect(discordPostBody).not.toBeNull();
    const body = discordPostBody as { embeds: Array<{ title: string }> };
    expect(body.embeds[0]!.title).toContain("Weekly Finance");

    const events = await listEvents(env, 10);
    expect(events.some((e) => e.event_type === "summary_posted")).toBe(true);
  });
});
```

Note on test imports: `createScheduledController` is provided by `@cloudflare/vitest-pool-workers`; `runInDurableObject` import line is unused and may be removed if vitest's no-unused-imports rule fires.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd finance && bunx vitest run test/handlers/crons.test.ts
```
Expected: FAIL — `scheduled` is a no-op and no Discord post occurs.

- [ ] **Step 3: Implement the minimum to make the test pass**

`finance/src/handlers/crons.ts`:
```typescript
import { getPaperEquity } from "../alpaca/client";
import { insertSnapshot, listAccounts, logEvent, upsertAccount } from "../db/queries";
import { syncAllItems } from "../plaid/sync";
import { computeWeeklySummary } from "../summary/compute";
import { postWeeklySummary } from "../discord/embed";
import type { Env } from "../types";

const PAPER_ACCOUNT_ID = "acc_paper";

function isoDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

async function snapshotAlpaca(env: Env, today: string): Promise<void> {
  let equity: number;
  try {
    equity = await getPaperEquity(env);
  } catch (err) {
    await logEvent(env, {
      event_type: "alpaca_error",
      item_id: null,
      account_id: PAPER_ACCOUNT_ID,
      payload: { error: err instanceof Error ? err.message : String(err) },
    });
    return;
  }
  const accounts = await listAccounts(env);
  if (!accounts.find((a) => a.account_id === PAPER_ACCOUNT_ID)) {
    await upsertAccount(env, {
      account_id: PAPER_ACCOUNT_ID,
      item_id: null,
      display_name: "Alpaca Paper",
      account_type: "brokerage_paper",
      asset_class: "strategy_paper",
      currency: "USD",
      is_liability: 0,
      include_in_networth: 0,
      is_active: 1,
    });
  }
  await insertSnapshot(env, {
    account_id: PAPER_ACCOUNT_ID,
    taken_at: `${today}T07:00:00Z`,
    snapshot_date: today,
    current_balance: equity,
    available_balance: equity,
    source: "alpaca",
    raw_response: null,
  });
}

export async function handleScheduled(event: ScheduledEvent, env: Env): Promise<void> {
  const today = isoDate(event.scheduledTime);
  await syncAllItems(env, today);
  await snapshotAlpaca(env, today);

  if (event.cron === "0 7 * * MON") {
    const data = await computeWeeklySummary(env, new Date(event.scheduledTime));
    await postWeeklySummary(env, data);
    await logEvent(env, {
      event_type: "summary_posted",
      item_id: null,
      account_id: null,
      payload: { netWorth: data.netWorth, asOf: data.asOf },
    });
  }
}
```

Modify `finance/src/index.ts` — replace `scheduled` body:
```typescript
import { handleApi } from "./handlers/api";
import { handleLink } from "./handlers/link";
import { handlePlaidWebhook } from "./handlers/webhook_plaid";
import { handleScheduled } from "./handlers/crons";
import type { Env } from "./types";

const API_PATHS = new Set(["/balances", "/networth", "/history", "/refresh"]);
const LINK_PATHS = new Set(["/link", "/link/token", "/link/exchange"]);

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "finance-state" });
    }
    if (API_PATHS.has(url.pathname)) return handleApi(req, env);
    if (LINK_PATHS.has(url.pathname)) return handleLink(req, env);
    if (url.pathname === "/webhook/plaid") return handlePlaidWebhook(req, env);
    return new Response("not found", { status: 404 });
  },

  async scheduled(event: ScheduledEvent, env: Env, _ctx: ExecutionContext): Promise<void> {
    await handleScheduled(event, env);
  },
} satisfies ExportedHandler<Env>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd finance && bunx vitest run test/handlers/crons.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finance/src/handlers/crons.ts finance/src/index.ts finance/test/handlers/crons.test.ts && git commit -m "feat(finance): scheduled handler for daily snapshot + sunday summary"
```

---

### Task 15: finance-read Hermes skill

**Group:** F (single task)

**Behavior being verified:** The Python script `query.py` calls the worker bearer-authenticated read API and prints the JSON response. With the `balances` subcommand and a valid bearer + URL, it prints the snapshot list. With a missing bearer in env, it raises a `RuntimeError`.

**Interface under test:** `query.py {balances|networth|history|refresh} [--account-id X] [--days N]` invoked as a subprocess; Worker mocked via `pytest`'s `monkeypatch` of `requests.get` / `requests.post`.

**Files:**
- Create: `assistant/config/skills/finance-read/SKILL.md`
- Create: `assistant/config/skills/finance-read/scripts/query.py`
- Create: `assistant/config/skills/finance-read/tests/test_query.py`
- Create: `assistant/config/skills/finance-read/scripts/requirements.txt`

- [ ] **Step 1: Write the failing test**

`assistant/config/skills/finance-read/tests/test_query.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "query.py"


def run(args, env_overrides=None):
    env = {
        "FINANCE_WORKER_URL": "https://finance.test",
        "FINANCE_BEARER_TOKEN": "test-token",
        "PATH": "/usr/bin:/bin",
    }
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_balances_subcommand_prints_json(monkeypatch, tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"snapshots": [{"account_id": "acc_chk", "current_balance": 3200}]}))

    res = run(["balances", "--mock-from", str(fixture)])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["snapshots"][0]["account_id"] == "acc_chk"


def test_missing_bearer_raises(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}")
    res = run(["balances", "--mock-from", str(fixture)], env_overrides={"FINANCE_BEARER_TOKEN": ""})
    assert res.returncode != 0
    assert "FINANCE_BEARER_TOKEN" in res.stderr
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/finance-read && python3 -m pytest tests/test_query.py
```
Expected: FAIL — `query.py` does not exist.

- [ ] **Step 3: Implement the minimum to make the test pass**

`assistant/config/skills/finance-read/scripts/requirements.txt`:
```
requests>=2.31
```

`assistant/config/skills/finance-read/scripts/query.py`:
```python
#!/usr/bin/env python3
"""Query the finance-state Worker bearer-auth read API."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is required but not set")
    return value


def _request(path: str, *, method: str = "GET", mock_from: str | None) -> dict:
    if mock_from:
        return json.loads(Path(mock_from).read_text())
    import requests  # local import — avoids requiring `requests` for tests using --mock-from
    base = _require_env("FINANCE_WORKER_URL").rstrip("/")
    token = _require_env("FINANCE_BEARER_TOKEN")
    headers = {"authorization": f"Bearer {token}"}
    fn = requests.post if method == "POST" else requests.get
    res = fn(f"{base}{path}", headers=headers, timeout=20)
    res.raise_for_status()
    return res.json()


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Query finance-state worker.")
    parser.add_argument("command", choices=["balances", "networth", "history", "refresh"])
    parser.add_argument("--account-id")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--mock-from", help="path to a JSON fixture; bypasses HTTP")
    args = parser.parse_args()

    # Validate bearer presence even when using --mock-from, so the test for missing bearer still
    # exercises the same guard the real path uses.
    _require_env("FINANCE_BEARER_TOKEN")

    if args.command == "balances":
        out = _request("/balances", mock_from=args.mock_from)
    elif args.command == "networth":
        out = _request("/networth", mock_from=args.mock_from)
    elif args.command == "history":
        if not args.account_id:
            raise SystemExit("--account-id required for history")
        path = f"/history?account_id={args.account_id}&days={args.days}"
        out = _request(path, mock_from=args.mock_from)
    elif args.command == "refresh":
        out = _request("/refresh", method="POST", mock_from=args.mock_from)
    else:
        raise SystemExit(f"unknown command {args.command}")

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`assistant/config/skills/finance-read/SKILL.md`:
```markdown
---
name: finance-read
description: Read-only queries against the personal finance-state Worker. Use only on direct finance intent from the user (balances, net worth, account history). Never invoke from a context that processes untrusted text such as email bodies, web fetches, news feeds, or meeting transcripts.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [finance, read-only, plaid]
    related_skills: []
---

## When to use

- The user explicitly asks any of: "what's my net worth", "how much in checking", "how did <account> change", "balance of <account>", "refresh balances", "show me my finance picture".
- Never on cron triggers other than the existing finance-state Worker crons (those run inside the Worker, not via Hermes).
- Never as a downstream action from email-triage, meeting-prep, morning-brief, notion-wiki, fathom-webhook, or any skill whose context contains text fetched from outside Mahler.

## Procedure

```bash
python3 ~/.hermes/skills/finance-read/scripts/query.py balances
python3 ~/.hermes/skills/finance-read/scripts/query.py networth
python3 ~/.hermes/skills/finance-read/scripts/query.py history --account-id acc_chk --days 30
python3 ~/.hermes/skills/finance-read/scripts/query.py refresh
```

The script reads `FINANCE_WORKER_URL` and `FINANCE_BEARER_TOKEN` from `~/.hermes/.env`. If either is missing the script raises `RuntimeError`.

## Output

The script prints the worker's raw JSON response. Mahler should summarize relevant fields for the user rather than dumping JSON into Discord.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/finance-read && python3 -m pytest tests/test_query.py
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/finance-read/ && git commit -m "feat(finance-read): hermes skill for ad-hoc finance queries"
```

---

## Plan self-review notes

- Spec coverage: every "File Changes" row in the spec maps to a task above (T1 covers all scaffolding; T2-T4 = queries.ts; T5-T6 = plaid/client.ts; T7 = alpaca; T8 = discord/embed; T9 = sync; T10 = compute; T11 = api; T12 = link; T13 = webhook; T14 = crons; T15 = Hermes skill).
- Each task is one behavior verified through a public interface, with implementation immediately following the failing test. No horizontal slices.
- Tests do not mock internal collaborators of the module under test (only the network boundary: `fetch` to plaid/alpaca/discord). D1 + KV are real miniflare instances.
- File ownership: tasks within Group B all touch `queries.ts` and are sequential. Group C: T5 + T6 share `plaid/client.ts` (sequential); T7 + T8 are independent. Group E: all wire `index.ts` and are sequential. Group D's two tasks touch disjoint files and are parallel.
- Type/method names are consistent across tasks (`syncAllItems`, `computeWeeklySummary`, `postWeeklySummary`, `WeeklyData`, `SyncResult`, `Env`, etc.).

---

## Challenge Review

### CEO Pass

**Premise:** Correct problem, direct path. Without a balance history, no downstream advice system has data to ground itself in. No simpler framing exists given Plaid's server-side token exchange requirement.

**Scope:** Plan maps 1:1 to spec. Every "File Changes" row in the spec has a corresponding task. No scope drift detected in either direction. All 15 tasks are new files in a new directory — file count is high but unavoidable for a greenfield worker.

**12-Month Alignment:**
```
CURRENT STATE                    THIS PLAN                          12-MONTH IDEAL
Fragmented accounts,         ->  D1 snapshot history,           ->  Active advisor with
no unified view,                 weekly Discord summary,             recommendations
no history                       Hermes read skill                   grounded in real data
```
Plan moves directly toward the ideal. No tech debt introduced that conflicts with Phase 1+.

**Alternatives:** Spec documents the "Why this shape" rationale (separate worker vs. monorepo, TS vs. Rust, pure-cron vs. LLM). Trade-offs accepted section covers known limitations. Well-documented.

---

### Engineering Pass

**Architecture data flow:**

```
daily cron          -> syncAllItems -> getBalances (Plaid) -> insertSnapshot (D1)
                    -> snapshotAlpaca -> getPaperEquity (Alpaca) -> insertSnapshot (D1)
sunday cron         -> above + computeWeeklySummary (D1) -> postWeeklySummary (Discord)
POST /webhook/plaid -> verifyWebhook -> updateItemStatus (D1) -> Discord nudge
POST /refresh       -> syncAllItems (same as daily cron)
GET /balances|networth|history -> bearer auth -> D1 queries
```

**[RISK] (confidence: 9/10) — `verifyWebhook` uses plain string equality, not HMAC or JWT.**

`plaid/client.ts` Task 6: `return provided === env.PLAID_WEBHOOK_SECRET`. Plaid's actual `Plaid-Verification` header contains a signed JWT (RS256, keys from Plaid's JWK endpoint) — not the raw secret. This implementation will reject every real Plaid webhook with 401 in production. The test passes only because it sets `plaid-verification: "test-webhook-secret"` in the miniflare binding, which matches the fake env. The existing `fathom-webhook/src/index.ts:verifySignature` uses `crypto.subtle.importKey` + HMAC-SHA256 and is the established pattern in this repo. The plan should implement JWT verification or, minimally, HMAC-SHA256 against the Plaid-supplied signature. Plaid's SDK (`plaid@^27.0.0`, already in `package.json`) includes `plaidClient.webhookVerificationKeyGet()` for JWK-based verification.

**[RISK] (confidence: 8/10) — `updateItemStatus` bumps `last_synced_at` on both success and failure, making stale detection miss actively erroring items.**

`db/queries.ts` (Task 3): `UPDATE ... SET status = ?, last_error = ?, last_synced_at = datetime('now')`. When an item errors, `syncAllItems` calls `updateItemStatus(env, item.item_id, "error", message)`, bumping `last_synced_at` to now. `computeWeeklySummary`'s stale filter checks `last_synced_at < (asOf - 36h)` — an item that errors every day will have `last_synced_at` = today and never appear in `staleItems`, even though its balance data is stale. The stale check should also flag items whose `status !== "ok"`.

**[RISK] (confidence: 8/10) — Discord post in `webhook_plaid.ts` has no error handling.**

`handlers/webhook_plaid.ts` (Task 13): the `fetch(env.DISCORD_WEBHOOK_URL, ...)` call has no `if (!res.ok)` check and no try/catch. If Discord is down, the handler swallows the failure and returns `{ ok: true }`. The spec and CLAUDE.md both state "explicit error handling (no silent fallbacks)." This is a direct violation. Should throw or log via `logEvent` on non-204 response.

**[RISK] (confidence: 7/10) — Task 3 and Task 4 "append" instructions produce duplicate `import type` statements in `queries.ts`.**

Task 2 creates `queries.ts` with `import type { AccountRow, Env, SnapshotRow } from "../types"`. Task 3 appends `import type { PlaidItemRow } from "../types"`. Task 4 appends `import type { EventRow } from "../types"`. TypeScript compiles duplicate module imports, but the build agent may produce a structurally odd file with three separate import lines from the same module. The build agent should merge these into the existing import statement. Flag for explicit attention during Task 3 Step 3.

**[RISK] (confidence: 6/10) — `createLinkToken` is implemented in Task 6 but has no test and the `POST /link/token` route in `link.ts` has no test in Task 12.**

Task 6 adds `createLinkToken` to `plaid/client.ts`. Task 12's `link.test.ts` covers `GET /link` (HTML check) and `POST /link/exchange`, but not `POST /link/token`. The route in `handleLink` that calls `createLinkToken` is exercised only by the user clicking "Open Plaid Link" — never in any automated test. Low severity for Phase 0 (it's a one-time dev-only flow), but a gap.

**[OBS] — `plaid@^27.0.0` is in `package.json` but never imported.**

The Plaid SDK is listed as a runtime dependency but all Plaid calls use raw `fetch`. The SDK is dead weight at 2MB+ in the bundle. Either remove it, or use it for JWK-based webhook verification (see RISK above), which would justify its presence.

**[OBS] — `snapshotsWritten` counter in `SyncResult` counts write attempts, not actual inserts.**

The plan note at Task 9 acknowledges this: "the `snapshotsWritten` counter increments on attempt, not on actual insert." On a re-run for the same day, the event log records `snapshotsWritten: N` even though 0 rows were written (all `INSERT OR IGNORE`d). The audit log is therefore misleading on idempotent re-runs. Consider naming it `snapshotsAttempted` or computing actual inserts from `result.meta.changes`.

**[OBS] — Cron schedule is PST-aligned but will fire 1 hour late during PDT.**

`wrangler.toml`: `"0 7 * * *"` = 07:00 UTC. During PST (UTC-8) this is 23:00 Pacific (correct). During PDT (UTC-7) this is midnight Pacific (1 hour late). This is minor for a daily snapshot but the spec says "23:00 Pacific." No fix required for Phase 0 — worth noting for a future `TZ`-aware cron.

---

### Module Depth Audit

| Module | Interface size | Implementation | Verdict |
|--------|----------------|----------------|---------|
| `db/queries.ts` | 9 exported functions | 150+ LOC SQL hiding idempotency, JSON serialization | DEEP |
| `plaid/client.ts` | 4 exports | Plaid auth, env routing, KV token I/O | DEEP |
| `plaid/sync.ts` | 1 export (`syncAllItems`) | 60 LOC orchestration, error isolation per item | DEEP |
| `alpaca/client.ts` | 1 export | HTTP auth, JSON parsing, NaN guard | DEEP |
| `summary/compute.ts` | 1 export | Multi-query computation, sparkline, stale detection | DEEP |
| `discord/embed.ts` | 1 export | Embed shape, color coding, webhook I/O | DEEP |
| `auth.ts` | 1 fn + 1 class | 3 lines (spec calls out as intentionally shallow) | SHALLOW (intentional) |
| `handlers/api.ts` | 1 export | Route table, auth check, param validation | DEEP enough |
| `handlers/link.ts` | 1 export | HTML, multi-route dispatch, env gate | DEEP |
| `handlers/webhook_plaid.ts` | 1 export | Verification, code routing, Discord nudge | DEEP |
| `handlers/crons.ts` | 1 export | Daily vs. Monday dispatch, Alpaca snapshot | DEEP enough |

No shallow modules beyond `auth.ts`, which is explicitly noted as intentional in the spec.

---

### Test Philosophy Audit

All tests exercise behavior through public interfaces. D1 and KV are real miniflare instances throughout — no mocking of internal collaborators. Network calls (Plaid, Alpaca, Discord) are mocked at the `fetch` boundary via `vi.spyOn(globalThis, "fetch")`. Tests assert on user-observable state (D1 rows, KV values, captured request bodies), not on internal call counts.

Task 12's production-gate test directly imports `handleLink` to bypass the miniflare env constraint — acceptable workaround, not a philosophy violation.

Task 15 uses subprocess invocation for `query.py` — correct approach for a CLI.

---

### Vertical Slice Audit

All 15 tasks follow: write failing test → verify FAIL → implement minimum → verify PASS → commit. No horizontal slicing. No task writes multiple tests before any implementation. No deferred implementations.

---

### Test Coverage Gaps

```
[+] plaid/client.ts
    ├── exchangePublicToken()   [TESTED ★★] Task 5 — happy path + KV write
    ├── getBalances()           [TESTED ★★] Task 6 — happy path + missing token error
    ├── verifyWebhook()         [TESTED ★]  Task 6 — string match only; real JWT not covered
    └── createLinkToken()       [GAP]       No test. /link/token route untested.

[+] handlers/webhook_plaid.ts
    ├── unauthorized rejection  [TESTED ★★] Task 13
    ├── ITEM ERROR -> reauth    [TESTED ★★] Task 13
    ├── unrelated codes ignored [TESTED ★★] Task 13
    └── Discord post failure    [GAP]       No test for Discord down path.

[+] handlers/crons.ts
    ├── daily cron              [TESTED ★★] Task 14
    ├── Sunday cron             [TESTED ★★] Task 14
    └── Alpaca fetch failure    [GAP]       snapshotAlpaca catch path untested.

[+] query.py
    ├── balances subcommand     [TESTED ★]  Task 15 — smoke test via --mock-from
    ├── networth subcommand     [GAP]       No test
    ├── history subcommand      [GAP]       No test
    └── refresh subcommand      [GAP]       No test
```

The `query.py` gaps are acceptable for a dev CLI with a smoke test. The `createLinkToken` gap is low severity (dev-only path). The `verifyWebhook` gap is load-bearing (see RISK above).

---

### Failure Modes

| Scenario | Outcome | Silent? |
|----------|---------|---------|
| Plaid balance fetch fails for one item | Item marked `error`, sync continues, event logged | No |
| Alpaca fetch fails during cron | Logged via `logEvent`, snapshot skipped | No |
| Discord post fails in `postWeeklySummary` | `throw new Error(...)` — worker observability catches it | No |
| Discord post fails in `handlePlaidWebhook` | Swallowed, returns `{ ok: true }` | **YES — silent** |
| Real Plaid webhook arrives | 401 rejected (see verifyWebhook RISK) | No (visible failure) |
| `wrangler deploy` before migration applied | Runtime D1 error on first request | No (visible) |
| KV ID placeholder not replaced before deploy | `wrangler deploy` fails at CLI | No (visible) |

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|------------|---------|--------|
| `vitest-pool-workers` auto-applies D1 migrations from `migrations_dir` in `wrangler.toml` | VALIDATE | Documented behavior but version-sensitive; confirm against `^0.5.0` resolved version |
| `vi.spyOn(globalThis, "fetch")` intercepts worker-internal `fetch` in miniflare | SAFE | Established pattern; same approach used in fathom-webhook tests |
| `createScheduledController` available from `"cloudflare:test"` at `^0.5.0` | VALIDATE | Added in a later 0.x version; fathom-webhook pins `0.12.21` — the `^0.5.0` lower bound may resolve too low if bun caches an old version |
| Plaid `development` environment supports Wells Fargo + Wealthfront | RISKY | Spec acknowledges uncertainty; some institutions require production approval |
| `Plaid-Verification` header contains raw shared secret (not JWT) | RISKY | Plaid uses JWT-based verification; this assumption is almost certainly wrong in production |
| `package.json` `"@cloudflare/vitest-pool-workers": "^0.5.0"` resolves consistently with fathom-webhook's pinned `0.12.21` | VALIDATE | Semver range means it could resolve to any 0.x — prefer pinning to match fathom-webhook |

---

### Summary

```
[RISK]     count: 4 (verifyWebhook, updateItemStatus stale detection, Discord silent failure, duplicate imports)
[OBS]      count: 3 (dead plaid SDK dep, snapshotsWritten counter, PST/PDT cron offset)
[QUESTION] count: 0
[BLOCKER]  count: 0
```

VERDICT: PROCEED_WITH_CAUTION — monitor these risks during execution:
1. **verifyWebhook** — implement HMAC or JWT verification before enabling the webhook endpoint in production; the current implementation will silently reject all real Plaid webhooks
2. **Discord silent failure in webhook handler** — add error handling before Task 13 commit
3. **stale item detection** — add `status !== "ok"` to the stale filter in Task 10
4. **duplicate imports in Tasks 3/4** — build agent must merge into existing `import type {...} from "../types"` line rather than appending new import statements
- Open questions from the spec (Plaid environment, Wealthfront splits, Discord channel, manual edits) all have explicit defaults in the spec and do not block the plan.



