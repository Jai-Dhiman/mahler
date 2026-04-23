# Personal Finance OS — Phase 0 Design

**Goal:** Capture daily balance snapshots from Wells Fargo, Wealthfront, and Alpaca paper into D1, and post a weekly Discord summary every Sunday — all read-only, with no LLM in the primary path.

**Not in scope:**
- Any write actions (transfers, contributions, trades)
- Validator service, executor, proposal protocol, rule DSL, signing, cooling-off
- Plaid Transactions sync; spending categorization
- Apple Card integration (visible via checking outflows)
- Web dashboard / HTML UI
- Recommendation engine / advisor logic
- Multi-user support
- Currency conversion (USD only)
- The `finance-playbook/` advisor knowledge base (parallel non-code workstream by the user)

## Problem

Cash sits idle and net worth is fragmented across Wells Fargo (checking), Wealthfront (cash + investment), and Alpaca (paper). There is no unified view, no historical record, and no agent surface that can reason across them. Without a data foundation, the eventual active advisor (Phase 2+) has nothing to ground its recommendations in.

The Phase 0 problem is narrower: produce a reliable, idempotent, append-only balance history and surface it weekly. Everything downstream — rebalancing, sweeps, advice — depends on this existing first.

## Solution (from the user's perspective)

- Once: run `wrangler dev` locally, open `localhost:8787/link`, complete Plaid Link for Wells Fargo, then again for Wealthfront. Access tokens stored in KV.
- Every day at 23:00 Pacific: a Worker cron polls Plaid + Alpaca, writes one snapshot row per account into `finance_balance_snapshot` (idempotent: re-running the same day is a no-op).
- Every Sunday at 23:00 Pacific: a Worker cron computes the weekly summary and posts a Discord embed to the home channel showing current net worth, per-account balances, week-over-week deltas, and a 4-week sparkline. No LLM involvement.
- When a Plaid item breaks: the Plaid webhook handler posts a Discord nudge ("Wells Fargo Plaid item needs re-auth, run `wrangler dev` and redo `/link`").
- Optionally, in Discord: the user can ask Mahler "what's my net worth" or "how did checking change this month" and Hermes invokes the `finance-read` skill, which calls the Worker's bearer-authenticated read API.

## Design

A new Cloudflare Worker `finance/` (sibling to `assistant/` and `traderjoe/`), TypeScript, deployed via `wrangler`. Shares the existing `mahler-db` D1 instance with prefixed `finance_*` tables. Uses a new dedicated KV namespace `FINANCE_KV` for Plaid access tokens and runtime state.

### Why this shape

- **Separate Worker, shared DB.** Isolation from the Hermes process (no Plaid creds in agent context) and from the Trader Joe Worker (separate secrets, separate deploy cadence, no chance a finance bug crashes a trading cron). Sharing `mahler-db` keeps cross-system reporting possible later without a sync layer.
- **TypeScript over Rust.** Plaid's official SDK is TS-first. Phase 0 is pure REST orchestration — none of Rust's strengths apply here.
- **Pure-cron weekly summary, no LLM.** The primary user-visible surface (the Sunday post) has zero prompt-injection surface because no model sees the data on its way to Discord.
- **Hermes `finance-read` skill is optional and narrow.** Triggered only on explicit finance intent, not declared as a dependency of any untrusted-input skill (`email-triage`, `meeting-prep`, `morning-brief`, etc.). Residual injection risk is bounded to balance numbers leaking to the user's own private Discord channel.
- **Idempotent daily snapshots via `UNIQUE(account_id, snapshot_date)`.** Re-running the cron mid-day, or after a transient failure, never double-writes.
- **Append-only `finance_event_log`.** Cheap to write, valuable for "did Sunday's summary actually run", and seeds the eventual audit log when the write side lands in Phase 1+.
- **Plaid Link is dev-only (`ENVIRONMENT === "dev"`).** Production deploy does not expose `/link`. Re-auth requires running `wrangler dev` from the user's laptop — acceptable friction (~quarterly) for not having a public auth surface.
- **Plaid network calls mocked in tests via `fetch` interception.** Real Plaid sandbox is used manually during local Link setup, not in the test suite.

### Trade-offs accepted

- **No transaction-level data.** Weekly summary cannot show spending breakdown. Mitigation: deferred to Phase 0.5 once balance summary has run for ≥4 weeks.
- **Apple Card untracked.** Spending shows indirectly as Wells Fargo outflow on the auto-pay date. Mitigation: noted in summary footer; Phase 0.5 may add manual entry.
- **Wealthfront via Plaid is known-flaky.** Mitigation: item-error webhook + Discord re-auth nudge make breakage visible within minutes; the snapshot table tolerates gaps.
- **Bearer-token auth on read API (single shared secret).** No per-caller scoping. Mitigation: token in `wrangler secret`, rotated quarterly. Single user, single caller (Hermes); not worth a multi-tenant key system at this scale.
- **Daily snapshot loses intra-day movement.** Mitigation: `POST /refresh` allows on-demand poll; intra-day data isn't needed for a weekly summary.

## Modules

All deep modules are tested through their public interfaces only. No tests mock internal collaborators of the module under test. Plaid, Alpaca, and Discord network calls are mocked at the `fetch` boundary.

- **`db/queries.ts`**
  - **Interface:** `getLatestSnapshots(env)`, `getHistory(env, accountId, days)`, `insertSnapshot(env, row)`, `listAccounts(env)`, `listItems(env)`, `upsertItem(env, item)`, `upsertAccount(env, account)`, `updateItemStatus(env, itemId, status, error?)`, `logEvent(env, event)`
  - **Hides:** every SQL statement, the `INSERT OR IGNORE` idempotency rule, the `UNIQUE(account_id, snapshot_date)` constraint handling, JSON serialization of `raw_response` and `payload`.
  - **Tested through:** the function set above, against a miniflare D1.

- **`plaid/client.ts`**
  - **Interface:** `createLinkToken(env)`, `exchangePublicToken(env, publicToken)`, `getBalances(env, itemId)`, `verifyWebhook(env, body, headers)`
  - **Hides:** Plaid SDK construction, environment routing (sandbox/development/production), KV token read/write keyed `plaid_item:{item_id}`, webhook JWT verification.
  - **Tested through:** the four functions above, with `fetch` to `*.plaid.com` intercepted.

- **`plaid/sync.ts`**
  - **Interface:** `syncAllItems(env, today: string): SyncResult`
  - **Hides:** iterating active items, calling `getBalances`, writing snapshots via `insertSnapshot`, reconciling Plaid `account_id`s with `finance_account` rows (auto-`upsertAccount` on first sight), classifying Plaid errors, calling `updateItemStatus` on failure, calling `logEvent` for both success and failure.
  - **Tested through:** `syncAllItems` returning `SyncResult` and post-call D1 state assertions.

- **`alpaca/client.ts`**
  - **Interface:** `getPaperEquity(env): Promise<number>`
  - **Hides:** Alpaca paper API URL, header auth, JSON parsing.
  - **Tested through:** `getPaperEquity` with `fetch` to `paper-api.alpaca.markets` intercepted.

- **`summary/compute.ts`**
  - **Interface:** `computeWeeklySummary(env, asOf: Date): WeeklyData`
  - **Hides:** querying snapshots for the relevant 28-day window, computing per-account week-over-week delta, summing net worth (excluding `include_in_networth = 0`), building a 4-point sparkline series, separating strategy paper P&L.
  - **Tested through:** `computeWeeklySummary` against seeded D1 fixtures.

- **`discord/embed.ts`**
  - **Interface:** `postWeeklySummary(env, data: WeeklyData): Promise<void>`
  - **Hides:** Discord embed shape, color coding (green/red on delta sign), webhook URL, retry on 429.
  - **Tested through:** `postWeeklySummary` with `fetch` to `discord.com` intercepted; assertion is on the captured request body shape.

- **`handlers/link.ts`**
  - **Interface:** `handleLink(req, env): Promise<Response>`
  - **Hides:** the HTML page that loads Plaid Link JS, the `/link/exchange` POST that calls `exchangePublicToken`, `upsertItem`, `logEvent`, the `ENVIRONMENT === "dev"` gate.
  - **Tested through:** `handleLink` with two requests: GET `/link` returns HTML containing the Plaid Link script; POST `/link/exchange` with a public token results in an item row and a KV access token.

- **`handlers/webhook_plaid.ts`**
  - **Interface:** `handlePlaidWebhook(req, env): Promise<Response>`
  - **Hides:** webhook verification, code routing (`ITEM_ERROR`, `PENDING_EXPIRATION`, others ignored), `updateItemStatus`, posting Discord re-auth nudge.
  - **Tested through:** `handlePlaidWebhook` with simulated Plaid webhook payloads; assertions on D1 status and captured Discord call.

- **`handlers/api.ts`**
  - **Interface:** `handleApi(req, env): Promise<Response>`
  - **Hides:** bearer-token check, route table (`GET /balances`, `GET /networth`, `GET /history`, `POST /refresh`), JSON serialization, error responses.
  - **Tested through:** `handleApi` with bearer headers; both happy path and 401 on missing/wrong token.

- **`handlers/crons.ts`**
  - **Interface:** `handleScheduled(event: ScheduledEvent, env): Promise<void>`
  - **Hides:** dispatching the daily snapshot cron (`0 7 * * *` UTC = 23:00 PT) to `syncAllItems`, dispatching the Sunday summary cron (`0 7 * * MON` UTC = Sunday 23:00 PT) to `computeWeeklySummary` + `postWeeklySummary`, error logging via `logEvent`.
  - **Tested through:** invoking `handleScheduled` with a mocked `ScheduledEvent` for each cron expression and asserting downstream side effects.

- **`assistant/config/skills/finance-read/scripts/query.py`**
  - **Interface:** `python3 query.py {balances|networth|history|refresh} [--account ID] [--days N]`
  - **Hides:** loading bearer + worker URL from `~/.hermes/.env`, HTTP call, JSON-to-human formatting.
  - **Tested through:** a single smoke test invoking `query.py balances` against a stubbed worker URL.

Shallow modules (`index.ts` router, `auth.ts`, `crons.ts` dispatcher) are intentionally shallow — pure routing logic. Each delegates immediately to a deep module.

## File Changes

| File | Change | Type |
|------|--------|------|
| `finance/wrangler.toml` | CF Worker config: name, main, two cron triggers, D1 binding to `mahler-db`, KV binding `FINANCE_KV`, `[vars] ENVIRONMENT`, observability | New |
| `finance/package.json` | Deps: `plaid@latest`, `vitest`, `@cloudflare/vitest-pool-workers`, `wrangler` | New |
| `finance/tsconfig.json` | TS strict, ES2022, Workers types | New |
| `finance/.dev.vars.example` | Template for `PLAID_CLIENT_ID`, `PLAID_SECRET_DEV`, `PLAID_WEBHOOK_SECRET`, `BEARER_TOKEN`, `DISCORD_WEBHOOK_URL`, `ALPACA_PAPER_KEY_ID`, `ALPACA_PAPER_SECRET` | New |
| `finance/CLAUDE.md` | Project conventions, deploy commands, schema reference | New |
| `finance/vitest.config.ts` | `@cloudflare/vitest-pool-workers` config pointing at `wrangler.toml` | New |
| `finance/migrations/0001_init.sql` | `finance_plaid_item`, `finance_account`, `finance_balance_snapshot`, `finance_event_log` (+ unique constraint, index) | New |
| `finance/src/index.ts` | `export default { fetch, scheduled }` — route table | New |
| `finance/src/types.ts` | `Env`, `PlaidItemRow`, `AccountRow`, `SnapshotRow`, `EventRow`, `WeeklyData`, `SyncResult` | New |
| `finance/src/auth.ts` | `requireBearer(req, env)` — throws on bad/missing token | New |
| `finance/src/db/queries.ts` | All D1 access functions listed in Modules above | New |
| `finance/src/plaid/client.ts` | Plaid SDK wrapper + KV token I/O + webhook verify | New |
| `finance/src/plaid/sync.ts` | `syncAllItems` orchestration | New |
| `finance/src/alpaca/client.ts` | `getPaperEquity` | New |
| `finance/src/summary/compute.ts` | `computeWeeklySummary` | New |
| `finance/src/discord/embed.ts` | `postWeeklySummary` | New |
| `finance/src/handlers/crons.ts` | Scheduled-event dispatcher | New |
| `finance/src/handlers/link.ts` | Dev-only Link page + exchange | New |
| `finance/src/handlers/webhook_plaid.ts` | Plaid item-error webhook | New |
| `finance/src/handlers/api.ts` | Bearer-auth read API + on-demand refresh | New |
| `finance/test/*.test.ts` | Integration tests per task; co-located by feature | New |
| `assistant/config/skills/finance-read/SKILL.md` | Hermes skill metadata, narrow `When to use` | New |
| `assistant/config/skills/finance-read/scripts/query.py` | Python CLI calling Worker via bearer | New |

## Phase 0 exit criteria (informational, not enforced by the plan)

1. Wells Fargo + Wealthfront Plaid items have run for ≥4 weeks without manual intervention (auto-recovery from one expected re-auth event counts).
2. Sunday Discord summary has fired ≥4 consecutive Sundays with no missing/garbage data.
3. The user can name at least one thing they want the system to *do* in response — informs Phase 1's first write action.
4. `finance_balance_snapshot` has 30+ days of clean rows per account.
5. The parallel `finance-playbook/` workstream has ≥5 advisor heuristics drafted.

## Open Questions

- **Q: Plaid environment for Wells Fargo and Wealthfront.** Plaid `development` allows up to 100 live items free; `production` requires application approval and per-call billing. Default: start in `development` — sufficient for two items, no application paperwork. Switch to `production` only if a third institution pushes past the limit.
- **Q: Wealthfront cash vs investment account split.** Wealthfront via Plaid often returns separate accounts for the cash account and each investment account. Default: auto-`upsertAccount` on first sight, with `display_name` taken from Plaid's `official_name`. The user can rename rows manually in D1 if needed.
- **Q: Discord channel for finance posts.** Default: same `home_channel` Mahler already uses (single-user server, single channel). If clutter becomes a problem in Phase 0.5, add a `FINANCE_CHANNEL_ID` env var.
- **Q: How to surface manual D1 edits.** The schema permits `source = 'manual'` snapshots, but Phase 0 has no UI for them. Default: edit via `wrangler d1 execute mahler-db --command "..."` when needed (rare in Phase 0). Phase 0.5 may add a `/finance set-balance` Hermes verb.
