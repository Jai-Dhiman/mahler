# Hermes Chief of Staff Design

**Goal:** A persistent AI assistant on Discord that triages email across Gmail and Outlook (including junk folders), delivers opinionated morning briefs and evening wraps, and serves as the foundation for task management, calendar intelligence, relationship tracking, and self-improvement in later phases.

**Not in scope (Phase 1):**
- Notion task integration (Phase 2)
- Google Calendar integration (Phase 2)
- Meeting prep and post-meeting extraction (Phase 2)
- Relationship CRM / per-person context (Phase 3)
- Email draft composition and approval gates (Phase 4)
- Wiki bridge (Phase 5)
- Kaizen self-improvement loop (Phase 6)

## Problem

The user manages two active email accounts (Gmail + Outlook) with important emails frequently landing in Outlook's junk folder. There is no unified triage -- the user manually checks multiple inboxes and folders throughout the day, missing time-sensitive items. There is no structured start-of-day or end-of-day rhythm. The user wants an autonomous agent that handles this without being asked.

## Solution (from the user's perspective)

A Discord bot with an opinionated chief-of-staff personality runs 24/7. Every 15 minutes it silently scans both email accounts (all folders including junk). At 9am the user gets a morning brief on Discord summarizing top priorities, overnight email highlights, and anything that needs attention. At 6pm the user gets an evening wrap covering what happened, what stalled, and what to prep for tomorrow. If something urgent arrives mid-day, the bot interrupts immediately. The user can also ask the bot questions about their email at any time and get context-rich answers.

## Design

### Architecture

Four layers, two runtimes (TypeScript for Cloudflare edge, Python for Hermes):

**Cloudflare Worker (TypeScript)**
- Handles Email Routing (push-based inbound email on custom domain)
- Cron triggers (every 15 min for email triage, 9am for morning brief, 6pm for evening wrap)
- Routes requests to the Discord Gateway DO and Container

**Durable Object: Container Keeper (TypeScript)**
- Backs the Cloudflare Container (Containers are DO-managed)
- Alarm fires every 30s to keep the Container alive and prevent scale-to-zero
- Note: DO WebSocket Hibernation only works for INBOUND WebSockets (DO as server). The Discord Gateway is an OUTBOUND WebSocket (our system connects TO Discord), so hibernation does not apply. Instead, Hermes runs its native Discord gateway inside the Container, and the DO alarm keeps the Container alive.

**Cloudflare Container: Hermes Runtime (Python)**
- Runs Hermes Agent with its native Discord gateway (persistent outbound WebSocket to Discord)
- Hermes handles Discord messages, cron jobs, and skill execution natively
- On startup: calls Worker proxy API (`/api/kv/*`, `/api/d1/triage-state`) to hydrate MEMORY.md, SOUL.md, USER.md, priority-map.md, and triage cursors into the Container filesystem
- Custom skills call Python API clients for Gmail and Microsoft Graph
- Periodic flush: dirty state pushed back to KV/D1 via Worker proxy API
- IMPORTANT: Containers do NOT have direct access to D1 or KV bindings. All storage operations from inside the Container go through the Worker's proxy API endpoints.

**Storage**
- D1: email_triage_log (classified emails + timestamps), triage_state (last-seen cursors per account/folder)
- KV: MEMORY.md, SOUL.md, USER.md, priority-map.md (markdown files the agent reads/writes)
- Email Routing: forwards copies of inbound email on custom domain to Worker for real-time processing

### Model

`x-ai/grok-4.1-fast` via OpenRouter. Configured in Hermes's `~/.hermes/config.yaml` (mapped from `hermes/cli-config.yaml` in the repo). All tool definitions pruned via `platform_toolsets` to reduce the default 14k token overhead -- only `web`, `terminal`, `file`, `skills`, `todo`, `cronjob` toolsets enabled.

### Email Triage Pipeline

```
Cron (every 15 min) OR Email Routing (real-time inbound)
  |
  v
Python scripts fetch new emails since last cursor
  - gmail_client.py: Gmail API, all folders
  - outlook_client.py: Microsoft Graph API, all folders including Junk
  |
  v
Emails deduplicated against D1 triage log (by message ID)
  |
  v
New emails passed to Hermes (LLM) for classification:
  - URGENT: immediate Discord alert
  - NEEDS_ACTION: queued for next brief
  - FYI: included in evening wrap
  - NOISE: silently dropped, logged
  |
  v
Classifications + metadata written to D1 email_triage_log
Cursors updated in D1 triage_state
```

The LLM classifies using the priority map (`priority-map.md`) for weighting. The priority map is a user-editable markdown file listing what matters this week/month -- domains, senders, topics, projects. The agent reads it before every classification pass.

### Daily Briefs

**Morning brief (9am cron):**
- Reads: overnight triage results from D1, priority-map.md from KV
- LLM synthesizes: top items needing attention, anything urgent that arrived overnight, the day's priorities from the priority map
- Personality: opinionated -- flags overload, calls out patterns ("third email from X this week with no response from you")
- Delivered via Discord REST API to a designated channel

**Evening wrap (6pm cron):**
- Reads: full day's triage results from D1
- LLM synthesizes: what was handled, what's still pending, what to prep for tomorrow
- Delivered via Discord REST API

### Email Routing (Real-Time Inbound)

Cloudflare Email Routing receives emails at `*@yourdomain.com`. The Worker extracts headers and body, stores in D1, and if the sender/subject matches URGENT patterns from the priority map, immediately triggers classification and Discord alert. This supplements the 15-minute polling cycle for emails forwarded from Gmail/Outlook to the custom domain.

Limitation: Outlook junk folder emails will not forward (forwarding rules execute after junk filtering). The Microsoft Graph API poll every 15 minutes catches these.

### Discord Bot Identity

Defined in `SOUL.md`. The bot is an opinionated chief of staff, not a passive assistant:
- Takes positions on priorities and time allocation
- Pushes back when the user is overcommitting
- Flags patterns (repeated ignored emails, growing backlogs)
- Concise, direct, no filler
- Has a name (configurable in SOUL.md)

### State Sync

The Container's filesystem is ephemeral. A state-sync layer bridges KV/D1 to local files. Because Containers cannot access KV or D1 directly, all sync operations go through the Worker's proxy API.

**On wake (container start, via container-init.sh):**
1. Call `GET /api/kv/state%2F{filename}` for each state file (MEMORY.md, SOUL.md, USER.md, priority-map.md) -> write to Container filesystem
2. Call `GET /api/d1/triage-state` -> write cursors to local triage-state.json

**On flush (after each triage cycle, from Python skill scripts):**
1. Read changed markdown files from Container filesystem
2. Call `PUT /api/kv/state%2F{filename}` to write changed files back to KV
3. Call `POST /api/d1/triage-log` for each new triage record
4. Call `POST /api/d1/query` to update triage_state cursors

The Worker proxy API (`/api/kv/*`, `/api/d1/*`) is the bridge between the Container runtime and Cloudflare storage. The `worker_proxy.py` module in the skill scripts directory provides Python helper functions for these calls.

### Fail Loud

Every failure produces a Discord message:
- API auth expired: "Outlook auth expired. Re-run setup."
- API rate limited: "Gmail rate limited -- triage delayed, retrying in 5 min."
- Container crash: "I crashed during triage. Restarting. Last successful run: [timestamp]."
- Model unreachable: "OpenRouter is down -- skipping this triage cycle."

No silent failures. If the user doesn't hear from the bot, something is wrong.

## Modules

### gmail_client.py
- **Interface:** `fetch_recent(folder: str, since: datetime) -> list[Email]`, `search(query: str) -> list[Email]`, `get_folders() -> list[str]`
- **Hides:** OAuth2 credential management and refresh, Gmail API pagination, MIME parsing, rate limit retry logic, folder ID resolution
- **Tested through:** Integration tests calling real Gmail API with a test account

### outlook_client.py
- **Interface:** `fetch_recent(folder: str, since: datetime) -> list[Email]`, `fetch_junk(since: datetime) -> list[Email]`, `get_folders() -> list[str]`
- **Hides:** MSAL authentication, Microsoft Graph API pagination, delta queries for incremental sync, junk folder access, retry logic
- **Tested through:** Integration tests calling real Microsoft Graph API with a test account

### brief_builder.py
- **Interface:** `build_morning_brief(triage_db_path: str, priority_map_path: str) -> BriefData`, `build_evening_wrap(triage_db_path: str) -> WrapData`
- **Hides:** D1 query construction (via Worker proxy), priority map parsing, email aggregation and scoring, time-window filtering
- **Tested through:** Unit tests with a local SQLite database seeded with known triage data (test-only; production calls use Worker proxy)

### worker_proxy.py
- **Interface:** `kv_get(key) -> str | None`, `kv_put(key, value) -> None`, `get_triage_state() -> list[dict]`, `log_triage_record(record) -> None`, `d1_query(sql, params) -> list[dict]`
- **Hides:** HTTP calls to the Worker proxy API (`WORKER_URL` env var), request/response serialization
- **Tested through:** Integration tests with a local wrangler dev server

### container-keeper.ts (Durable Object)
- **Interface:** `alarm()` for keepalive, `fetch()` for Worker -> Container routing
- **Hides:** Container lifecycle management, alarm rescheduling (cascading pattern from DO guide), health check logic, restart-on-crash
- **Tested through:** Integration test with vitest + @cloudflare/vitest-pool-workers: verify alarm fires and Container stays alive

### state-sync.ts
- **Interface:** `hydrate(kv: KVNamespace, d1: D1Database, targetDir: string) -> void`, `flush(kv: KVNamespace, d1: D1Database, sourceDir: string) -> void`
- **Hides:** KV batch operations, D1 read/write, dirty file detection, path mapping between KV keys and filesystem paths
- **Tested through:** Integration test with Cloudflare's local dev tooling (miniflare)

### worker.ts
- **Interface:** `fetch()`, `email()`, `scheduled()` -- standard Cloudflare Worker handlers
- **Hides:** Email Routing parsing, cron schedule routing, Container/DO coordination
- **Tested through:** Integration test with wrangler dev

## File Changes

| File | Change | Type |
|------|--------|------|
| `src/worker.ts` | Main Worker with email, cron, fetch handlers | New |
| `src/container-keeper.ts` | Container Keeper Durable Object (keepalive + routing) | New |
| `src/state-sync.ts` | KV/D1 <-> filesystem sync | New |
| `wrangler.toml` | Cloudflare config (D1, KV, DO, Container, cron, email) | New |
| `Dockerfile` | Hermes runtime container image | New |
| `container-init.sh` | Startup script for Container | New |
| `hermes/cli-config.yaml` | Hermes config (model, toolsets, cron) | New |
| `hermes/SOUL.md` | Chief of staff personality | New |
| `hermes/skills/email-triage/SKILL.md` | Email triage skill instructions | New |
| `hermes/skills/email-triage/scripts/gmail_client.py` | Gmail API client | New |
| `hermes/skills/email-triage/scripts/outlook_client.py` | Microsoft Graph API client | New |
| `hermes/skills/email-triage/scripts/email_types.py` | Shared types | New |
| `hermes/skills/email-triage/scripts/worker_proxy.py` | Worker proxy API client (KV/D1 from Container) | New |
| `hermes/skills/daily-brief/SKILL.md` | Daily brief skill instructions | New |
| `hermes/skills/daily-brief/scripts/brief_builder.py` | Brief data aggregation | New |
| `state/priority-map.md` | Priority weighting document | New |
| `schema/d1-schema.sql` | D1 table definitions | New |
| `pyproject.toml` | Python deps (google-auth, msal, httpx) | New |
| `tests/test_gmail_client.py` | Gmail integration tests | New |
| `tests/test_outlook_client.py` | Outlook integration tests | New |
| `tests/test_brief_builder.py` | Brief builder tests | New |
| `.dev.vars` | Local dev secrets (API keys, tokens) | New |
| `.gitignore` | Ignore .dev.vars, __pycache__, .hermes state | New |

## Full System Phases (for context -- only Phase 1 is spec'd above)

### Phase 2: Tasks + Calendar + Meetings
- Notion task database (schema: status, priority, due date, source, context)
- `notion-tasks` skill: CRUD against Notion API
- Google Calendar integration: read events, detect conflicts
- Meeting prep briefs: 60 min before meeting, context delivered to Discord
- Post-meeting extraction: action items from notes -> Notion tasks

### Phase 3: Relationship Intelligence
- Per-person context files in KV (history, touchpoints, commitments)
- Commitment tracking: what user owes, what's owed, aging alerts
- Relationship context enriches meeting prep briefs

### Phase 4: Proactive Operations
- Task sweep: daily priority promotion, overdue flagging, rolled-5-days patterns
- Email follow-up tracking: flag emails without replies after N days
- Draft email composition in user's voice, queued for approval
- Approval gates on all outbound actions via Discord reactions
- Travel/receipt extraction from booking confirmations

### Phase 5: Wiki Bridge
- Read-only access to ~/Documents/wiki/ (concepts, sources)
- Surface relevant research in meeting briefs and queries
- Does not duplicate wiki curation pipeline

### Phase 6: Kaizen Self-Improvement
- Weekly cron: scan community, surface improvement ideas
- Subconscious loop: ideation -> debate -> synthesis -> persist
- Noise calibration: propose filter changes based on override patterns
- Skill auto-generation from completed tasks

### Cross-Cutting (added incrementally)
- Priority map for triage weighting (Phase 1)
- Approval gates on outbound actions (Phase 4)
- Do-not-disturb / quiet hours (Phase 4)
- Fail loud on all errors (Phase 1)

## Open Questions

- Q: Does a DO alarm every 30s reliably prevent Container sleep on Cloudflare?
  Default: Implement and test. If unreliable, add a Worker Cron trigger (every 1 min) as backup keepalive. Both approaches are cheap within the paid plan's included tiers.

- Q: DO WebSocket Hibernation does not apply to outbound WebSockets (DO as client). Does this affect cost?
  Default: No. Hermes runs its native Discord gateway inside the Container (outbound WebSocket). The Container stays alive via DO alarm. At ~128MB, 24/7 operation = ~324,000 GB-s/month, within the 400,000 GB-s/month included on the paid plan.

- Q: Will Outlook forwarding rules execute before or after junk filtering?
  Default: Assume junk is filtered first (conservative). Graph API polling every 15 min catches junk folder emails.
