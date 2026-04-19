# Mahler - Personal Chief of Staff

An opinionated AI chief of staff built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by NousResearch. Deployed to Fly.io, connected to Discord.

## Architecture

```
Discord <--WebSocket--> Hermes Agent (Fly.io) --API--> OpenRouter (Grok 4.1 Fast)
                              |
                              +--skills--> ~/.hermes/skills/ (custom skills)
                              +--state---> Cloudflare D1 + KV (persistent storage)
```

**Runtime:** Hermes Agent v0.9.0 (tag `v2026.4.13`) running in Docker on Fly.io (shared-cpu-1x, 512MB, SJC region). Pinned via `HERMES_VERSION` build arg in the Dockerfile — bump the tag and rebuild to upgrade.

**LLM:** `x-ai/grok-4.1-fast` via OpenRouter. (v0.9.0 added a native xAI provider; see "Upgrade levers" below.)

**Discord:** Gateway bot (`Mahler#9543`) in the user's personal server. Responds to @mentions in-channel (`auto_thread: false`). Allowed user: `223665447684407298`.

**Hermes is not forked.** Installed via the official install script inside Docker, pinned to a release tag. Custom config (SOUL.md, config.yaml) is copied on top. Upgrading means bumping `HERMES_VERSION` in the Dockerfile and rebuilding — in-container `hermes update` is not used because the install is immutable.

## Project Structure

```
assistant/
  Dockerfile              # Installs Hermes via official script, copies config
  entrypoint.sh           # Writes env vars to ~/.hermes/.env, starts gateway
  fly.toml                # Fly.io deployment config
  config/
    SOUL.md               # Mahler personality (opinionated chief of staff)
    config.yaml           # Model, provider, Discord settings
    skills/
      notion-tasks/       # Notion task management skill (CRUD via Discord)
        SKILL.md
        scripts/
          notion_client.py  # Notion API client (pages, databases, filters)
          tasks.py          # CLI: create, list, update, complete, delete
        tests/
      evening-sweep/      # Evening task sweep skill (cron at 01:00 UTC / 6pm Pacific)
        SKILL.md
        scripts/
          notion_client.py  # Notion API client (completed, past-due, open task queries)
          sweep.py          # CLI: run sweep, post summary to Discord, pick top 3 for tomorrow
        tests/
  .env                    # Local secrets (gitignored)
  .env.example            # Template for required secrets
  docs/
    archive/              # Previous spec/plan (abandoned Container/DO approach)
```

## Key Conventions

- **Hermes reads secrets from `~/.hermes/.env`**, not process environment variables. The `entrypoint.sh` bridges this by writing env vars to that file at startup.
- **Config key for model is `default`**, not `name`: `model.default: "x-ai/grok-4.1-fast"`.
- **Custom skills** go in the `config/skills/` directory and are COPYed to `~/.hermes/skills/` in the Dockerfile. Follow Hermes skill format: `SKILL.md` with YAML frontmatter + markdown body, optional `scripts/` directory.
- **Discord settings** (token, home_channel, allowed_users) are set via env vars, not config.yaml. Hermes's `_apply_env_overrides()` handles them.
- **Python package management:** use `uv`, not pip.
- **JS package management:** use `bun`, not npm.
- **SessionStop hook** registered in `~/.claude/settings.json` pointing to `assistant/hooks/project_log.py` (blocker mode).

## Deployment

```bash
# Deploy to Fly.io (run from assistant/ directory)
flyctl deploy --remote-only

# Set secrets
flyctl secrets set DISCORD_BOT_TOKEN=xxx OPENROUTER_API_KEY=xxx ...

# Check logs (v0.8+ structured logs: agent.log / errors.log / gateway.log)
# MUST ssh as the hermes user — `hermes logs` resolves ~/.hermes via $HOME and the
# default ssh console drops you in as root, which points at the empty /root/.hermes.
flyctl ssh console --user hermes -C "hermes logs -n 200"                         # last 200 lines of agent.log
flyctl ssh console --user hermes -C "hermes logs errors"                         # errors only
flyctl ssh console --user hermes -C "hermes logs --component gateway --since 1h" # gateway-only, last hour
flyctl ssh console --user hermes -C "hermes logs --level WARNING"                # WARN+ only
flyctl ssh console --user hermes -C "hermes logs -f"                             # follow in real time
# Raw fallback (works as any user since the path is absolute):
flyctl ssh console -C "tail -n 200 /home/hermes/.hermes/logs/agent.log"

# Check status
flyctl status
```

## Cloudflare Resources (Provisioned)

Available for custom skills:

- **D1 Database:** `mahler-db` (ID: `b6cb2eac-2903-46bd-baea-b4ff2dc904d0`)
  - Tables: `email_triage_log`, `triage_state`, `meeting_prep_log`, `priority_map`
  - Shared with traderjoe (assistant tables are prefixed by purpose, not name)
- **KV Namespace:** `KV` (ID: `0a93ac9040324708a8b9f00eed8715e9`)

Salvageable code from previous build in `hermes-assistant` repo's `.worktrees/feat/hermes-chief-of-staff`: `gmail_client.py`, `outlook_client.py`, `email_types.py`, `brief_builder.py`.

## Upgrade levers unlocked by v0.8.0 → v0.9.0

These are capabilities now available in the runtime that Mahler does not yet use. Ordered by leverage, not effort. None require a fork — they either live in config, skills, or light Dockerfile changes.

**Reliability & ops (take these first — low effort, immediate payoff):**

- **Inactivity-based agent timeouts** (v0.8) — gateway timeouts now track tool activity, not wall clock. Long-running email-triage / morning-brief runs won't get reaped mid-work. No config needed; already active.
- **`notify_on_complete` on background processes** (v0.8) — long background tasks notify the agent on completion instead of requiring polling. Use when a skill kicks off a deploy, backtest, or bulk fetch.
- **`watch_patterns`** (v0.9) — regex-match against a background process's live output and get notified on match. Useful for traderjoe backtest progress, "listening on port" readiness, or error detection in long runs.
- **Structured logging + `hermes logs`** (v0.8) — `~/.hermes/logs/agent.log` + `errors.log`. Replaces the old `cat gateway.log` workflow.
- **Config structure validation at startup** (v0.8) — catches malformed `config.yaml` before it silently breaks a deploy. Already active.
- **`hermes backup` / `hermes import`** (v0.9) — snapshot config, sessions, skills, memory. Run before risky skill edits; store in a Fly volume or pull locally.
- **`/debug` + `hermes debug share`** (v0.9) — one-shot diagnostic report. Faster than `flyctl ssh` spelunking when something breaks.

**Model & provider:**

- **Live `/model` switching** (v0.8) — swap models mid-session from Discord. Useful for temporarily routing a hard question to a stronger model without redeploying. Aggregator-aware (stays on OpenRouter when possible).
- **Native xAI (Grok) provider** (v0.9) — direct API access to `x-ai/grok-4.1-fast` with `x-grok-conv-id` prompt caching. Would require an xAI API key instead of (or alongside) OpenRouter. Worth benchmarking for latency and cost before switching; the OpenRouter route is fine for now.
- **Fast Mode `/fast`** (v0.9) — priority tier for OpenAI/Anthropic. Not applicable while Mahler is on Grok, but matters the moment a phase needs a stronger model (e.g., E6 approval drafting).

**Discord gateway:**

- **Approval buttons as native slash commands** (v0.8) — `/approve`, `/deny`, `/queue`, `/background`, `/btw` render as Discord slash commands with inline buttons instead of requiring typed commands. This is the missing ergonomic piece for Phase E6 (Email Approval Gates).
- **`ignored_channels` + `no_thread_channels`** (v0.8) — finer-grained than the current single `home_channel` setting. Enables isolating `#mahler` for briefs while allowing ad-hoc chat elsewhere.
- **`DISCORD_REPLY_TO_MODE`** (v0.9) — controls whether Mahler replies-to vs. sends in-channel. Quality-of-life.
- **Inbound text batching + adaptive delay** (v0.9) — groups rapid sequential messages into one turn instead of processing each. Saves tokens on multi-line asks.

**Skills & memory:**

- **Pluggable context engine slot** via `hermes plugins` (v0.9) — custom per-turn context injection/filtering. The right seam for **Phase E3 (Kaizen Loop)**: inject the current priority-map + recent outcome log on every turn instead of stuffing everything into `SOUL.md`. Rescopes that phase.
- **Thread user_id to memory plugins** (v0.8) — per-user memory scoping; relevant once more than one person talks to Mahler, not today.
- **Shared thread sessions by default** (v0.8) — multi-user threads work without extra config.
- **Supermemory / Hindsight / Honcho plugin overhauls** (v0.8/v0.9) — a real decision point when Kaizen Loop kicks off: pick a memory plugin or keep using D1 as the source of truth.

**Security (take these as part of the upgrade itself):**

- Path traversal fixes in checkpoint manager, shell-injection neutralization in sandbox writes, SSRF redirect guards, Twilio webhook signature validation (SMS RCE fix), API server auth enforcement, git argument injection prevention, approval button authorization (v0.9). Mostly invisible — just upgrading gets them.

**Deferred (know they exist, don't act yet):**

- **iMessage via BlueBubbles** (v0.9) — requires a persistent Mac running BlueBubbles (laptop-as-bridge or Mac mini / Mac-in-cloud). Not pursuing until the rest of the Execution layer is denser.
- **Local Web Dashboard** (v0.9) — browser-based gateway management. Useful for local dev, not the Fly deploy.
- **Multi-arch Docker image (amd64 + arm64)** (v0.9) — relevant only if Mahler moves off `shared-cpu-1x`.
- **16-platform gateway support** — WeChat, WeCom, Feishu, DingTalk, Matrix, Signal, SMS, Home Assistant. None are currently wanted.

## Roadmap

Phases are organized by execution wave — what can run in parallel vs. what is blocked. See "Execution Order" below.

### Knowledge Layer

**Phase K1: notion-wiki — shipped.** Three Notion DBs (Sources, Concepts, Log), local ingest + Fly-side read-only access. Gives Mahler a persistent, queryable knowledge base that compounds across sessions. Local sessions ingest via `mahler/wiki/scripts/ingest.py`; Hermes reads via the `notion-wiki` skill (`search`/`read`/`index`). Raw sources stay on the laptop. Supersedes the retired Phase 7 "Wiki Bridge".

### Execution Layer

**Phase E2: Email Triage.** Gmail OAuth2 fetch → classify → D1. This is the primary prerequisite for morning-brief having real content, urgent-alert firing on real data, and kaizen-reflection having email patterns to learn from. Gmail only (no Outlook planned). Classification uses the priority map already in D1 from E3. Activates: `morning-brief`, `urgent-alert`, `kaizen-reflection`, and Tier 1 finance parsing.
- Google Workspace OAuth2 skill (reuses Calendar OAuth already deployed in E4)
- URGENT / NEEDS_ACTION / FYI / NOISE classification against `priority_map` in D1
- Store results in `email_triage_log`

**Phase E2a: Honcho Memory Backend — shipped.** Honcho wired as the persistent memory provider. Config at `config/honcho.json` (`workspace: mahler`, `recallMode: hybrid`, `dialecticCadence: 3`). `HONCHO_API_KEY` bridged via `entrypoint.sh`. `SOUL.md` rules instruct Mahler to call `honcho_conclude` for durable facts and `honcho_search` for context-enriched answers. All subsequent phases deposit signal into Honcho automatically.

**Phase E3: Kaizen Loop — shipped.** Priority map in D1, `kaizen-context` plugin injects it on every turn, `kaizen-reflection` skill runs weekly to propose reclassifications. See `config/plugins/kaizen-context/` and `config/skills/kaizen-reflection/`.

**Phase E3+: Expand Kaizen Scope.** Broaden the kaizen loop beyond email patterns to watch conversation patterns, project logs, and reflection journal entries. Also add the reflection journal: a weekly cron (Sunday evening) asks 2-3 structured questions (what went well, what drained you, what are you avoiding), stores answers in D1, feeds them into both Honcho and kaizen-reflection. Depends on: E2a (Honcho), E4b (project log data).
- Expand `reflect.py` to query `project_log` and `reflection_log` in addition to `email_triage_log`
- Add `reflection-journal` skill: cron + Discord command for manual entries
- Reflection answers feed `honcho_conclude` for durable preference facts

**Phase E4: Calendar + meeting flow — shipped.** Google Calendar skill, meeting prep brief ~1 hour before meetings, calendar-aware plugin. See `config/skills/google-calendar/` and `config/skills/meeting-prep/`.

**Phase E4.1: Conversation history — shipped.** `conversation-history` plugin injects last 45 minutes of Discord channel history on the first turn of each new session.

**Phase E4b: Project Awareness — shipped.** Claude Code `SessionStop` hook (`assistant/hooks/project_log.py`) keyword-scans session transcripts, calls OpenRouter to synthesize a blocker summary, and writes win/blocker entries to D1 `project_log`. The `/ship` skill logs shipped wins automatically. The `project-context` Hermes plugin (`config/plugins/project-context/`) injects the last 7 days of project activity as context on every LLM turn.
- D1 table: `project_log (project, entry_type, summary, git_ref, created_at)`
- `assistant/hooks/project_log.py`: CLI — `win` mode (called by /ship), `blocker` mode (SessionStop hook)
- `config/plugins/project-context/plugin.py`: `pre_llm_call` plugin injects recent wins/blockers
- SessionStop hook registered in `~/.claude/settings.json`; requires `~/.mahler.env` with CF + OpenRouter creds

**Phase E5: Relationship CRM.** D1-backed contact tracking for professional and personal relationships. Tracks last contact date, open commitments, important context per person. Proactive follow-up detection sweeps sent Gmail threads for no-reply after N days (depends on E2). Honcho provides relationship memory layer on top. No formal pipeline UI — Discord commands for CRUD + summary.
- D1 table: `contacts (name, type, last_contact, context, open_commitments)`
- `relationship-manager` skill: add/update/list contacts, weekly follow-up sweep
- Follow-up detection: queries `email_triage_log` for sent threads with no reply (depends on E2)
- Depends on: E2a (Honcho for relationship memory), soft dependency on E2 for follow-up detection

**Phase E7: Meeting Follow-Through.** After a meeting, extract action items from notes and push them to Notion tasks + update the relevant contact in CRM. Closes the loop that meeting-prep opens. Depends on E5 (CRM for contact update).
- `meeting-followthrough` skill: Discord command to process notes → structured action items
- Pushes to `notion-tasks` skill
- Updates `contacts` table with meeting outcome and any new commitments

**Phase E8: Evening Task Sweep + Daily Rhythm — shipped.** 6pm Pacific cron reviews today's Notion tasks (completed, stalled, rolled-forward), flags patterns, stages tomorrow's top priorities, posts a short summary to Discord. Pairs with the morning brief to give the day a close. No hard dependencies — fully independent.
- Skill path: `config/skills/evening-sweep/` (cron at 01:00 UTC / 6pm Pacific via `entrypoint.sh`)
- Queries Notion for completed-today, past-due, and open tasks
- Posts structured summary to Discord; picks top 3 tasks for tomorrow; checks in on overdue items
- `scripts/notion_client.py`: Notion API client with date-filtered queries
- `scripts/sweep.py`: sweep runner, Discord post formatting, top-3 selection logic

**Phase E9: Finance Layer.**
- **Tier 1 (depends on E2):** Parse financial emails from Gmail — Wells Fargo transaction alerts, Wealthfront weekly/monthly summaries, Rocket Money reports. Classification patterns added to email triage. Monthly finance summary posted to Discord.
- **Tier 2 (independent):** Plaid read-only integration as a separate CF Worker cron. Pulls balances and transaction categories daily, writes summaries to D1. Mahler reads D1; never touches credentials. Wells Fargo and Wealthfront both supported via Plaid. Credentials stored only in Fly.io secrets.
- Security constraints: D1 stores summarized data only (spend by category, portfolio value, net worth trend), never raw transactions. Read-only scope everywhere.

**Phase E10: Life Tracking.** Low-friction personal tracking via Discord — health/fitness logs, reading tracker (compounds with notion-wiki), personal goals and habit patterns. Weekly reflection cron surfaces trends. Also adds curated news signal to morning brief: a configurable watchlist of domains (markets, AI, trading research) with a brief "things worth 10 minutes" section.
- `life-log` skill: Discord commands for health, reading, goal entries → D1 `life_log` table
- `morning-brief+` extension: adds curated news section (configurable source list)
- Weekly life-tracking review cron → feeds Honcho + reflection journal
- No dependencies; fully independent

### Execution Order

Dependencies determine three parallel waves:

**Wave 1 — No prerequisites, start immediately (run in parallel):**
- E2a: Honcho memory backend — shipped
- E4b: Project Awareness (SessionStop hook) — shipped
- E8: Evening Task Sweep — shipped

**Wave 2 — After Wave 1 is live (run in parallel):**
- E2: Gmail OAuth2 fetch + classify (E2a gives it Honcho to learn from immediately)
- E5: Relationship CRM (needs E2a for Honcho layer; follow-up detection added later when E2 ships)
- E10: Life Tracking (independent; benefits from E2a)

**Wave 3 — After E2 ships:**
- E3+: Expand Kaizen + Reflection Journal (needs email data + E2a)
- E7: Meeting Follow-Through (needs E5 CRM)
- E9 Tier 1: Finance email parsing (needs Gmail)

**Wave 4 — After Wave 3:**
- E9 Tier 2: Plaid integration (independent but low urgency until Tier 1 proves value)

### Architectural (deferred)

**Phase A1: Multi-agent profiles.** Not a prerequisite for any phase above. Revisit when a concrete boundary problem (context bleed, conflicting personas) appears. v0.9.0 fixed the Fly volume blocker for profiles — worth a spike when the monolithic agent shows strain.

### Retired

- **Phase E1: Trader-analyst morning brief** — traderjoe state already surfaces in the traderjoe Discord channel. Revisit if a unified brief becomes worth the plumbing.
- **Phase E6: Approval Gates** — removed. No email drafting planned.
- **Phase 7: Wiki Bridge** — replaced by K1 (notion-wiki).
- **Phase 8: Kaizen Loop** — shipped as E3.
