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
  - Tables: `email_triage_log`, `triage_state`, `meeting_prep_log`
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

Organized in three layers (Knowledge → Execution → Architectural). Ordering inside Execution is by ROI, not chronology.

### Knowledge Layer

**Phase K1: notion-wiki — shipped.** Three Notion DBs (Sources, Concepts, Log), local ingest + Fly-side read-only access. Gives Mahler a persistent, queryable knowledge base that compounds across sessions. Local sessions ingest via `mahler/wiki/scripts/ingest.py`; Hermes reads via the `notion-wiki` skill (`search`/`read`/`index`). Raw sources stay on the laptop. Supersedes the retired Phase 7 "Wiki Bridge".

### Execution Layer

**Phase E1: Trader-analyst morning brief.** Extend the existing `morning-brief` skill to pull traderjoe state (open positions, yesterday's fills, credit-spread deltas, rationale) from shared D1 and deliver an opinionated read in `#mahler`. Highest-leverage next move given the traderjoe/assistant crossover — the daily brief becomes the single place positions, P&L, and priorities land.

**Phase E2: Email Triage depth.**
- Gmail + Outlook fetch skills (OAuth2 / MSAL auth)
- Junk folder rescue for Outlook
- Classification engine (URGENT / NEEDS_ACTION / FYI / NOISE) using priority map
- Store triage results in Cloudflare D1

**Phase E3: Kaizen Loop (pulled forward from old Phase 8).** Every triage/brief writes an outcome row to D1: what was surfaced, what the human actioned, what was ignored. Weekly reflection skill reads the log and proposes `SOUL.md` / priority-map edits for human approval. This is the compounding layer — pulled forward so every later phase contributes training data from day one instead of being retrofitted later.

- **v0.9 rescope:** use the pluggable context engine slot (`hermes plugins`) to inject the priority-map + last N outcome rows on every turn, instead of stuffing them into `SOUL.md`. Cleaner, keeps SOUL.md focused on personality, and the context engine is the designed seam for exactly this.

**Phase E4: Calendar + meeting flow — shipped.** Google Calendar skill (`gcal_client.py`, `gcal.py`) for listing and creating calendar events via Discord. Meeting prep skill (`d1_client.py`, `dedup.py`, `email_context.py`) delivers an intelligent prep brief ~1 hour before meetings, deduped via D1 `meeting_prep_log`. Calendar-aware plugin (`plugin.py`) injects upcoming meeting context on every LLM turn via the `pre_llm_call` hook.

**Phase E5: Relationship CRM.**
- Per-person context tracking
- Communication pattern detection feeding the morning brief

**Phase E6: Approval Gates.** Email draft composition with Discord-based approval flow. Depends on E2.

- **v0.8 unblock:** approval buttons now render as native Discord slash commands (`/approve`, `/deny`) with inline buttons + per-turn authorization. The ergonomic blocker that deferred this phase is gone; the only remaining work is draft generation + plumbing the approval hook into the send step.

### Architectural (deferred, needs design spike)

**Phase A1: Multi-agent profiles.** Hermes v0.6.0 introduced subagent profiles; Mahler runs as a single monolithic agent today. Natural split: `research` (wiki + web), `ops` (email / calendar / tasks), `trader` (traderjoe state, rationale, risk). Each subagent reads the shared notion-wiki.

- **v0.9 note:** profile handling saw significant work — "profile paths fixed in Docker — profiles go to mounted volume" (#7170), per-profile subprocess HOME isolation (#7357), profile-scoped memory isolation (v0.8). The Fly container blocker is probably resolvable now: mount a Fly volume at the profile directory and the state survives redeploys. Still worth a short spike, not a full design pass.
- **Not a prerequisite for E1–E6.** Pursue only when a concrete boundary problem (context bleed, conflicting personas, skill scoping) makes it worth the architectural cost.

### Retired

- **Phase 7: Wiki Bridge** — replaced by notion-wiki (K1). Local filesystem → KV sync approach was one-way with no query surface.
- **Phase 8: Kaizen Loop** — pulled forward to E3.
