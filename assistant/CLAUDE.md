# Mahler - Personal Chief of Staff

An opinionated AI chief of staff built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by NousResearch. Deployed to Fly.io, connected to Discord.

## Architecture

```
Discord <--WebSocket--> Hermes Agent (Fly.io) --API--> OpenRouter (Grok 4.1 Fast)
                              |
                              +--skills--> ~/.hermes/skills/ (custom skills)
                              +--state---> Cloudflare D1 + KV (persistent storage)
```

**Runtime:** Hermes Agent v0.7.0 running in Docker on Fly.io (shared-cpu-1x, 512MB, SJC region).

**LLM:** `x-ai/grok-4.1-fast` via OpenRouter.

**Discord:** Gateway bot (`Mahler#9543`) in the user's personal server. Responds to @mentions in-channel (`auto_thread: false`). Allowed user: `223665447684407298`.

**Hermes is not forked.** Installed via the official install script inside Docker. Custom config (SOUL.md, config.yaml) is copied on top. `hermes update` works inside the container by rebuilding the Docker image.

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

# Check logs
flyctl ssh console -C "cat /home/hermes/.hermes/logs/gateway.log"

# Check status
flyctl status
```

## Cloudflare Resources (Provisioned)

Available for custom skills:

- **D1 Database:** `mahler-db` (ID: `b6cb2eac-2903-46bd-baea-b4ff2dc904d0`)
  - Tables: `email_triage_log`, `triage_state`
  - Shared with traderjoe (assistant tables are prefixed by purpose, not name)
- **KV Namespace:** `KV` (ID: `0a93ac9040324708a8b9f00eed8715e9`)

Salvageable code from previous build in `hermes-assistant` repo's `.worktrees/feat/hermes-chief-of-staff`: `gmail_client.py`, `outlook_client.py`, `email_types.py`, `brief_builder.py`.

## Roadmap

Organized in three layers (Knowledge → Execution → Architectural). Ordering inside Execution is by ROI, not chronology.

### Knowledge Layer

**Phase K1: notion-wiki (in flight).** Three Notion DBs (Sources, Concepts, Log), local ingest + Fly-side read-only access. Gives Mahler a persistent, queryable knowledge base that compounds across sessions. See `docs/specs/2026-04-12-notion-wiki-design.md` and `docs/plans/2026-04-12-notion-wiki.md`. Supersedes the retired Phase 7 "Wiki Bridge".

### Execution Layer

**Phase E1: Trader-analyst morning brief.** Extend the existing `morning-brief` skill to pull traderjoe state (open positions, yesterday's fills, credit-spread deltas, rationale) from shared D1 and deliver an opinionated read in `#mahler`. Highest-leverage next move given the traderjoe/assistant crossover — the daily brief becomes the single place positions, P&L, and priorities land.

**Phase E2: Email Triage depth.**
- Gmail + Outlook fetch skills (OAuth2 / MSAL auth)
- Junk folder rescue for Outlook
- Classification engine (URGENT / NEEDS_ACTION / FYI / NOISE) using priority map
- Store triage results in Cloudflare D1

**Phase E3: Kaizen Loop (pulled forward from old Phase 8).** Every triage/brief writes an outcome row to D1: what was surfaced, what the human actioned, what was ignored. Weekly reflection skill reads the log and proposes `SOUL.md` / priority-map edits for human approval. This is the compounding layer — pulled forward so every later phase contributes training data from day one instead of being retrofitted later.

**Phase E4: Calendar + meeting flow.**
- Google Calendar integration
- Meeting prep brief and post-meeting extraction
- `notion-tasks` already shipped for the task side (create, list, update, complete, delete via Discord)

**Phase E5: Relationship CRM.**
- Per-person context tracking
- Communication pattern detection feeding the morning brief

**Phase E6: Approval Gates.** Email draft composition with Discord-based approval flow. Depends on E2.

### Architectural (deferred, needs design spike)

**Phase A1: Multi-agent profiles.** Hermes v0.6.0 introduced subagent profiles; Mahler runs as a single monolithic agent today. Natural split: `research` (wiki + web), `ops` (email / calendar / tasks), `trader` (traderjoe state, rationale, risk). Each subagent reads the shared notion-wiki.

- **Blocker:** verify subagent profiles work inside the Fly.io container runtime before committing. Profiles assume separate state and may expect a writable local install; the Hermes install on Fly is container-baked. Spike this before `/brainstorm`.
- **Not a prerequisite for E1–E6.** Pursue only when a concrete boundary problem (context bleed, conflicting personas, skill scoping) makes it worth the architectural cost.

### Retired

- **Phase 7: Wiki Bridge** — replaced by notion-wiki (K1). Local filesystem → KV sync approach was one-way with no query surface.
- **Phase 8: Kaizen Loop** — pulled forward to E3.
