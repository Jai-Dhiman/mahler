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

### Phase 2: Email Triage Skills
- Gmail + Outlook fetch skills (OAuth2 / MSAL auth)
- Junk folder rescue for Outlook
- Classification engine (URGENT / NEEDS_ACTION / FYI / NOISE) using priority map
- Store triage results in Cloudflare D1

### Phase 3: Morning Brief + Evening Wrap
- Cron-triggered briefs delivered to #mahler channel via Hermes cron system
- Brief builder aggregates triage data from D1
- Opinionated summaries, not data dumps

### Phase 4: Calendar + Task Integration
- Google Calendar integration
- ~~Notion task sync~~ — shipped: `notion-tasks` skill (create, list, update, complete, delete via Discord)
- Meeting prep and post-meeting extraction

### Phase 5: Relationship CRM
- Per-person context tracking
- Communication pattern detection

### Phase 6: Approval Gates
- Email draft composition with Discord-based approval flow

### Phase 7: Wiki Bridge — shipped as `notion-wiki` skill
- ~~Sync local wiki content to KV for agent access~~ — shipped differently: the `notion-wiki` skill gives Hermes read-only access to a Notion-backed wiki. Local Claude Code sessions ingest sources into Notion via `mahler/wiki/scripts/ingest.py`; Hermes reads via `wiki.py` (search/read/index). Raw sources stay on the laptop.

### Phase 8: Kaizen Loop
- Self-improvement: track what worked, what was ignored, refine triage rules
