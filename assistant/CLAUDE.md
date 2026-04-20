# Mahler - Personal Chief of Staff

An opinionated AI chief of staff built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by NousResearch. Deployed to Fly.io, connected to Discord.

## Architecture

```
Discord <--WebSocket--> Hermes Agent (Fly.io) --API--> OpenRouter (Grok 4.1 Fast)
                              |
                              +--skills--> ~/.hermes/skills/ (custom skills)
                              +--state---> Cloudflare D1 + KV (persistent storage)
```

**Runtime:** Hermes Agent v0.9.0 (tag `v2026.4.13`) running in Docker on Fly.io (shared-cpu-1x, 512MB, SJC region). Pinned via `HERMES_VERSION` build arg in the Dockerfile.

**LLM:** `openai/gpt-5-nano` via OpenRouter.

**Discord:** Gateway bot (`Mahler#9543`) in the user's personal server. Responds to @mentions in-channel (`auto_thread: false`). Allowed user: `223665447684407298`.

**Hermes is not forked.** Installed via the official install script inside Docker, pinned to a release tag. Upgrading means bumping `HERMES_VERSION` in the Dockerfile and rebuilding.

## Project Structure

```
assistant/
  Dockerfile              # Installs Hermes via official script, copies config
  entrypoint.sh           # Writes env vars to ~/.hermes/.env, starts gateway
  fly.toml                # Fly.io deployment config
  hooks/
    project_log.py        # SessionStop hook: keyword-scans transcripts, writes to D1 project_log
  config/
    SOUL.md               # Mahler personality
    config.yaml           # Model, provider, Discord settings
    honcho.json           # Honcho memory backend config
    priority-map.md       # Email classification rules
    auth/
      gmail_auth.py       # Gmail OAuth2
      outlook_auth.py     # Outlook OAuth2
    plugins/
      calendar-aware/     # Injects upcoming meetings context
      conversation-history/ # Injects last 45min of Discord history on first turn
      kaizen-context/     # Injects current priority map on every turn
      project-context/    # Injects last 3 days of project wins/blockers on first turn
    skills/
      email-triage/       # Gmail fetch → classify → D1; Outlook reply attribution
      evening-sweep/      # 6pm Pacific cron: reviews today's tasks, stages tomorrow top 3
      google-calendar/    # Google Calendar read access
      kaizen-reflection/  # Weekly: proposes priority map reclassifications
      meeting-prep/       # Pre-meeting brief ~1hr before meetings
      morning-brief/      # Daily brief with news feed (news_sources.json)
      notion-tasks/       # Notion task CRUD via Discord
      notion-wiki/        # Search/read personal knowledge wiki
      reflection-journal/ # Sunday cron: reflection questions → D1 → Honcho
      relationship-manager/ # Contact CRM with Google Calendar auto-sync
      urgent-alert/       # Fires on URGENT-classified emails
```

## Key Conventions

- **Hermes reads secrets from `~/.hermes/.env`**, not process environment. `entrypoint.sh` bridges this at startup.
- **Config key for model is `default`**: `model.default: "openai/gpt-5-nano"`.
- **Custom skills** go in `config/skills/`, COPYed to `~/.hermes/skills/` in Dockerfile. Follow Hermes skill format: `SKILL.md` with YAML frontmatter + markdown body, optional `scripts/` directory.
- **Discord settings** (token, home_channel, allowed_users) are set via env vars, not config.yaml.
- **Python package management:** use `uv`, not pip.
- **SessionStop hook** registered in `~/.claude/settings.json` pointing to `assistant/hooks/project_log.py` (blocker mode). Requires `~/.mahler.env` with CF + OpenRouter creds.

## Cloudflare Resources

- **D1 Database:** `mahler-db` (ID: `b6cb2eac-2903-46bd-baea-b4ff2dc904d0`)
  - Tables: `email_triage_log`, `triage_state`, `meeting_prep_log`, `priority_map`, `project_log`, `reflection_log`, `contacts`
- **KV Namespace:** `KV` (ID: `0a93ac9040324708a8b9f00eed8715e9`)

## Deployment

```bash
# Deploy (run from assistant/ directory)
flyctl deploy --remote-only

# Set secrets
flyctl secrets set DISCORD_BOT_TOKEN=xxx OPENROUTER_API_KEY=xxx ...

# Logs (must ssh as hermes user — root points at empty /root/.hermes)
flyctl ssh console --user hermes -C "hermes logs -n 200"
flyctl ssh console --user hermes -C "hermes logs errors"
flyctl ssh console --user hermes -C "hermes logs --component gateway --since 1h"
flyctl ssh console --user hermes -C "hermes logs -f"

# Status
flyctl status
```

## Roadmap

**Shipped:** K1 (notion-wiki), E2 (email-triage + reply-attribution), E2a (Honcho memory), E3 (kaizen-reflection + kaizen-context), E3+ (reflection-journal + project analysis), E4 (google-calendar + meeting-prep + calendar-aware), E4.1 (conversation-history), E4b (project-context + SessionStop hook), E5 (relationship-manager), E7 (fathom-webhook CF Worker + meeting-followthrough skill), E8 (evening-sweep), E10 morning-brief news extension.
