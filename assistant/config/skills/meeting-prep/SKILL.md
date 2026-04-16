---
name: meeting-prep
description: Check for upcoming meetings in the next hour and deliver an intelligent prep brief to Discord. Run every 15 minutes via cron. Uses google-calendar, notion-tasks, notion-wiki, and email context to synthesize a brief.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [calendar, meetings, briefing, productivity, prep]
    related_skills: [google-calendar, notion-tasks, notion-wiki, email-triage]
---

## When to use

- Invoked automatically by Hermes cron every 15 minutes
- When the user asks "prepare me for my next meeting" or "what do I need to know before my call?"

## Prerequisites

| Variable | Purpose |
|---|---|
| `GMAIL_CLIENT_ID` | OAuth2 client ID |
| `GMAIL_CLIENT_SECRET` | OAuth2 client secret |
| `GMAIL_REFRESH_TOKEN` | OAuth2 refresh token with calendar.events scope |
| `CF_ACCOUNT_ID` | Cloudflare account ID |
| `CF_D1_DATABASE_ID` | D1 database ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 read/write |
| `DISCORD_TRIAGE_WEBHOOK` | Discord webhook for brief delivery |
| `NOTION_API_TOKEN` | Notion token for task lookup |
| `NOTION_DATABASE_ID` | Notion tasks database ID |

## Procedure

### Step 1 — Fetch upcoming events

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py list --hours-ahead 2
```

Parse the output. Filter for events whose start time (ISO 8601) is between 45 and 75 minutes from the current UTC time. Ignore all-day events (start time contains no `T` separator). If no events are in the 45–75 minute window, stop — nothing to do.

### Step 2 — Check deduplication

For the matching event, extract its event_id from the output line. Then:

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/dedup.py check --event-id EVENT_ID
```

- Exit 0: not yet briefed — continue to Step 3.
- Exit 1: already briefed — stop.
- Non-zero with RuntimeError: D1 failure — surface the error and stop.

### Step 3 — Gather context

Run these in parallel where possible:

**a) Recent emails from attendees** (skip if no attendees listed):

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/email_context.py email-context \
  --attendees "email1@x.com,email2@x.com"
```

**b) Open Notion tasks** (tasks due on or before the meeting date):

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  --status "Not started" --due-before YYYY-MM-DD
```

**c) Wiki lookup** (skip if meeting title is a name only, "1:1", "sync", "standup", or "catch up" with no description):

Extract 1–2 key topic words from the meeting title and description. For each:

```bash
python3 ~/.hermes/skills/notion-wiki/scripts/wiki.py search --query "TOPIC"
```

Read the top hit with `wiki.py read --id PAGE_ID`.

### Step 4 — Synthesize and post brief

Using all gathered context, compose a Discord embed with these sections:
- **Meeting:** title, start time, attendees
- **Recent emails:** formatted output from email_context.py (omit if none)
- **Open tasks:** list of relevant tasks (omit if none)
- **Wiki context:** 1–2 sentences from the wiki hit (omit if none)
- **What to know:** 3–5 bullet point synthesis (always present)

Post to `DISCORD_TRIAGE_WEBHOOK` using the urgent-alert format (embed with title + fields). Do NOT use the email-triage webhook pattern — call the Discord webhook directly with a JSON embed body.

### Step 5 — Log completion

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/dedup.py log \
  --event-id EVENT_ID \
  --summary "MEETING_TITLE" \
  --start-time "ISO8601_START_TIME"
```

This step must succeed. If it raises RuntimeError, surface the error — do not silently skip the log write.

## Failure modes

- `gcal.py` raises RuntimeError → surface to Discord as an error message and stop
- `dedup.py check` exits 1 → stop silently (normal dedup path)
- `dedup.py check` raises RuntimeError → surface to Discord and stop
- Email context or wiki lookup fails → log the error inline in the brief but continue with what was gathered
- notion-tasks list fails → omit tasks section, continue
- `dedup.py log` raises RuntimeError → surface to Discord (brief was already sent; this is critical state)
