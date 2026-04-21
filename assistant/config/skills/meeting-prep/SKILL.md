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

## Procedure

### Step 1 — Fetch upcoming events

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py upcoming --min-minutes 45 --max-minutes 75 \
  --skip-keywords "orchestra,rehearsal,bohemian,jinks,encampment"
```

This command does all time filtering in Python and outputs only events whose start time is 45–75 minutes from now (UTC). All-day events and events matching any skip keyword (checked against title + description, case-insensitive) are excluded automatically. If output is `No meetings in window.`, stop — nothing to do.

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

Compose 3–5 bullet points summarising what to know for this meeting, then call:

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/post_brief.py \
  --title "MEETING_TITLE" \
  --start "ISO8601_UTC_START" \
  --synthesis "• Bullet 1\n• Bullet 2\n• Bullet 3" \
  [--attendees "email1, email2"] \
  [--emails "Recent email summary text"] \
  [--tasks "Task 1\nTask 2"] \
  [--wiki "Wiki context sentence"]
```

Omit optional flags when that context is empty. If the script raises RuntimeError, surface the error to Discord and stop — do NOT proceed to Step 5.

### Step 5 — Log completion

Only run this after Step 4 succeeds (prints "Brief sent."):

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/dedup.py log \
  --event-id EVENT_ID \
  --summary "MEETING_TITLE" \
  --start-time "ISO8601_START_TIME"
```

If it raises RuntimeError, surface the error — the brief was sent but the dedup record is missing.

## Failure modes

- `gcal.py` raises RuntimeError → surface to Discord as an error message and stop
- `dedup.py check` exits 1 → stop silently (normal dedup path)
- `dedup.py check` raises RuntimeError → surface to Discord and stop
- Email context or wiki lookup fails → log the error inline in the brief but continue with what was gathered
- notion-tasks list fails → omit tasks section, continue
- `dedup.py log` raises RuntimeError → surface to Discord (brief was already sent; this is critical state)
