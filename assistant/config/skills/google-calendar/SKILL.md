---
name: google-calendar
description: List upcoming Google Calendar events and create new events. Use for any request involving the user's schedule, meetings, or calendar.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [calendar, google, scheduling, meetings, productivity]
    related_skills: [meeting-prep, notion-tasks]
---

## When to use

- When the user asks "what do I have today/tomorrow/this week?", "what's on my calendar?", or "am I free on Friday?"
- When the user asks to schedule, book, or create a meeting or event
- When the meeting-prep skill needs to fetch upcoming events

## Date handling

All dates passed to the CLI must be in ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`. Before invoking any command, convert relative dates from the user's message to absolute ISO 8601 using today's date and the user's timezone (Pacific). Example: "Friday at 3pm" → `2026-04-17T22:00:00Z` (UTC).

## Operations

### List upcoming events

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py list --days N
```

`--days N` shows events from now to N days ahead (default: 7). Use `--hours-ahead N` to restrict to the next N hours.

Output format per event:
```
2026-04-16T15:00:00Z  Team standup
  Attendees: alice@x.com, bob@y.com
  Daily sync
```

### Create an event

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py create \
  --title "TITLE" \
  --start "YYYY-MM-DDTHH:MM:SSZ" \
  --end "YYYY-MM-DDTHH:MM:SSZ" \
  [--attendees "email1@x.com,email2@y.com"] \
  [--description "TEXT"]
```

Output on success: `Created: <event_id> — <title>`

## Output

Any failure raises `RuntimeError` and exits non-zero. Surface the error message to the user directly.
