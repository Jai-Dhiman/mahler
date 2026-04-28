---
name: meeting-prep
description: Check for upcoming meetings in the next hour and deliver an intelligent prep brief to Discord. Run every 15 minutes via cron. Uses google-calendar, notion-tasks, notion-wiki, email context, and CRM to synthesize a brief.
version: 0.2.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [calendar, meetings, briefing, productivity, prep]
    related_skills: [google-calendar, notion-tasks, notion-wiki, email-triage, relationship-manager]
---

## When to use

- Invoked automatically by Hermes cron every 15 minutes
- When the user asks "prepare me for my next meeting" or "what do I need to know before my call?"

## Procedure

Run the orchestrator:

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/orchestrate.py
```

Report whatever it prints to stdout. If stdout is `NO_WORK`, stop silently. If exit code is non-zero, report the error to the user.

## Failure modes

- `gcal.py` fails → orchestrator raises, print error to user and stop
- Already briefed (dedup) → `NO_WORK`, stop silently
- Email context or wiki lookup fails → omitted from brief, continues
- CRM summarize fails for an attendee → that attendee skipped, continues
- `post_brief.py` fails → orchestrator raises, print error to user and stop
- `dedup.py log` fails → orchestrator raises, print error (brief was sent but dedup record is missing)
