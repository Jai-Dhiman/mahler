---
name: morning-brief
description: Post a morning or evening email brief to Discord. Queries the last 12 hours of triage results from D1 and formats a structured summary by classification bucket.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [email, brief, discord, notifications, productivity]
    related_skills: [email-triage, urgent-alert]
---

## When to use

- Cron-triggered at 8am Pacific (morning brief) and 8pm Pacific (evening brief)
- When the user asks any of: "morning brief", "email summary", "what did I miss", "evening brief", "what came in today"
- When the user wants a structured overview of recent triage results without listing every URGENT alert (those are already sent in real-time)

## Prerequisites

The following environment variables must be set as Fly.io secrets before this skill will function. The script raises a `RuntimeError` for any missing variable.

| Variable | Purpose |
|---|---|
| `CF_ACCOUNT_ID` | Cloudflare account ID for D1 API calls |
| `CF_D1_DATABASE_ID` | D1 database ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 read permission |
| `DISCORD_TRIAGE_WEBHOOK` | Webhook URL for brief embeds (same as urgent-alert) |

## Procedure

Post the morning brief (last 12 hours):

```bash
python3 ~/.hermes/skills/morning-brief/scripts/brief.py --period morning
```

Post the evening brief (last 12 hours):

```bash
python3 ~/.hermes/skills/morning-brief/scripts/brief.py --period evening
```

Dry run — print the embed payload to stdout without posting to Discord:

```bash
python3 ~/.hermes/skills/morning-brief/scripts/brief.py --period morning --dry-run
```

Custom lookback window (e.g. last 6 hours):

```bash
python3 ~/.hermes/skills/morning-brief/scripts/brief.py --period morning --since-hours 6
```

Flags may be combined:

```bash
python3 ~/.hermes/skills/morning-brief/scripts/brief.py --period evening --since-hours 6 --dry-run
```

## Output

The script posts a Discord embed with:
- Title: "Morning Brief — Apr 7" or "Evening Brief — Apr 7"
- Needs Action: each email on its own line with sender, subject, and one-sentence summary
- FYI: subject lines, one per line
- Noise: count of filtered emails
- If nothing needs attention: description says so and NEEDS_ACTION/FYI fields are omitted

URGENT emails are excluded (they were already alerted in real-time via `urgent-alert`).

Prints `"Brief posted."` on success. Raises `RuntimeError` on any failure — no silent degradation.
