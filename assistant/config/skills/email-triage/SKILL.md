---
name: email-triage
description: Fetch unread emails from Gmail and Outlook, classify as URGENT/NEEDS_ACTION/FYI/NOISE using the priority map, store results in Cloudflare D1, and send Discord alerts for URGENT emails.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [email, triage, gmail, outlook, productivity, automation]
    related_skills: [urgent-alert]
---

## When to use

- On a cron schedule (every 15 minutes, automatic)
- When the user asks any of: "check my email", "any urgent emails?", "what's in my inbox?", "triage my email", "do I have anything important?"
- When the user wants a summary of unread email across Gmail and Outlook

## Prerequisites

The following environment variables must be set as Fly.io secrets before this skill will function. The triage script raises a `RuntimeError` for any missing variable — do not silently skip or default to a degraded mode.

| Variable | Purpose |
|---|---|
| `GMAIL_CLIENT_ID` | OAuth2 client ID for Gmail API access |
| `GMAIL_CLIENT_SECRET` | OAuth2 client secret for Gmail API access |
| `GMAIL_REFRESH_TOKEN` | OAuth2 refresh token for Gmail API access |
| `OUTLOOK_EMAIL` | Outlook/Microsoft 365 email address |
| `OUTLOOK_APP_PASSWORD` | App password for IMAP access (host: `outlook.office365.com`) |
| `CF_ACCOUNT_ID` | Cloudflare account ID for D1 API calls |
| `CF_D1_DATABASE_ID` | D1 database ID (`fa2dbfa8-ee1c-4c7d-ba4c-6e304ee5bc21`) |
| `CF_API_TOKEN` | Cloudflare API token with D1 write permission |
| `DISCORD_TRIAGE_WEBHOOK` | Webhook URL for URGENT alert embeds |
| `OPENROUTER_API_KEY` | API key for the LLM classifier (already set for main model) |

## Procedure

Standard run (fetch, classify, store, alert on URGENT):

```bash
python3 ~/.hermes/skills/email-triage/scripts/triage.py
```

Dry run — fetch and classify without writing to D1 or sending Discord alerts:

```bash
python3 ~/.hermes/skills/email-triage/scripts/triage.py --dry-run
```

Recent emails only — restrict fetch window to the last N hours:

```bash
python3 ~/.hermes/skills/email-triage/scripts/triage.py --since-hours 1
```

Both flags may be combined:

```bash
python3 ~/.hermes/skills/email-triage/scripts/triage.py --dry-run --since-hours 1
```

## Output

The script prints a plain-text summary to stdout on completion:

- Total emails fetched, broken down by source (Gmail / Outlook)
- Count of emails classified at each tier: URGENT / NEEDS_ACTION / FYI / NOISE
- Subject lines of all URGENT emails
- Any errors encountered (fetch failure, classification failure, D1 write failure) with the affected email identified

Exit code is 0 on full success. Any unrecoverable error raises a `RuntimeError` and exits non-zero.

## Priority map

The LLM classifier loads `~/.hermes/workspace/priority-map.md` at runtime to determine how to classify each email. The priority map defines the four tiers with examples and tie-breaking rules. If the file is not found, the script raises a `RuntimeError` and exits non-zero — classification without the priority map is not permitted.

## Storage

Triage results are written to the `email_triage_log` table in the Cloudflare D1 database. Each row records: message ID, source, sender, subject, classification tier, a one-sentence summary, and the timestamp of triage. Duplicate message IDs (already triaged in a prior run) are skipped without error.

## Alerting

For each email classified as URGENT, the skill invokes the `urgent-alert` skill to post a formatted Discord embed to `DISCORD_TRIAGE_WEBHOOK`. The alert includes sender, subject, and a one-sentence summary. Failed alerts raise a `RuntimeError` — URGENT emails must not be silently dropped.
