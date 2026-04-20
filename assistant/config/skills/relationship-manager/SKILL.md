---
name: relationship-manager
description: >
  Track professional and personal relationships. Add contacts, update context,
  view summaries with open Notion tasks, log interactions, and sync Google Calendar
  for auto-detected last contact dates.
triggers:
  - add contact
  - update contact
  - summarize contact
  - list contacts
  - talked to
  - delete contact
  - sync calendar contacts
env:
  - CF_ACCOUNT_ID
  - CF_D1_DATABASE_ID
  - CF_API_TOKEN
  - NOTION_API_TOKEN
  - NOTION_DATABASE_ID
  - GMAIL_CLIENT_ID
  - GMAIL_CLIENT_SECRET
  - GMAIL_REFRESH_TOKEN
---

# Relationship Manager

Track professional and personal contacts in D1. Open commitments are Notion tasks
with `[Name]` prefix (e.g., `[Alice Chen] Send IC memo`). Last contact dates
auto-update daily from Google Calendar attendees at 08:00 UTC.

## Commands

All commands run:
```
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py <subcommand> [args]
```

### Add a contact

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py add \
  --name "Alice Chen" \
  --email "alice@example.com" \
  --type professional \
  --context "Partner at Sequoia, intro'd by Marcus"
```

Output: `Added: Alice Chen (professional)`

`--type` must be `professional` or `personal`. `--context` is optional.

### Summarize a contact

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py summarize \
  --name "Alice Chen"
```

Output: contact card (name, type, email, last contact date, context) followed by
open Notion tasks whose title starts with `[Alice Chen]`.

### List all contacts

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py list
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py list --type professional
```

### Log an interaction (manual touch)

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py talked-to \
  --name "Alice Chen"
```

Sets `last_contact` to today's date. Use for phone calls, in-person meetings, or
any interaction not captured by calendar sync.

Output: `Noted: talked to Alice Chen on 2026-04-19`

### Update a contact field

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py update \
  --name "Alice Chen" \
  --field context \
  --value "Now General Partner at Sequoia"
```

`--field` must be one of: `name`, `email`, `type`, `context`.

Output: `Updated: Alice Chen`

### Delete a contact

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py delete \
  --name "Alice Chen"
```

Output: `Deleted: Alice Chen`

### Sync calendar (runs automatically at 08:00 UTC)

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py sync-calendar --days 1
```

Fetches Google Calendar events from the past N days. For each event attendee whose
email matches a known contact, updates `last_contact` to the event date.

Output: `Synced calendar: updated last_contact for 2 contacts (Alice Chen, Bob Smith)`

## Notion task convention

Open commitments to a contact are tracked as Notion tasks with `[Name]` prefix:

```
[Alice Chen] Send Q2 IC memo
[Alice Chen] Intro to Marcus at Benchmark
```

Create these with the `notion-tasks` skill. The `summarize` command surfaces them automatically.

## Notes

- Name matching is case-insensitive for all commands.
- Email must be unique per contact — `add` on a duplicate email updates the existing record.
- The D1 `contacts` table is in `mahler-db` (same database as `email_triage_log`).
