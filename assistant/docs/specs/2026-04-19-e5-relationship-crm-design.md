# E5 Relationship CRM Design

**Goal:** Track professional and personal relationships via Discord commands, with `last_contact` auto-updated from Google Calendar attendee emails and open commitments surfaced as Notion tasks.

**Not in scope:** follow-up sweep or reminders, Honcho integration, email-based contact detection (deferred until E2 Gmail ships), multi-user access, any UI beyond Discord commands.

## Problem

Without this, relationship maintenance lives entirely in memory — there is no record of last contact dates, no place to surface open commitments per person, and no proactive signal when a relationship has gone cold. The data to auto-detect interactions already exists (Google Calendar attendees) but is never captured.

## Solution (from the user's perspective)

Jai tells Mahler `add contact Alice Chen alice@example.com professional — works at Sequoia, intro'd by Marcus`. Mahler writes the row to D1. Each morning, a cron runs `sync-calendar --days 1`, fetches yesterday's Google Calendar events, matches attendee emails against known contacts, and stamps `last_contact`. When Jai asks `summarize Alice Chen`, Mahler returns her contact card (name, type, email, last contact date, context note) plus any open Notion tasks prefixed `[Alice Chen]`. Jai can say `talked to Bob today` to manually stamp a contact for off-calendar interactions.

## Design

D1 is the only store (no Honcho). Open commitments live as Notion tasks with `[Name]` prefix — no new Notion schema needed, and existing notion-tasks skill handles creation. `last_contact` is auto-detected from Google Calendar attendee emails (primary) with manual override via Discord command (fallback). The calendar auto-detection reuses the existing `gcal_client.py` OAuth2 + event-fetch pattern from the `google-calendar` skill — no new auth flow.

Key decisions:

- **Notion tasks vs D1 column for commitments:** `[Name]` prefix on existing Notion tasks avoids schema complexity and reuses existing task infrastructure. Trade-off: task queries require a title-contains filter per contact, not a join.
- **Calendar auto-detection vs manual-only:** Calendar is primary (zero friction for the most common professional interactions), manual fallback for phone calls and in-person meetings. Requires `email` field on contact — makes adding a contact slightly heavier but enables reliable matching.
- **D1-only vs Honcho:** D1 only — the `context` text field handles freeform notes. Honcho can be added later if the single-field model proves limiting.
- **No follow-up sweep:** Removed from scope. The sweep's value proposition requires consistent data quality first; add it in a later iteration once contact data is established.

## Modules

### D1Client (`d1_client.py`)

- **Interface:** `ensure_table()`, `upsert_contact(name, email, type, context)`, `get_contact(name) → dict`, `list_contacts(type=None) → list[dict]`, `touch_last_contact(name, date) → None`, `delete_contact(name) → None`
- **Hides:** SQL construction, D1 REST API HTTP calls, parameterized query safety, schema creation/migration, HTTPS-only opener setup
- **Tested through:** public methods — mock `_OPENER`, verify outbound SQL + params, verify returned row dicts

### NotionClient (`notion_client.py`)

- **Interface:** `list_tasks_for_contact(name) → list[dict]`
- **Hides:** Notion API HTTP calls, title-contains + status-not-Done filter construction, pagination via `has_more` / `next_cursor`, property extraction from nested Notion response format
- **Tested through:** `list_tasks_for_contact` — mock `_OPENER`, verify outbound filter JSON, verify normalized task list

### contacts.py (CLI orchestrator)

- **Interface:** CLI subcommands: `add`, `update`, `summarize`, `list`, `talked-to`, `delete`, `sync-calendar`
- **Hides:** argparse, `_supplement_env_from_hermes()` bootstrap, orchestration of D1Client + NotionClient + gcal_client, stdout formatting
- **Tested through:** mock `D1Client` and `NotionClient` classes; capture stdout; assert output strings and method calls

### gcal_client.py (copied from google-calendar skill)

- **Interface:** `refresh_access_token(client_id, client_secret, refresh_token) → str`, `list_events(access_token, time_min, time_max, max_results) → list[dict]`
- **Hides:** OAuth2 token exchange, Google Calendar REST API HTTP calls, event normalization (attendee email extraction, all-day vs timed date parsing)
- **Tested through:** already tested in `google-calendar` skill; the copy is trusted and not re-tested

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/config/skills/relationship-manager/SKILL.md` | Hermes skill definition — triggers, env vars, command reference | New |
| `assistant/config/skills/relationship-manager/scripts/d1_client.py` | D1 CRUD for contacts table | New |
| `assistant/config/skills/relationship-manager/scripts/notion_client.py` | Notion tasks filtered by `[Name]` prefix | New |
| `assistant/config/skills/relationship-manager/scripts/gcal_client.py` | Copy of gcal_client from google-calendar skill | New |
| `assistant/config/skills/relationship-manager/scripts/contacts.py` | CLI orchestrator for all subcommands | New |
| `assistant/config/skills/relationship-manager/tests/conftest.py` | Add `../scripts` to sys.path for test imports | New |
| `assistant/config/skills/relationship-manager/tests/test_d1_client.py` | D1 client behavior tests | New |
| `assistant/config/skills/relationship-manager/tests/test_notion_client.py` | Notion client behavior tests | New |
| `assistant/config/skills/relationship-manager/tests/test_contacts.py` | CLI orchestrator behavior tests | New |
| `assistant/entrypoint.sh` | Register `relationship-manager` calendar-sync cron at 08:00 UTC | Modify |

## Open Questions

- Q: Should `get_contact` match by exact name or substring?  Default: exact case-insensitive match — avoids ambiguity when multiple contacts share a surname.
- Q: What happens when `sync-calendar` finds an event with multiple matching attendees?  Default: update `last_contact` for all matching contacts independently, using the event's start date for each.
