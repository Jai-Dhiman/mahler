# Evening Task Sweep (E8) Design

**Goal:** At 6pm Pacific daily, Mahler queries Notion for today's completed, past-due, and open tasks, posts a short summary to Discord, picks tomorrow's top 3 priorities, and checks in on overdue items.

**Not in scope:** Roll-count tracking, writing back to Notion automatically, LLM calls inside sweep.py, a --dry-run flag, modifications to the existing notion-tasks skill.

## Problem

There is no daily close ritual. The morning brief opens the day with email context; nothing closes it. There is no mechanism to surface tasks that slipped their due dates, and no prompt to think about tomorrow's focus before the workday ends.

## Solution (from the user's perspective)

At 6pm Pacific, a Discord message appears summarizing what was completed today, flagging any tasks past their due date with a direct check-in question ("Is [task] done?"), and presenting the agent's pick of the top 3 tasks to focus on tomorrow.

## Design

`sweep.py` is a pure data-fetching script: it makes three Notion queries and prints structured output to stdout. The Hermes agent (already running as the cron executor) reads that output, applies LLM reasoning to rank tomorrow's top 3 tasks, and posts the formatted summary to Discord via the gateway.

This keeps `sweep.py` independently testable (no Discord dependency, no LLM call) and lets the agent own the reasoning step without an external LLM API call in the script.

Past-due detection is purely query-based: tasks with `due < today` and `status != Done`. Python-side filter after the Notion response.

The `last_edited_after` filter required to find tasks completed today is a Notion timestamp filter — structurally different from property filters but combinable in AND clauses. `notion_client.py` is copied from `notion-tasks` and extended with this capability; the original is not modified.

## Modules

### NotionClient (evening-sweep/scripts/notion_client.py)
- **Interface:** `NotionClient(api_token, database_id)`, `.list_tasks(status, priority, due_before, last_edited_after) → list[dict]`; each dict contains `id, title, status, due, priority, last_edited_time`
- **Hides:** HTTP request construction, pagination loop, Notion API filter shapes, timestamp filter vs property filter distinction, `last_edited_time` extraction from page top-level field (not from properties)
- **Tested through:** `tests/test_notion_client.py` — patches `_OPENER.open` at HTTP boundary, asserts on request bodies and returned dicts

### sweep.py (evening-sweep/scripts/sweep.py)
- **Interface:** `python3 sweep.py` → stdout with three labeled sections; `main(argv=None, _today=None)` with injectable `_today: date` for test isolation
- **Hides:** Three Notion queries, Python-side filter excluding Done tasks from past-due bucket, days-overdue calculation, section formatting logic
- **Tested through:** `tests/test_sweep.py` — patches `sweep.NotionClient`, captures stdout, asserts on section headers and task lines

### SKILL.md (evening-sweep/SKILL.md)
- **Interface:** Hermes agent reads it to know when and how to invoke `sweep.py` and how to interpret the output
- **Hides:** Nothing — intentionally shallow coordination layer
- **Depth:** SHALLOW (correct — this is a prompt contract, not a code module)

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/config/skills/evening-sweep/scripts/notion_client.py` | Notion client with `last_edited_after` filter and `last_edited_time` extraction | New |
| `assistant/config/skills/evening-sweep/scripts/sweep.py` | Three-bucket Notion sweep, structured stdout | New |
| `assistant/config/skills/evening-sweep/SKILL.md` | Cron skill definition and agent prompt contract | New |
| `assistant/config/skills/evening-sweep/tests/test_notion_client.py` | Tests for `last_edited_after` filter and `last_edited_time` extraction | New |
| `assistant/config/skills/evening-sweep/tests/test_sweep.py` | Tests for stdout output structure across all three buckets | New |
| `assistant/Dockerfile` | Add COPY line for evening-sweep skill | Modify |
| `assistant/entrypoint.sh` | Add evening-sweep cron job at `0 1 * * *` | Modify |

## Open Questions

None — all design decisions resolved during brainstorm.
