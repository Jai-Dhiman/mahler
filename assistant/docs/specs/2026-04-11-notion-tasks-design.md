# Notion Tasks Design

**Goal:** Manage personal tasks and todos in Notion via natural language through Discord — create, list, update, complete, and delete tasks without leaving the chat.

**Not in scope:**
- Subtasks or nested task hierarchies
- Task comments or rich text body content
- Notion page content beyond the four properties (Name, Status, Due, Priority)
- Recurring tasks or reminders
- Integration with Google Calendar (Phase 4 calendar work is separate)
- Any second LLM hop inside the skill script — NL parsing is Hermes's job

## Problem

There is no way to interact with the user's Notion task list through Mahler. Tasks must be managed directly in the Notion app. There is also no automated wiki bridge or Phase 7 sync mechanism — this skill is the first Notion integration and replaces the proposed Phase 7 local-wiki approach entirely.

## Solution (from the user's perspective)

The user tells Mahler in Discord: "add a task to review the TraderJoe backtest by Friday, high priority." Mahler creates the task in Notion and responds with the confirmation. Later, "what do I need to do this week?" returns a formatted list with IDs. "Mark the TraderJoe review done" completes it. IDs in list output let Hermes chain operations without a second search step.

## Design

**Approach:** Thin CLI script (`tasks.py`) dispatches to a deep Notion API client (`notion_client.py`). Hermes (already an LLM) handles natural language → structured CLI args via precise operation definitions in `SKILL.md`. No second LLM call inside the script — task operations are unambiguous once parsed, unlike email classification which requires judgment across hundreds of items.

**Key decisions:**
- NL → structured args happens in Hermes via SKILL.md, not in a script-level OpenRouter call. This eliminates a second LLM hop (~500ms) and an extra API dependency.
- `notion_client.py` owns all Notion API interaction including pagination, block format serialization, and response parsing. Nothing leaks to `tasks.py`.
- HTTPS-only opener (same SSRF-safe pattern as `d1_client.py`) — no external HTTP library.
- Pagination is fully transparent to callers: `list_tasks()` follows `next_cursor` until `has_more` is false.
- `delete_task` archives the Notion page (recoverable from Notion UI), but SKILL.md instructs Hermes to confirm with the user before invoking it.
- Date format contract: ISO 8601 (`YYYY-MM-DD`). SKILL.md instructs Hermes to convert "Friday" → absolute date before calling the script.
- `complete_task` is a thin wrapper over `update_task(page_id, status="Done")` — no separate API call.

**Notion API version:** `2022-06-28`

**New Fly.io secrets required:**
- `NOTION_API_TOKEN` — Notion internal integration token
- `NOTION_DATABASE_ID` — ID of the tasks database page

**Notion database schema (created manually once):**
- `Name` (title property) — required
- `Status` (select: `Todo` / `In Progress` / `Done`) — default `Todo` on create
- `Due` (date, optional)
- `Priority` (select: `High` / `Medium` / `Low`, optional)

## Modules

### `notion_client.py` — DEEP

**Interface:**
```python
class NotionClient:
    def __init__(self, api_token: str, database_id: str): ...
    def create_task(self, title: str, due: str | None, priority: str | None) -> dict: ...
    def list_tasks(self, status: str | None, priority: str | None, due_before: str | None) -> list[dict]: ...
    def update_task(self, page_id: str, **fields) -> dict: ...
    def complete_task(self, page_id: str) -> dict: ...
    def delete_task(self, page_id: str) -> None: ...
```

All methods return flat task dicts: `{"id": str, "title": str, "status": str, "due": str | None, "priority": str | None}`.

**Hides:**
- HTTPS opener construction (SSRF-safe, no FileHandler/FTPHandler)
- Notion API block format serialization (nested property JSON)
- Pagination cursor loop on `list_tasks`
- Response parsing and field extraction (`_extract_task`)
- HTTP status checking and error mapping (404 → `RuntimeError("Task not found: {id}")`)
- Notion API version header injection

**Tested through:** Public methods only. HTTP boundary mocked at `_OPENER.open` using `patch.object`.

**Depth verdict:** DEEP — 6-method interface hides Notion block format complexity, pagination, serialization, and HTTPS enforcement.

### `tasks.py` — SHALLOW (justified)

**Interface:** CLI only — `main(argv)` dispatches subcommands.

**Hides:** argparse wiring, env var loading, stdout formatting.

**Depth verdict:** SHALLOW by design. Its sole job is translating CLI args to `NotionClient` calls and formatting output. Depth lives in `notion_client.py`. A shallow dispatch layer is correct here.

### `SKILL.md` — contract artifact, not a module

The critical design artifact. Defines exact CLI signatures for each operation, the date format contract, priority inference rules, and the delete confirmation instruction. Vague SKILL.md = Hermes guessing = wrong operations. This is where most of the design effort for the Hermes integration lives.

## File Changes

| File | Change | Type |
|------|--------|------|
| `config/skills/notion-tasks/SKILL.md` | Hermes skill definition with CLI signatures | New |
| `config/skills/notion-tasks/scripts/notion_client.py` | Deep Notion API client | New |
| `config/skills/notion-tasks/scripts/tasks.py` | CLI entry point | New |
| `config/skills/notion-tasks/tests/test_notion_client.py` | Behavior tests for NotionClient | New |
| `config/skills/notion-tasks/tests/test_tasks.py` | Behavior tests for CLI dispatch | New |
| `Dockerfile` | Add COPY line for notion-tasks skill | Modify |

## Open Questions

- Q: Should `list` without filters return only `Todo` tasks by default (not Done)?  
  Default: No — return all tasks regardless of status. The user or Hermes can add `--status Todo` to filter. Surprising implicit filtering causes confusion.
