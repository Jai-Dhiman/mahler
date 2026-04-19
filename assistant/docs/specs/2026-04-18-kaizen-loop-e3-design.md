# Kaizen Loop (Phase E3) Design

**Goal:** The email priority map stored in Cloudflare D1 improves automatically each week based on observed triage patterns, with per-proposal Discord approval before any change is applied.

**Not in scope:**
- Explicit user feedback (no "mark as handled" commands)
- SOUL.md edits
- Outcome log table separate from `email_triage_log`
- iMessage, Notion, or calendar signal inputs
- Multi-user support

---

## Problem

The email classifier in `triage.py` loads its priority map from a file at `/home/hermes/.hermes/workspace/priority-map.md`. This file lives inside the Docker container and is wiped on every `flyctl deploy`. Any manual edits to the priority map do not survive a redeploy. The classifier never adapts to actual email patterns — a sender that has appeared 50 times as NEEDS_ACTION with no follow-up action stays NEEDS_ACTION forever.

---

## Solution (from the user's perspective)

The priority map lives in D1 and survives deploys. Every Sunday evening, Mahler posts a numbered list of proposed reclassifications to Discord — one approve/deny button per proposal. Approving a proposal immediately rewrites the D1 priority map. The updated map is used for all future triage runs and injected into Mahler's LLM context on every chat turn.

---

## Design

**Priority map storage:** Migrated from container filesystem to the `priority_map` table in `mahler-db`. One row: `content TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL`. Durable across deploys. `triage.py` reads from D1 at runtime; the fallback filesystem path is removed.

**Passive inference:** No user action required to train the loop. The only signal is the existing `email_triage_log` table. The weekly reflection queries senders or domains appearing 3 or more times at the same classification tier in the past 7 days and flags them as candidates for reclassification.

**Weekly reflection:** `kaizen-reflection` skill, cron-triggered every Sunday at 18:00 UTC (11am PST). `reflect.py --run` queries D1 patterns, calls OpenRouter to generate structured proposals, and outputs a JSON array to stdout. Mahler presents each proposal to Discord with per-proposal approve/deny buttons. On approval, Mahler calls `reflect.py --apply PROPOSAL_JSON`, which reads the current priority map from D1, calls OpenRouter to apply the edit, and writes the result back with `version++`.

**Context injection:** `kaizen-context` plugin uses the `pre_llm_call` hook to inject the current priority map content from D1 into every LLM turn. Silent-fail on D1 error (same pattern as `calendar-aware`). Injects the map content, not raw triage rows.

**Key trade-offs chosen:**
- Passive inference over explicit feedback: no friction, relies on volume signal (3+ occurrences)
- Priority map only (not SOUL.md): keeps the loop auditable; SOUL.md edits require human judgment
- Per-proposal buttons: more clicks than batch-approve, but prevents bulk bad changes from one bad reflection run
- D1 over Fly volume: no new infra, same API pattern already in use

---

## Modules

**`email-triage/scripts/d1_client.py` (modified)**
- Interface: `get_priority_map() -> str`, `set_priority_map(content: str) -> None`, updated `ensure_tables()`
- Hides: D1 SQL for `priority_map` table, version increment logic in `set_priority_map`, HTTPS-only opener
- Tested through: public method calls with mocked HTTP responses

**`email-triage/scripts/triage.py` (modified)**
- Interface: unchanged CLI (`python3 triage.py [--dry-run] [--since-hours N]`)
- Hides: D1 priority map read replacing filesystem read; `_load_priority_map()` removed
- Tested through: integration test — D1 mock returns known map, assert classification uses it

**`kaizen-context/plugin.py` (new)**
- Interface: `register(ctx)`, `priority_map_context(session_id, user_message, is_first_turn, **kwargs) -> dict | None`
- Hides: D1 read, `~/.hermes/.env` loading, sys.path manipulation to reach skill d1_client
- Depth: SHALLOW by design (plugin hooks are intentionally thin; depth is in D1 and map content)
- Tested through: public `priority_map_context()` with mocked `_query_priority_map()`

**`kaizen-reflection/scripts/d1_client.py` (new)**
- Interface: `get_triage_patterns(since_days: int, min_count: int) -> list[dict]`, `get_priority_map() -> str`, `set_priority_map(content: str) -> None`, `ensure_priority_map_table() -> None`
- Hides: GROUP BY SQL aggregation across `email_triage_log`, version management, HTTPS-only opener
- Depth: DEEP

**`kaizen-reflection/scripts/reflect.py` (new)**
- Interface: CLI `python3 reflect.py --run [--since-days N] [--dry-run]` | `python3 reflect.py --apply PROPOSAL_JSON`
- Hides: pattern fetch, LLM proposal generation prompt, JSON proposal serialization, LLM-assisted map edit, D1 write-back with version increment
- Depth: DEEP — simple two-mode CLI interface over substantial orchestration logic

**`kaizen-reflection/scripts/migrate.py` (new)**
- Interface: CLI `python3 migrate.py --file PATH`
- Hides: reads file, validates D1 table exists, inserts initial row; raises RuntimeError if row already exists
- Depth: SHALLOW but intentional (single-purpose one-time migration utility)

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `config/skills/email-triage/scripts/d1_client.py` | Add `get_priority_map()`, `set_priority_map()`, add priority_map table to `ensure_tables()` | Modify |
| `config/skills/email-triage/scripts/triage.py` | Replace `_load_priority_map()` with `d1.get_priority_map()` inline | Modify |
| `config/skills/email-triage/tests/test_d1_client.py` | Add tests for `get_priority_map()`, `set_priority_map()` | Modify |
| `config/skills/email-triage/tests/test_triage_integration.py` | Remove `_load_priority_map` patch; set `d1.get_priority_map.return_value` | Modify |
| `config/skills/kaizen-reflection/SKILL.md` | New skill definition with --run / --apply procedure | New |
| `config/skills/kaizen-reflection/scripts/d1_client.py` | D1Client for reflection: patterns query + priority map read/write | New |
| `config/skills/kaizen-reflection/scripts/reflect.py` | Main reflection script: --run and --apply modes | New |
| `config/skills/kaizen-reflection/scripts/migrate.py` | One-time migration: seeds priority_map table from file | New |
| `config/skills/kaizen-reflection/tests/test_d1_client.py` | Tests for get_triage_patterns, get/set priority map | New |
| `config/skills/kaizen-reflection/tests/test_reflect.py` | Tests for --run, no-proposals, --apply modes | New |
| `config/plugins/kaizen-context/plugin.py` | pre_llm_call hook injecting priority map from D1 | New |
| `config/plugins/kaizen-context/tests/test_plugin.py` | Tests for context injection and silent-fail on D1 error | New |
| `entrypoint.sh` | Register `kaizen-reflection` cron job: Sundays 18:00 UTC | Modify |
| `Dockerfile` | Add COPY for kaizen-reflection skill | Modify |

---

## Open Questions

None. All design decisions resolved in brainstorm.
