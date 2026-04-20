# E3+ Expanded Kaizen Scope + Reflection Journal Design

**Goal:** Broaden the weekly kaizen loop to silently deposit project-activity and reflection patterns into Honcho memory, and add a standalone reflection-journal skill that collects weekly freeform reflections, stores them in D1, and concludes durable personal facts into Honcho.

**Not in scope:**
- Changing the email proposal flow (Discord approve/deny) in any way
- Reading from Honcho (Mahler's agent loop handles retrieval)
- Reflection question customization via config
- Life-log / habit tracking (E10)
- Any Discord approval gate for project-analysis or reflection-analysis conclusions

## Problem

The kaizen loop (`reflect.py --run`) only analyzes `email_triage_log`. Project wins and blockers accumulate in `project_log` (written by the SessionStop hook) but are never surfaced to Honcho as durable memory. Weekly reflection entries don't exist at all. Mahler has no persistent signal about work patterns, recurring blockers, or personal state beyond what Honcho incidentally captures from conversation.

## Solution (from the user's perspective)

Every Sunday:
1. **02:00 UTC** — Mahler posts three reflection questions to Discord in one message. The user replies with a freeform block. Mahler stores the raw reply in D1 and concludes 2-3 synthesized facts into Honcho. No further action required.
2. **18:00 UTC** — Kaizen runs as before, but after the email proposal pass it also silently analyzes `project_log` (last 7 days) and `reflection_log` (last 4 weeks) and concludes any detected patterns into Honcho. The user sees no additional Discord output from these passes.

Mahler gradually learns work patterns and personal preferences through Honcho without any manual effort after setup.

## Design

**Two-pass expansion of `reflect.py --run`:** The existing email proposal pass is untouched. Two new silent passes — `_run_project_analysis()` and `_run_reflection_analysis()` — are called sequentially after it. Each is wrapped in a bare `except Exception` that logs to stderr and continues, so a failure in either silent pass never blocks the email proposal output.

**Honcho constants are inlined** in both skill scripts (`_HONCHO_BASE_URL = "https://api.honcho.dev"`, `_HONCHO_APP_NAME = "mahler"`, `_HONCHO_USER_ID = "jai"`) matching the pattern established in `email-triage/scripts/triage.py`. `HONCHO_API_KEY` is read from the environment at call time. If absent, the silent pass logs a warning to stderr and skips — not a hard failure.

**LLM fact extraction:** Both silent passes and `journal.py --record` call the LLM and parse its output for lines prefixed with `FACT: `. Each such line becomes one `honcho_client.conclude()` call. `NO_PATTERNS` or no `FACT:` lines → zero conclude calls, no error.

**`reflection_log` table** is owned by the reflection-journal skill's `ensure_table()` call (run once manually after deploy, identical pattern to `migrate.py`). Kaizen's `_run_reflection_analysis()` treats an empty or missing `reflection_log` as a skip — it catches the D1 error and returns without logging anything to Discord.

**honcho_client.py is duplicated** across `kaizen-reflection/scripts/` and `reflection-journal/scripts/`. The implementation is ~50 lines with a fixed session_id per skill (`"kaizen-reflection"` and `"reflection-journal"` respectively). Shared library extraction is not done — Hermes skills deploy independently and a shared lib would complicate the Dockerfile.

**Reflection questions (fixed):**
```
How did last week go overall?
What drained your energy or felt hard this week?
What are you avoiding or putting off?
```

**journal.py --record** flow:
1. Write raw text + ISO week string to `reflection_log` in D1 (hard failure if D1 write fails)
2. Call LLM to synthesize facts (hard failure if LLM fails — D1 write already committed)
3. Call `honcho_client.conclude()` for each extracted fact (hard failure if Honcho fails)
4. Print `"Reflection recorded."` on success

## Modules

**kaizen-reflection/scripts/honcho_client.py**
- Interface: `conclude(text, api_key, base_url, app_name, user_id) -> None`
- Hides: session creation (idempotent POST, 409 accepted), metamessage POST, HTTP error mapping to RuntimeError
- Tested through: direct calls with mocked `_OPENER.open`

**kaizen-reflection/scripts/d1_client.py** (additions)
- Interface: `get_recent_project_log(since_days: int) -> list[dict]`, `get_recent_reflections(since_weeks: int) -> list[dict]`
- Hides: SQL parameterization, D1 REST call, response parsing
- Tested through: D1Client method calls with mocked `_OPENER.open`

**kaizen-reflection/scripts/reflect.py** (additions)
- Interface: `reflect.py --run` (unchanged CLI)
- Hides: three sequential analysis passes, exception isolation between passes, Honcho conclude calls
- Tested through: `reflect.main(["--run"])` with mocked D1Client and honcho_client

**reflection-journal/scripts/d1_client.py**
- Interface: `insert_reflection(week_of: str, raw_text: str) -> None`, `get_recent_reflections(since_weeks: int) -> list[dict]`, `ensure_table() -> None`
- Hides: SQL, D1 REST, response parsing, table DDL
- Tested through: D1Client method calls with mocked `_OPENER.open`

**reflection-journal/scripts/honcho_client.py**
- Interface: `conclude(text, api_key, base_url, app_name, user_id) -> None`
- Hides: session creation, metamessage POST, error mapping
- Tested through: direct calls with mocked `_OPENER.open`

**reflection-journal/scripts/journal.py**
- Interface: `journal.py --prompt`, `journal.py --record ANSWER_TEXT`
- Hides: question set, week-of calculation, D1 insert, LLM synthesis, Honcho conclude, env loading
- Tested through: `journal.main(["--prompt"])` and `journal.main(["--record", text])` with mocked collaborators

## File Changes

| File | Change | Type |
|------|--------|------|
| `config/skills/kaizen-reflection/scripts/honcho_client.py` | New module — conclude() for kaizen silent passes | New |
| `config/skills/kaizen-reflection/scripts/d1_client.py` | Add get_recent_project_log(), get_recent_reflections() | Modify |
| `config/skills/kaizen-reflection/scripts/reflect.py` | Add _run_project_analysis(), _run_reflection_analysis() silent passes | Modify |
| `config/skills/kaizen-reflection/tests/test_d1_client.py` | Tests for two new query methods | Modify |
| `config/skills/kaizen-reflection/tests/test_reflect.py` | Tests for two new passes + failure isolation | Modify |
| `config/skills/reflection-journal/SKILL.md` | Skill descriptor with cron + procedure | New |
| `config/skills/reflection-journal/scripts/d1_client.py` | D1 client for reflection_log | New |
| `config/skills/reflection-journal/scripts/honcho_client.py` | Honcho conclude for journal | New |
| `config/skills/reflection-journal/scripts/journal.py` | CLI: --prompt and --record | New |
| `config/skills/reflection-journal/tests/test_d1_client.py` | Tests for insert_reflection, get_recent_reflections | New |
| `config/skills/reflection-journal/tests/test_journal.py` | Tests for --prompt and --record behaviors | New |
| `Dockerfile` | COPY reflection-journal skill | Modify |
| `entrypoint.sh` | Register reflection-journal cron at 0 2 * * 0 | Modify |

## Open Questions

- Q: Should `_run_project_analysis()` skip silently when `HONCHO_API_KEY` is missing, or raise?
  Default: skip with `sys.stderr.write("WARNING: HONCHO_API_KEY not set — skipping project analysis\n")` — consistent with how email-triage handles missing Honcho key.
