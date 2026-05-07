# Daily Synthesis Brief Design

**Goal:** Receive a daily Discord push (CONNECTIONS / PATTERN / QUESTION) at ~6am PT weekdays that surfaces non-obvious links across recent and older notes, so the personal knowledge system pushes synthesis instead of waiting to be queried.

**Not in scope:**
- Weekend posts (Mon–Fri only).
- Local memory dir or wiki content reaching Fly.io directly (only via the Stop hook's D1 mirror).
- Replacing the existing 8am email morning-brief — it remains; the synthesis is prepended as a new field.
- Voice / image / mobile-push delivery.
- Two-way ("ask me a question, take my answer") flows.

## Problem

Today the user pulls from Honcho, the wiki, and `~/.claude/.../memory/` only when explicitly asking. Vaults that don't push back become graveyards. Specifically:

- Honcho conclusions accumulate but only surface on the Sunday `memory-kaizen` distillation.
- Notion wiki concept pages are never re-read once written.
- D1 `project_log` wins (1,453 rows, 27 wins in 18 days) are not synthesized except weekly.
- Local memory files (`~/.claude/projects/-Users-jdhiman-Documents-mahler/memory/`) and recent git activity across `~/Documents/*` repos are completely invisible to anything Fly-resident.
- The `project_log` blockers channel is dead — keyword scan in `assistant/hooks/project_log.py:11-23` has produced 0 rows in 18 days.

## Solution (from the user's perspective)

Each weekday morning at ~8am PT a single Discord message lands containing four sections:
1. **Synthesis** (new, prepended): CONNECTIONS / PATTERN / QUESTION derived from the last 1–14 days.
2. Needs Action (existing, email).
3. FYI (existing, email).
4. Noise (existing, email).
5. What's Worth Reading (existing, news).

If the synthesis brief was skipped (thin context, validator failure, or staler than 24h), the Synthesis section is omitted and the message looks identical to today's morning-brief.

A second invisible improvement: every time the user finishes a Claude Code session, the Stop hook also (a) ships memory-dir deltas and last-24h git commits to D1 so synthesis has fresh inputs, and (b) re-classifies blockers via a headless `claude -p` subprocess instead of the keyword list, so the blockers channel finally produces signal.

## Design

### Approach

1. **New Fly.io skill `synthesis-brief`** runs at `0 13 * * 1-5` UTC (5am PDT / 6am PST). It loads inputs, calls the LLM with a strict JSON-output prompt, validates the result, and writes the brief to two places:
   - `synthesis_brief` table (history).
   - `mahler_kv` row at key `synthesis_brief:latest` (for morning-brief to read).
2. **Existing `morning-brief`** (8am PT) reads `synthesis_brief:latest`. If present and `posted_at` ≥ now − 24h, prepend a Synthesis embed field.
3. **Extended Stop hook** ships local context to a new `local_capture` D1 table on every session end, idempotent via `content_hash UNIQUE`.
4. **Stop hook blocker classifier** swapped from OpenRouter to `claude -p` subprocess (uses Pro subscription, no API cost).

### Why this over alternatives

- **Why a new skill, not an extension of morning-brief?** Synthesis runs early and may take ~10s of LLM time; morning-brief runs at a fixed 8am gate and shouldn't stretch. Decoupling lets thin-context days skip cleanly without touching the email path.
- **Why D1+KV-via-mahler_kv, not Cloudflare KV?** The `mahler_kv` D1 table already exists with `get_kv`/`set_kv` helpers (`assistant/config/skills/email-triage/scripts/d1_client.py:67-79`). Using it avoids a second binding and keeps the cron self-contained.
- **Why negative-list dedup, not positive context?** Drift loop risk — the model would treat its own past synthesis as fact. Negative list (last 14d of briefs) is fed only as "DO NOT REPEAT," never as truth.
- **Why local sync via existing Stop hook, not a new launchd agent?** Zero new infra, reuses CF creds already loaded from `~/.mahler.env`. Trade-off: misses days when the user doesn't open Claude Code (rare based on 835 sessions/7d).
- **Why bundle the blocker classifier swap?** Stop hook is being heavily edited anyway; doing both at once minimises hook-test churn.

### Quality gates (all three)

- **must-cite-source:** Each of the 3 connections must reference ≥2 distinct items from `inputs.identifiers` (`{source, id}` tuples). If <2 of 3 connections cite ≥2 valid identifiers, abort with reason `"insufficient_citations"`.
- **length cap:** Each section ≤600 chars; total ≤2000 chars. Overshoot → abort with `"length_exceeded"` (no truncation; we want the model to learn).
- **thin-context skip:** If `len(local_capture last-24h) + len(project_log wins last-24h) + len(honcho conclusions last-24h) < 3` AND `same totals over last-7d < 5`, abort with `"thin_context"` before any LLM call.

On any abort, the cron logs the reason to stdout, posts no Discord message, and exits 0. The morning-brief at 8am will see no fresh KV row and silently omit the section.

### Cron timing

`0 13 * * 1-5` UTC = 5am PDT / 6am PST, Monday–Friday. We deliberately do not DST-compensate; the morning-brief at 8am has a 2–3h gap either way.

## Modules

### `inputs.py`

- **Interface:** `load_all(d1: D1Client, honcho, notion_client, recent_days=1, context_days=14) -> InputBundle`
- **Hides:** D1 SQL for `project_log` wins and `local_capture`; Honcho SDK call; Notion DB query; identifier minting (`f"{source}:{row_id}"`); time-window math; past-brief negative-list assembly.
- **Returns:** `InputBundle` dataclass with attributes `recent_items: list[Item]`, `context_items: list[Item]`, `past_briefs: list[dict]`, `identifiers: set[str]`. `Item` is `{source, id, content, captured_at}`.
- **Tested through:** `test_inputs.py` exercises `load_all` with stubbed D1/Honcho/Notion clients, asserts on returned `InputBundle` shape and contents.

### `validator.py`

- **Interface:** `validate(brief: dict, inputs: InputBundle) -> tuple[bool, str]`
- **Hides:** Three independent gates (`_check_citations`, `_check_length`, `_check_thin_context`). Returns `(True, "")` or `(False, reason_code)`.
- **Reason codes:** `"thin_context"`, `"insufficient_citations"`, `"length_exceeded"`, `"malformed"`.
- **Tested through:** `test_validator.py` feeds canned brief dicts + canned bundles, asserts on `(ok, reason)`.

### `synthesize.py` (orchestrator)

- **Interface:** `main(argv)` — argparse on `--run` and `--dry-run`.
- **Flow:** load env → build clients → `inputs.load_all(...)` → thin-context pre-check → build prompt → call LLM (OpenRouter `openai/gpt-5-nano`) → parse JSON → `validator.validate(...)` → on success, write `synthesis_brief` row + `mahler_kv` row + post Discord status line; on failure, log reason and exit 0.
- **Justification for thinness:** ~80 lines. Composes deep modules; itself trivial.

### `assistant/hooks/project_log.py` (extended)

- **New helpers:** `_sync_memory_dir(d1)` (diff `~/.claude/projects/.../memory/*.md` against last-known content_hashes, INSERT new rows), `_sync_git_recent(d1)` (for each repo under `~/Documents/*`, `git log --since=24.hours.ago --pretty=format:'%h %s'`, INSERT new rows), `_classify_blocker_via_claude(transcript)` (subprocess `claude -p '<prompt>'`, parse one-line response, return summary string).
- **Public entrypoints unchanged:** `log_session_heartbeat`, `log_blocker_if_triggered`, `log_win` keep their signatures.
- **`stop` mode flow updated:** `log_blocker_if_triggered` (new classifier) → `log_session_heartbeat` → `sync_local_to_d1` (new top-level: calls memory + git helpers, swallows per-source errors so one source's failure doesn't kill the others).

### `morning-brief/scripts/brief.py` (extended)

- **`build_embed` gains keyword `synthesis_section: dict | None = None`** with shape `{"connections": [...], "pattern": str, "question": str}`.
- **`main()` reads `mahler_kv:synthesis_brief:latest`**: if row exists, parse JSON, check `posted_at ≥ now − 24h`, pass into `build_embed`. If absent or stale, omit (no error).
- **Embed change:** if `synthesis_section`, prepend a single field `Synthesis` containing formatted Connections / Pattern / Question.

### Schema additions

```sql
CREATE TABLE IF NOT EXISTS local_capture (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL CHECK(source IN ('memory','git')),
  project TEXT,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  captured_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_local_capture_recent ON local_capture(captured_at);

CREATE TABLE IF NOT EXISTS synthesis_brief (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  posted_at TEXT NOT NULL DEFAULT (datetime('now')),
  connections_json TEXT NOT NULL,
  pattern TEXT NOT NULL,
  question TEXT NOT NULL
);
```

Tables are created lazily by `inputs.py` and `project_log.py` via `CREATE TABLE IF NOT EXISTS` on first call (matches existing convention in `assistant/config/skills/email-triage/scripts/d1_client.py`).

## File Changes

| File | Change | Type |
|---|---|---|
| `assistant/config/skills/synthesis-brief/SKILL.md` | New skill manifest, cron + manual invocation docs | New |
| `assistant/config/skills/synthesis-brief/scripts/inputs.py` | Deep module: 5-source loader returning `InputBundle` | New |
| `assistant/config/skills/synthesis-brief/scripts/validator.py` | Deep module: 3-gate validator returning `(ok, reason)` | New |
| `assistant/config/skills/synthesis-brief/scripts/synthesize.py` | Thin orchestrator with `--run`/`--dry-run` | New |
| `assistant/config/skills/synthesis-brief/tests/__init__.py` | empty | New |
| `assistant/config/skills/synthesis-brief/tests/test_inputs.py` | Behavior tests for `load_all` | New |
| `assistant/config/skills/synthesis-brief/tests/test_validator.py` | Behavior tests for `validate` | New |
| `assistant/config/skills/synthesis-brief/tests/test_synthesize.py` | E2E behavior tests for `main` (stubbed LLM) | New |
| `assistant/hooks/project_log.py` | Add `sync_local_to_d1`; swap blocker classifier to `claude -p` | Modify |
| `assistant/hooks/tests/__init__.py` | empty | New |
| `assistant/hooks/tests/test_project_log.py` | Behavior tests for new sync + new classifier | New |
| `assistant/config/skills/morning-brief/scripts/brief.py` | `build_embed` accepts `synthesis_section`; `main` reads `mahler_kv` | Modify |
| `assistant/config/skills/morning-brief/tests/test_brief.py` | Add prepend test | Modify |
| `assistant/Dockerfile` | `COPY` `synthesis-brief` skill into `/home/hermes/.hermes/skills/` | Modify |
| `assistant/entrypoint.sh` | Register `synthesis-brief` cron `0 13 * * 1-5` | Modify |

## Open Questions

- **Q: Should the Discord post on dry-run / abort cases be a quiet log line or a Discord status message?** Default: silent — only successful briefs reach Discord. Aborts log to stdout (visible via `flyctl logs`).
- **Q: When `local_capture` row count grows large, should we prune?** Default: no pruning in v1. The `idx_local_capture_recent` index keeps the 7-day query cheap; revisit at 100k rows.
- **Q: What happens if `claude` CLI is not installed on the laptop running the hook?** Default: blocker classifier returns empty string and no `blocker` row is written; hook continues (memory + git sync still run). Logged once per session via `print(..., file=sys.stderr)`.
