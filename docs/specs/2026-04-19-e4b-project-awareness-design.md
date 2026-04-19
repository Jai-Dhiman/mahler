# E4b Project Awareness Design

**Goal:** Mahler surfaces recent development wins and blockers from Claude Code sessions in every conversation, giving it live context about what is being built and where work is stuck.

**Not in scope:** Automated weekly project health notifications (deferred to E3+), sentiment analysis or frustration scoring, tracking sessions outside Claude Code, multi-project aggregation views, or any push notification to Discord.

## Problem

Mahler has no awareness of active development work. When asked "what should I focus on?" or "how is the trading system going?", it has no real-time signal beyond what the user says in the current turn. Blockers articulated during Claude Code sessions — "the main issue is that the model isn't giving accurate scores" — are never surfaced in later Mahler conversations. Shipped features go unnoticed too: Mahler cannot reference what was just built without the user re-explaining it.

## Solution (from the user's perspective)

After shipping a feature with `/ship`, Mahler knows what shipped and can reference it. After a development session where a blocker is articulated, Mahler knows about it in subsequent conversations — "you mentioned yesterday that the scoring model is producing wrong outputs, any progress on that?" The context is injected automatically; the user does nothing extra.

## Design

Option B from the brainstorm: `/ship` synthesizes wins inline (Claude is already running — zero additional API cost), and a `SessionStop` Claude Code hook scans for blocker keywords at session end (free), calling OpenRouter/Grok only if a keyword match is found.

`project_log.py` is a two-mode CLI script in `assistant/hooks/`. The `blocker` mode runs as a Claude Code `Stop` hook; the `win` mode is called explicitly by the `/ship` skill after merging. Both write to a new `project_log` D1 table via the existing `D1Client`. Credentials live in `~/.mahler.env` (four keys: `CF_ACCOUNT_ID`, `CF_D1_DATABASE_ID`, `CF_API_TOKEN`, `OPENROUTER_API_KEY`) — scoped separately from the assistant's Fly.io secrets.

The `project-context` Hermes plugin follows the identical pattern as `kaizen-context`: `pre_llm_call` → D1 query → context string or None. Never raises.

Keyword set for blocker detection (case-insensitive scan of user-role messages): `stuck`, `blocked`, `can't figure out`, `cannot figure out`, `issue is`, `problem is`, `broken`, `failing`, `doesn't work`, `not working`, `frustrat`.

## Modules

**`assistant/hooks/project_log.py`**
- Interface: `log_win(project: str, summary: str, git_ref: str) -> None` and `log_blocker_if_triggered(transcript: dict, cwd: str) -> None`. Also a CLI entry point: `blocker` mode (stdin JSON) and `win` mode (`--project`, `--summary`, `--git-ref` flags).
- Hides: loading `~/.mahler.env`, keyword scan over user messages, OpenRouter API call construction and response parsing, project name derivation from `git remote get-url origin`, `D1Client` setup and parameterized INSERT, top-level exception swallowing in `blocker` mode.
- Tested through: `log_win(...)` and `log_blocker_if_triggered(...)` public functions with `D1Client.query` and `_call_openrouter` stubbed.

**`assistant/config/plugins/project-context/plugin.py`**
- Interface: `project_context(session_id: str, user_message: str, is_first_turn: bool, **kwargs) -> dict | None` and `register(ctx)`.
- Hides: loading `~/.hermes/.env`, `sys.path` injection to reach `D1Client`, SQL query for last 7 days of `project_log`, row grouping and formatting, all exception handling.
- Tested through: `project_context(...)` return value with `_query_project_log` stubbed.

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/hooks/project_log.py` | Two-mode CLI: `blocker` (Stop hook) and `win` (ship skill call) | New |
| `assistant/hooks/tests/test_project_log.py` | Behavioral tests for `log_win` and `log_blocker_if_triggered` | New |
| `assistant/config/plugins/project-context/plugin.py` | Hermes `pre_llm_call` plugin, same pattern as `kaizen-context` | New |
| `assistant/config/plugins/project-context/tests/test_plugin.py` | Behavioral tests for `project_context` | New |
| `assistant/config/skills/email-triage/scripts/d1_client.py` | Add `project_log` table to `ensure_tables()`, add `insert_project_log()` and `get_recent_project_log()` methods | Modify |
| `assistant/config/skills/email-triage/tests/test_d1_client.py` | Add tests for the three new `D1Client` methods | Modify |
| `~/.claude/settings.json` | Add `Stop` hook entry pointing to `project_log.py blocker` | Modify |
| `~/.claude/skills/ship/SKILL.md` | Add Step 1.5: synthesize win summary, call `project_log.py win` | Modify |
| `assistant/Dockerfile` | Already covered: `COPY config/plugins` copies all plugins including `project-context` | No change needed |

## Open Questions

- Q: Does the Claude Code `Stop` hook payload include a `cwd` field?  Default: use `data.get("cwd", os.getcwd())` — falls back to process working directory if the field is absent.
