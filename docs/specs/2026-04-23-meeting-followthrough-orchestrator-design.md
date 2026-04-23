# Meeting Follow-Through Orchestrator Design

**Goal:** Replace the LLM-agent-driven meeting-followthrough skill with a deterministic Python orchestrator so that empty cron ticks cost near-zero and Step 8 (Discord post) can never be silently skipped.
**Not in scope:**
- Changing the `fathom-webhook` Cloudflare Worker (already fixed separately).
- Moving cron dispatch away from Hermes (we still fire via `hermes cron`; the skill body is just minimal).
- Refactoring other skills (email-triage, meeting-prep, etc.) to the same pattern.
- Building any skip-if-empty mechanism at the Hermes layer.

## Problem

The `meeting-followthrough` skill (`assistant/config/skills/meeting-followthrough/SKILL.md`) is an LLM-agent procedure. Hermes cron fires the skill every 5 min; the skill LLM (currently `openai/gpt-5-nano`) reads the SKILL.md body, decides which bash commands to run, and produces a Discord message at the end. Two observed failures:

1. **Step 8 silently skipped.** On 2026-04-23 05:51:46 UTC, the skill processed two queued Fathom test calls in D1 (both rows got `processed_at` set) but posted no Discord summary. Step 8 in SKILL.md is the only step without an explicit bash command — the LLM is expected to know how to invoke the Discord webhook. It didn't.
2. **Heredoc tangle on recovery attempt.** A later run (after we added `scripts/post_discord.py`) posted a message where the heredoc terminator and the following `&&` command leaked into the message body: `MSG && python3 ~/.hermes/skills/meeting-followthrough/scripts/post_discord.py < /tmp/discord_post_input.txt`. The LLM got the shell syntax wrong.

Secondary: every cron tick costs an LLM call even when `poll.py fetch` returns `NO_PENDING_MEETINGS`. At 5-min cadence that's ~288 empty-tick LLM calls per day.

## Solution (from the user's perspective)

When a Fathom meeting is recorded:
1. The Cloudflare Worker enqueues it to D1 (unchanged).
2. Within ~15 min, a Discord message lands in the triage channel with the meeting title, the action items that were auto-created in Notion, and which CRM contacts were updated.
3. If no meetings are pending, nothing happens — no Discord noise, no wasted LLM tokens.

## Design

**Move all workflow logic into a single Python orchestrator invoked by the cron skill.** The skill body shrinks to "run `orchestrate.py`; if it prints anything, surface it; if it exits non-zero, post the error." The orchestrator does the entire flow deterministically except for the one place reasoning is actually required — generating action items from the meeting summary — which is a direct OpenRouter call from inside the script.

Decisions and trade-offs:

- **One file, one command.** `orchestrate.py` owns the whole cron-tick workflow. Trade-off: tight coupling between D1 polling, CRM fan-out, LLM call, Notion writes, and Discord post. Accepted because they share a single lifecycle (one meeting → one set of side effects) and splitting introduces cross-file state-passing without hiding complexity.
- **Subprocess boundary to other skills' scripts (`contacts.py`, `tasks.py`).** They live in other skill directories, load their own env, and already have stable CLIs. Shelling out keeps them as black boxes; importing them would require restructuring their path/env handling.
- **Import `D1Client` in-process** (not shelling out to `poll.py`). The class already exists in the same package — subprocess-ing our own helper adds stdout-parsing complexity for no isolation benefit.
- **Model selection matches email-triage exactly.** `_DEFAULT_MODEL = "openai/gpt-5-nano"` constant at top of file; `os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)` at call site. To shift: change the env var (shared with email-triage) or edit the constant.
- **Cron cadence `*/5` → `*/15`.** Post-mortems don't need sub-5-min latency. Cuts tick count by 3×.
- **`poll.py` kept during rollout, deleted after ship.** Belt-and-suspenders in case orchestrate.py has a bug we only catch in production; `poll.py` is the manual-debug escape hatch. Removal tracked as the final task in the plan.
- **SKILL.md becomes a thin pass-through.** Body is ~5 lines. LLM cost on empty ticks drops to a trivial prompt + one-sentence reply (~few hundred tokens on `gpt-5-nano`).

Why not go further and skip the LLM entirely on empty ticks? Hermes cron dispatches skills, not bash commands — there's no `--skip-if-script-empty` flag. Removing the LLM entirely would require OS-level cron or a Hermes upgrade, both out of scope.

## Modules

### `scripts/orchestrate.py` (new, DEEP)
- **Interface:**
  - `main(argv: list[str] | None = None) -> int` — CLI entry; returns process exit code. Prints a one-line status to stdout.
  - `process_meeting(row: dict, *, runner, llm_caller, discord_poster, d1_client) -> str` — processes one queued meeting end-to-end; returns the Discord summary line that was posted.
  - `generate_action_items(summary: str, attendees: list[dict], crm_context: dict[str, str], open_tasks: list[str], llm_caller) -> list[dict]` — builds prompt, calls LLM, parses response; returns `[{"title": str, "priority": "High"|"Medium"|"Low", "attendee": str|None}]`.
- **Hides:** D1 polling, per-attendee CRM summarize fan-out, Notion task-list fetch, OpenRouter HTTP call, Notion create-task calls, CRM talked-to updates, Discord post, D1 mark-done, error aggregation, stdout summary formatting.
- **Tested through:** `main()` (end-to-end with all four deps stubbed) and `generate_action_items()` (pure function with `llm_caller` stubbed).

### `scripts/poll.py` (existing, unchanged during rollout; deleted in final task)
- Manual-debug CLI. Kept during rollout for safety; removed once orchestrate.py has run cleanly on at least one real meeting.

### `scripts/post_discord.py` (existing, unchanged)
- Helper that posts stdin content to `$DISCORD_TRIAGE_WEBHOOK`. Imported by `orchestrate.py`.

### `scripts/d1_client.py` (existing, unchanged)
- `D1Client` class. Imported by `orchestrate.py`.

### `SKILL.md` (existing, rewritten body)
- Frontmatter unchanged. Body becomes: "run `orchestrate.py`, report its stdout verbatim, surface any non-zero exit via Discord."

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/config/skills/meeting-followthrough/scripts/orchestrate.py` | New orchestrator; imports `D1Client` and `post_discord` helpers | New |
| `assistant/config/skills/meeting-followthrough/tests/__init__.py` | Empty — makes tests package-importable | New |
| `assistant/config/skills/meeting-followthrough/tests/test_orchestrate.py` | Behavior tests for `orchestrate.py`, stubbing subprocess / LLM / Discord / D1 at the boundary | New |
| `assistant/config/skills/meeting-followthrough/SKILL.md` | Body rewritten to invoke `orchestrate.py`; Step 1–8 procedure removed | Modify |
| `assistant/config/skills/meeting-followthrough/scripts/poll.py` | Deleted in final task (after production verification) | Delete |
| `assistant/Dockerfile` | No changes — existing COPY already picks up new files under `config/skills/` | — |
| Hermes cron entry `meeting-followthrough` | Schedule `*/5` → `*/15` via `hermes cron edit`; runtime state only, not in git | Modify (runtime) |

## Open Questions

- Q: What should `orchestrate.py` do when `generate_action_items()` raises (e.g., OpenRouter 500)?  Default: catch the exception, post an error message to Discord naming the meeting, and exit non-zero *without* calling `poll.py mark-done` — so the next tick retries. If OpenRouter is down for >1 hour, the user sees repeated errors in Discord, which is the desired signal.
- Q: What if `tasks.py create` fails partway through a list of action items (e.g., 3 of 5 created)?  Default: stop processing further action items for that meeting, post what was created + an error line, mark the meeting done anyway (don't retry — avoids duplicate tasks on replay).
- Q: Should the CRM update (Step 6) run for the user's own email as an attendee?  Default: no — `contacts.py summarize --name "Jai Dhiman"` will likely fail or be a no-op. The orchestrator skips attendees whose email matches `os.environ["MAHLER_OWNER_EMAIL"]`. If that env var is unset, no attendees are skipped (behavior degrades gracefully rather than hard-failing).
