# Memory Loop Design

**Goal:** Hermes learns from Claude Code sessions and improves its memory quality over time through two weekly Sunday skills: `project-synthesis` (D1 → Honcho) and `memory-kaizen` (Honcho → distilled Honcho), backed by a shared `honcho-ai` SDK client that replaces three duplicate v1 API clients.

**Not in scope:**
- Real-time (per-session) Honcho writes — the loop is weekly batch only
- Deleting or updating existing Honcho conclusions
- Reading conclusions written by the old v1 API (different storage path; those fade naturally)
- Changes to the D1 `project_log` write path (SessionStop hook is unchanged)
- Agent watchdog or podcast digest features

---

## Problem

Three per-skill `honcho_client.py` files duplicate identical v1 API code (`/v1/apps/mahler/users/jai/sessions/.../metamessages`). None can read back what was written.

`kaizen-reflection/reflect.py` has `_run_combined_analysis()` — an ad-hoc tack-on that reads D1 project log + reflections and concludes patterns into Honcho. It silently skips when `HONCHO_API_KEY` is absent and is never tested.

Hermes has no dedicated, reliable way to synthesize what Jai is working on from coding sessions, and no mechanism to distill accumulated Honcho conclusions into high-signal durable facts.

---

## Solution (from the user's perspective)

Every Sunday at 18:00 UTC, `project-synthesis` reads the week's coding session wins and blockers from D1, synthesizes one cross-project paragraph about where Jai's attention is and what the trajectory looks like, and writes it to Honcho. Hermes confirms in Discord.

Every Sunday at 19:00 UTC, `memory-kaizen` reads the last 30 days of Honcho conclusions, identifies 2–4 high-signal durable patterns (only those appearing across multiple entries), and writes each as a new conclusion. Hermes reports how many patterns were written.

Both skills use a shared `honcho_client.py` backed by the `honcho-ai` SDK. The three existing per-skill clients are deleted. `kaizen-reflection`'s `_run_combined_analysis` is removed — its job is now split correctly between `project-synthesis` and `reflection-journal`.

---

## Design

### honcho-ai SDK

Package: `honcho-ai` (v2.1.1+). Import: `from honcho import Honcho`.

The v2 API is peer-centric. `honcho.json` maps to:
- `workspace` → `workspace_id` (passed to `Honcho(workspace_id=...)`)
- `aiPeer` → `ai_peer_id` — the agent peer that "knows about" the user
- `peerName` → `user_peer_id` — the user peer that conclusions are about

Conclusion operations:
```python
honcho = Honcho(workspace_id="mahler", api_key=api_key)
peer = honcho.peer("mahler")                    # agent peer
conclusions = peer.conclusions_of("jai")        # what mahler knows about jai

conclusions.create([{"content": "...", "session_id": "project-synthesis"}])
list(conclusions.list())                        # all conclusions
conclusions.query("what is Jai working on")     # semantic search
```

**Migration risk:** Old v1 `metamessages` API and new SDK conclusions are separate storage. Conclusions previously written by the v1 clients will not appear in `conclusions.list()`. We accept this — proceed with new SDK going forward; old entries fade naturally as Honcho's background reasoning updates the peer representation.

### Shared honcho_client.py

Deep module. Hides: SDK initialization, config file loading, pagination, `created_at` filtering for `list_conclusions`.

Interface (module-level functions, consistent with existing pattern):
```python
conclude(text: str, session_id: str = "mahler-memory") -> None
list_conclusions(since_days: int = 30) -> list[Conclusion]
query_conclusions(query: str, top_k: int = 10) -> list[Conclusion]
```

Config loaded internally from `~/.hermes/honcho.json` + `HONCHO_API_KEY` env var. Raises `RuntimeError` on API failure or missing config.

### project-synthesis

Cron: Sunday 18:00 UTC. Script: `synthesize.py --run`.

Reads D1 `project_log` for last 7 days. If empty, posts "No project activity this week" to Discord and exits 0. Otherwise, calls LLM with all wins/blockers formatted as a list, producing one paragraph (~150 tokens) covering: which project(s) had the most activity, overall trajectory (shipping/stuck/grinding), and recurring friction. Calls `conclude(paragraph, session_id="project-synthesis")`. Posts confirmation to Discord.

### memory-kaizen

Cron: Sunday 19:00 UTC (one hour after project-synthesis, ensuring this week's synthesis is in the pool). Script: `kaizen.py --run`.

Calls `list_conclusions(since_days=30)`. If fewer than 5 conclusions, posts "Insufficient data for memory kaizen" and exits 0. Otherwise, calls LLM with all conclusion `.content` values formatted as a numbered list, identifying 2–4 patterns that appear across multiple entries. Each pattern must start with "Jai " and be prefixed "PATTERN: ". Calls `conclude(pattern, session_id="memory-kaizen")` for each. Posts summary to Discord.

### Removal of _run_combined_analysis

`kaizen-reflection/reflect.py:_run_combined_analysis` is removed. Its coding-session synthesis role moves to `project-synthesis`. Its reflection-pattern role was already handled by `reflection-journal`. `_run()` no longer calls it. `HONCHO_API_KEY` is no longer referenced in kaizen-reflection.

---

## Modules

**`config/shared/honcho_client.py`**
- Interface: `conclude(text, session_id)`, `list_conclusions(since_days)`, `query_conclusions(query, top_k)`
- Hides: Honcho SDK client construction, `honcho.json` parsing, auto-pagination of `list()`, `created_at` date filtering, `RuntimeError` mapping from SDK exceptions
- Tested through: `config/shared/tests/test_honcho_client.py` — patches `Honcho` class, verifies SDK method chain is called with correct args; verifies `list_conclusions` filters by date; verifies `RuntimeError` on SDK failure

**`config/skills/project-synthesis/scripts/synthesize.py`**
- Interface: `main(["--run"])`
- Hides: nothing substantial — shallow orchestrator
- Tested through: `test_synthesize.py` — mocks `D1Client` and `honcho_client` module

**`config/skills/memory-kaizen/scripts/kaizen.py`**
- Interface: `main(["--run"])`
- Hides: nothing substantial — shallow orchestrator
- Tested through: `test_kaizen.py` — mocks `honcho_client` module

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `config/shared/honcho_client.py` | New shared SDK-backed client | New |
| `config/shared/tests/test_honcho_client.py` | Tests for shared client | New |
| `config/skills/project-synthesis/SKILL.md` | Skill declaration | New |
| `config/skills/project-synthesis/scripts/synthesize.py` | Main script | New |
| `config/skills/project-synthesis/tests/test_synthesize.py` | Behavior tests | New |
| `config/skills/memory-kaizen/SKILL.md` | Skill declaration | New |
| `config/skills/memory-kaizen/scripts/kaizen.py` | Main script | New |
| `config/skills/memory-kaizen/tests/test_kaizen.py` | Behavior tests | New |
| `config/skills/reflection-journal/scripts/honcho_client.py` | Delete old v1 client | Delete |
| `config/skills/reflection-journal/tests/test_honcho_client.py` | Delete old v1 test | Delete |
| `config/skills/reflection-journal/scripts/journal.py` | Update import + call site | Modify |
| `config/skills/reflection-journal/tests/test_journal.py` | Add shared path to sys.path | Modify |
| `config/skills/kaizen-reflection/scripts/honcho_client.py` | Delete old v1 client | Delete |
| `config/skills/kaizen-reflection/tests/test_honcho_client.py` | Delete old v1 test | Delete |
| `config/skills/kaizen-reflection/scripts/reflect.py` | Remove `_run_combined_analysis` | Modify |
| `config/skills/email-triage/scripts/honcho_client.py` | Delete old v1 client | Delete |
| `config/skills/email-triage/tests/test_honcho_client.py` | Delete old v1 test | Delete |
| `config/skills/email-triage/scripts/triage.py` | Update import + call site | Modify |
| `config/skills/email-triage/tests/test_triage_integration.py` | Add shared path to sys.path | Modify |
| `assistant/Dockerfile` | Add `pip install honcho-ai`, two COPY lines | Modify |

---

## Open Questions

- Q: Does `honcho-ai` SDK raise a specific exception type on API failure (e.g., `honcho.APIError`), or just a generic one?  Default: catch `Exception`, wrap in `RuntimeError` with message. Update to specific type once confirmed during Task 1 implementation.
- Q: Does `conclusions.list()` return `created_at` as a `datetime` object or ISO string?  Default: handle both — try `datetime` attribute first, fall back to `fromisoformat()`.
