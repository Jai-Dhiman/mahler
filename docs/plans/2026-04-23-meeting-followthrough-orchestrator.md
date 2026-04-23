# Meeting Follow-Through Orchestrator Implementation Plan

> **For the build agent:** Every task writes to `scripts/orchestrate.py` and/or `tests/test_orchestrate.py`. Tasks 1–12 must run sequentially — each extends the same two files. Tasks 13–15 depend on 1–12 being merged. Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the LLM-agent-driven meeting-followthrough skill with a deterministic Python orchestrator so that empty cron ticks cost near-zero and Step 8 (Discord post) can never be silently skipped.
**Spec:** docs/specs/2026-04-23-meeting-followthrough-orchestrator-design.md
**Style:** Follow CLAUDE.md (assistant/) + project conventions. Python 3.12+, stdlib only (no external deps). Tests use `unittest` + `MagicMock` matching `reflection-journal/tests/test_journal.py`.

## Task Groups

- **Group A (sequential, all touch `orchestrate.py` + `test_orchestrate.py`):** Tasks 1–12.
- **Group B (depends on A):** Task 13 (SKILL.md rewrite). Touches only SKILL.md.
- **Group C (depends on B):** Task 14 (deploy + cron edit + prod smoke test). Runtime only.
- **Group D (depends on C):** Task 15 (delete `poll.py`). Touches only `scripts/poll.py`.

---

### Task 1: Skeleton — empty queue prints NO_WORK
**Group:** A

**Behavior being verified:** When `fetch_pending()` returns no rows, `main()` prints `NO_WORK` to stdout and returns 0.
**Interface under test:** `main(argv, *, d1_client, runner, llm_caller, discord_poster) -> int`.

**Files:**
- Create: `assistant/config/skills/meeting-followthrough/scripts/orchestrate.py`
- Create: `assistant/config/skills/meeting-followthrough/tests/__init__.py`
- Create: `assistant/config/skills/meeting-followthrough/tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/meeting-followthrough/tests/test_orchestrate.py
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestEmptyQueue(unittest.TestCase):
    def test_empty_queue_prints_no_work_and_returns_zero(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = []
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = orchestrate.main(
                argv=[],
                d1_client=d1,
                runner=MagicMock(),
                llm_caller=MagicMock(),
                discord_poster=MagicMock(),
            )
        self.assertEqual(rc, 0)
        self.assertIn("NO_WORK", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
```

Create `tests/__init__.py` as an empty file.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrate'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/meeting-followthrough/scripts/orchestrate.py
"""Meeting follow-through orchestrator. Invoked by cron every 15 min."""
from __future__ import annotations
import os
from pathlib import Path


_DEFAULT_MODEL = "openai/gpt-5-nano"


def _load_hermes_env() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    if not hermes_env.exists():
        return
    with open(hermes_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def main(argv, *, d1_client, runner, llm_caller, discord_poster) -> int:
    rows = d1_client.fetch_pending()
    if not rows:
        print("NO_WORK")
        return 0
    return 0
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 1 test ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/__init__.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): orchestrator skeleton with empty-queue path"
```

---

### Task 2: Trivial meeting — process_meeting returns summary, posts Discord, marks done
**Group:** A (depends on Task 1)

**Behavior being verified:** Given one pending row with no attendees, `process_meeting` returns a formatted summary line, calls `discord_poster` with the same content, and calls `d1_client.mark_done(recording_id)`.
**Interface under test:** `process_meeting(row, *, runner, llm_caller, discord_poster, d1_client) -> str`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrate.py`:

```python
class TestTrivialMeeting(unittest.TestCase):
    def test_trivial_meeting_posts_summary_and_marks_done(self):
        import orchestrate
        row = {
            "recording_id": 42,
            "title": "Test call",
            "attendees": "[]",
            "summary": "Fathom intro test.",
        }
        d1 = MagicMock()
        runner = MagicMock()
        runner.return_value = MagicMock(returncode=0, stdout="", stderr="")
        llm = MagicMock(return_value="no action items")
        poster = MagicMock()
        result = orchestrate.process_meeting(
            row, runner=runner, llm_caller=llm, discord_poster=poster, d1_client=d1
        )
        expected = (
            "Post-meeting: Test call\n"
            "Action items created:\n"
            "  None\n"
            "CRM updated: No CRM matches"
        )
        self.assertEqual(result, expected)
        poster.assert_called_once_with(expected)
        d1.mark_done.assert_called_once_with(42)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AttributeError: module 'orchestrate' has no attribute 'process_meeting'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `scripts/orchestrate.py`:

```python
import json


def process_meeting(row, *, runner, llm_caller, discord_poster, d1_client) -> str:
    title = row["title"]
    action_lines = "  None"
    crm_line = "CRM updated: No CRM matches"
    summary = (
        f"Post-meeting: {title}\n"
        f"Action items created:\n"
        f"{action_lines}\n"
        f"{crm_line}"
    )
    discord_poster(summary)
    d1_client.mark_done(row["recording_id"])
    return summary
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 2 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): process trivial meeting end-to-end"
```

---

### Task 3: generate_action_items returns empty list for "no action items" LLM response
**Group:** A (depends on Task 2)

**Behavior being verified:** Given an LLM response of `"no action items"`, `generate_action_items` returns `[]`.
**Interface under test:** `generate_action_items(summary, attendees, crm_context, open_tasks, llm_caller) -> list[dict]`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrate.py`:

```python
class TestGenerateActionItems(unittest.TestCase):
    def test_no_action_items_llm_response_returns_empty_list(self):
        import orchestrate
        llm = MagicMock(return_value="no action items")
        result = orchestrate.generate_action_items(
            summary="a meeting happened",
            attendees=[],
            crm_context={},
            open_tasks=[],
            llm_caller=llm,
        )
        self.assertEqual(result, [])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AttributeError: module 'orchestrate' has no attribute 'generate_action_items'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `scripts/orchestrate.py`:

```python
def generate_action_items(summary, attendees, crm_context, open_tasks, llm_caller) -> list[dict]:
    prompt = _build_prompt(summary, attendees, crm_context, open_tasks)
    raw = llm_caller(prompt)
    return _parse_action_items(raw)


def _build_prompt(summary, attendees, crm_context, open_tasks) -> str:
    return f"summary: {summary}"


def _parse_action_items(raw: str) -> list[dict]:
    items: list[dict] = []
    for line in raw.splitlines():
        if line.startswith("TASK:"):
            items.append(_parse_task_line(line))
    return items


def _parse_task_line(line: str) -> dict:
    body = line[len("TASK:"):].strip()
    title_part, _, priority_part = body.partition("| PRIORITY:")
    title = title_part.strip()
    priority = priority_part.strip() or "Medium"
    attendee = None
    if title.startswith("["):
        end = title.find("]")
        if end > 0:
            attendee = title[1:end]
    return {"title": title, "priority": priority, "attendee": attendee}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 3 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): generate_action_items parses empty LLM response"
```

---

### Task 4: generate_action_items parses a structured action line
**Group:** A (depends on Task 3)

**Behavior being verified:** Given an LLM response `"TASK: [Alice] Send Q2 memo | PRIORITY: High"`, `generate_action_items` returns `[{"title": "[Alice] Send Q2 memo", "priority": "High", "attendee": "Alice"}]`.
**Interface under test:** `generate_action_items`.

**Files:**
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append to `TestGenerateActionItems`:

```python
    def test_single_task_line_with_attendee_and_priority(self):
        import orchestrate
        llm = MagicMock(return_value="TASK: [Alice] Send Q2 memo | PRIORITY: High")
        result = orchestrate.generate_action_items(
            summary="meeting",
            attendees=[],
            crm_context={},
            open_tasks=[],
            llm_caller=llm,
        )
        self.assertEqual(
            result,
            [{"title": "[Alice] Send Q2 memo", "priority": "High", "attendee": "Alice"}],
        )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS already — Task 3's parser handles this format.

If it PASSES without a code change: no implementation is needed here. This is a regression test that pins parser behavior. Proceed to commit as a test-only task (see Step 3).

- [ ] **Step 3: No implementation change needed**

The parser from Task 3 already supports this format. This task locks the contract via a regression test so future changes to `_parse_task_line` cannot silently break it.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 4 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "test(meeting-followthrough): pin generate_action_items parse contract for structured line"
```

---

### Task 5: LLM prompt includes open_tasks so the model can dedupe
**Group:** A (depends on Task 4)

**Behavior being verified:** When `generate_action_items` is called with `open_tasks=["Follow up with Alice"]`, the prompt passed to `llm_caller` contains that title.
**Interface under test:** `generate_action_items`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append to `TestGenerateActionItems`:

```python
    def test_open_tasks_flow_into_prompt_for_dedup(self):
        import orchestrate
        captured = {}
        def llm(prompt):
            captured["prompt"] = prompt
            return "no action items"
        orchestrate.generate_action_items(
            summary="meeting",
            attendees=[],
            crm_context={},
            open_tasks=["Follow up with Alice", "Review Q2 deck"],
            llm_caller=llm,
        )
        self.assertIn("Follow up with Alice", captured["prompt"])
        self.assertIn("Review Q2 deck", captured["prompt"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AssertionError: 'Follow up with Alice' not found in 'summary: meeting'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `_build_prompt` in `scripts/orchestrate.py` with:

```python
def _build_prompt(summary, attendees, crm_context, open_tasks) -> str:
    attendees_block = "\n".join(
        f"- {a.get('name') or a.get('email') or 'unknown'}: {crm_context.get(a.get('email', ''), 'not in CRM')}"
        for a in attendees
    ) or "- none"
    open_tasks_block = "\n".join(f"- {t}" for t in open_tasks) or "- none"
    return (
        "You are generating post-meeting action items.\n\n"
        f"Meeting summary:\n{summary}\n\n"
        f"Attendees:\n{attendees_block}\n\n"
        f"Existing open tasks (do NOT duplicate these):\n{open_tasks_block}\n\n"
        "Output format: one action item per line, prefixed with 'TASK: '. "
        "Include priority as '| PRIORITY: High|Medium|Low'. "
        "Prefix the title with '[Attendee Name]' when the item relates to a specific attendee. "
        "If there are no action items, respond with exactly 'no action items'."
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 5 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): include open_tasks in LLM prompt for dedup"
```

---

### Task 6: CRM summarize output for external attendees flows into the LLM prompt
**Group:** A (depends on Task 5)

**Behavior being verified:** Given a meeting with attendee `{"name": "Alice", "email": "alice@ext.com"}`, `process_meeting` runs `contacts.py summarize --name "Alice"`, and the runner's stdout appears in the prompt passed to `llm_caller`.
**Interface under test:** `process_meeting`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrate.py`:

```python
class TestCrmFanout(unittest.TestCase):
    def test_crm_context_for_external_attendee_flows_to_llm(self):
        import orchestrate
        row = {
            "recording_id": 51,
            "title": "Sync",
            "attendees": '[{"name": "Alice", "email": "alice@ext.com", "is_external": true}]',
            "summary": "we talked",
        }
        def runner(argv, **_):
            if "summarize" in argv and "Alice" in argv:
                return MagicMock(returncode=0, stdout="Alice is a senior PM", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        captured_prompt = {}
        def llm(prompt):
            captured_prompt["p"] = prompt
            return "no action items"
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=MagicMock(),
            d1_client=MagicMock(),
        )
        self.assertIn("Alice is a senior PM", captured_prompt["p"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AssertionError: 'Alice is a senior PM' not found in ...`. The `process_meeting` from Task 2 never calls the runner or LLM.

- [ ] **Step 3: Implement the minimum to make the test pass**

Rewrite `process_meeting` in `scripts/orchestrate.py`:

```python
import subprocess


def _parse_attendees(raw: str) -> list[dict]:
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _fetch_crm_context(attendees: list[dict], runner) -> dict[str, str]:
    owner_email = os.environ.get("MAHLER_OWNER_EMAIL", "").lower()
    ctx: dict[str, str] = {}
    for a in attendees:
        email = (a.get("email") or "").lower()
        name = a.get("name")
        if not name or not email or email == owner_email:
            continue
        result = runner(
            ["python3", str(Path.home() / ".hermes" / "skills" / "relationship-manager" / "scripts" / "contacts.py"), "summarize", "--name", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            ctx[email] = result.stdout.strip()
    return ctx


def process_meeting(row, *, runner, llm_caller, discord_poster, d1_client) -> str:
    title = row["title"]
    attendees = _parse_attendees(row["attendees"])
    crm_context = _fetch_crm_context(attendees, runner)
    action_items = generate_action_items(
        summary=row["summary"],
        attendees=attendees,
        crm_context=crm_context,
        open_tasks=[],
        llm_caller=llm_caller,
    )
    if action_items:
        action_lines = "\n".join(f"  · {i['title']}" for i in action_items)
    else:
        action_lines = "  None"
    crm_line = "CRM updated: No CRM matches"
    summary = (
        f"Post-meeting: {title}\n"
        f"Action items created:\n"
        f"{action_lines}\n"
        f"{crm_line}"
    )
    discord_poster(summary)
    d1_client.mark_done(row["recording_id"])
    return summary
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 6 tests ok. The Task 2 trivial-meeting test still passes because empty attendees → empty crm_context → same stdout summary.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): fetch CRM context for external attendees"
```

---

### Task 7: Owner email attendee is excluded from CRM fan-out
**Group:** A (depends on Task 6)

**Behavior being verified:** When `MAHLER_OWNER_EMAIL=jai@mahler.local` and the meeting's only attendee has that email, the runner is never called for `summarize`, and the LLM prompt contains no CRM lines for the owner.
**Interface under test:** `process_meeting`.

**Files:**
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append to `TestCrmFanout`:

```python
    def test_owner_attendee_not_fetched_from_crm(self):
        import orchestrate
        row = {
            "recording_id": 52,
            "title": "Solo",
            "attendees": '[{"name": "Jai Dhiman", "email": "jai@mahler.local", "is_external": false}]',
            "summary": "solo prep",
        }
        calls: list[list[str]] = []
        def runner(argv, **_):
            calls.append(argv)
            return MagicMock(returncode=0, stdout="", stderr="")
        captured = {}
        def llm(prompt):
            captured["p"] = prompt
            return "no action items"
        with patch.dict("os.environ", {"MAHLER_OWNER_EMAIL": "jai@mahler.local"}):
            orchestrate.process_meeting(
                row,
                runner=runner,
                llm_caller=llm,
                discord_poster=MagicMock(),
                d1_client=MagicMock(),
            )
        summarize_calls = [c for c in calls if "summarize" in c]
        self.assertEqual(summarize_calls, [])
        self.assertNotIn("jai@mahler.local", captured["p"])
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — Task 6's `_fetch_crm_context` already compares against `MAHLER_OWNER_EMAIL`. This test pins that behavior so a future refactor cannot silently drop the owner-skip.

- [ ] **Step 3: No implementation change needed**

Regression test only. If Step 2 FAILED, fix `_fetch_crm_context` so it compares lowercased email to `os.environ.get("MAHLER_OWNER_EMAIL", "").lower()`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 7 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "test(meeting-followthrough): owner-email attendee excluded from CRM fan-out"
```

---

### Task 8: Open tasks from tasks.py list flow into the LLM prompt
**Group:** A (depends on Task 7)

**Behavior being verified:** `process_meeting` invokes `tasks.py list --status "Not started"` and the parsed titles appear in the LLM prompt.
**Interface under test:** `process_meeting`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append new test class to `tests/test_orchestrate.py`:

```python
class TestOpenTasksFlow(unittest.TestCase):
    def test_open_tasks_from_tasks_list_appear_in_llm_prompt(self):
        import orchestrate
        row = {
            "recording_id": 60,
            "title": "Planning",
            "attendees": "[]",
            "summary": "planning chat",
        }
        def runner(argv, **_):
            if "list" in argv and "--status" in argv:
                return MagicMock(returncode=0, stdout="Follow up with Alice\nReview Q2 deck", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        captured = {}
        def llm(prompt):
            captured["p"] = prompt
            return "no action items"
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=MagicMock(),
            d1_client=MagicMock(),
        )
        self.assertIn("Follow up with Alice", captured["p"])
        self.assertIn("Review Q2 deck", captured["p"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — open_tasks passed to generate_action_items is hard-coded `[]` in `process_meeting`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add a helper and wire it in. Update `scripts/orchestrate.py`:

```python
def _fetch_open_tasks(runner) -> list[str]:
    result = runner(
        ["python3", str(Path.home() / ".hermes" / "skills" / "notion-tasks" / "scripts" / "tasks.py"), "list", "--status", "Not started"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
```

In `process_meeting`, replace `open_tasks=[]` with:

```python
        open_tasks=_fetch_open_tasks(runner),
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 8 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): feed existing open tasks into dedup prompt"
```

---

### Task 9: Action items become tasks.py create invocations
**Group:** A (depends on Task 8)

**Behavior being verified:** When the LLM returns two TASK lines, `process_meeting` invokes `tasks.py create --title <t> --priority <p>` for each.
**Interface under test:** `process_meeting`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append new test class:

```python
class TestTasksCreate(unittest.TestCase):
    def test_each_action_item_becomes_tasks_create_call(self):
        import orchestrate
        row = {
            "recording_id": 70,
            "title": "1:1",
            "attendees": "[]",
            "summary": "1:1 summary",
        }
        calls: list[list[str]] = []
        def runner(argv, **_):
            calls.append(argv)
            return MagicMock(returncode=0, stdout="", stderr="")
        llm = MagicMock(return_value=(
            "TASK: [Alice] Send Q2 memo | PRIORITY: High\n"
            "TASK: Follow up on Series A | PRIORITY: Medium"
        ))
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=MagicMock(),
            d1_client=MagicMock(),
        )
        create_calls = [c for c in calls if "create" in c]
        self.assertEqual(len(create_calls), 2)
        titles = {c[c.index("--title") + 1] for c in create_calls}
        self.assertEqual(titles, {"[Alice] Send Q2 memo", "Follow up on Series A"})
        priorities = {c[c.index("--priority") + 1] for c in create_calls}
        self.assertEqual(priorities, {"High", "Medium"})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AssertionError: 0 != 2`. `process_meeting` doesn't invoke tasks.py create.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add a helper and call it:

```python
def _create_tasks(action_items: list[dict], runner) -> list[str]:
    created: list[str] = []
    for item in action_items:
        argv = [
            "python3",
            str(Path.home() / ".hermes" / "skills" / "notion-tasks" / "scripts" / "tasks.py"),
            "create",
            "--title", item["title"],
            "--priority", item["priority"],
        ]
        result = runner(argv, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            created.append(item["title"])
    return created
```

In `process_meeting`, after the `generate_action_items` call add:

```python
    created_titles = _create_tasks(action_items, runner)
```

And use `created_titles` instead of `action_items` when rendering `action_lines`:

```python
    if created_titles:
        action_lines = "\n".join(f"  · {t}" for t in created_titles)
    else:
        action_lines = "  None"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 9 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): create Notion tasks for each action item"
```

---

### Task 10: CRM talked-to runs only for attendees that had a successful CRM summarize
**Group:** A (depends on Task 9)

**Behavior being verified:** Given two external attendees where `contacts.py summarize` succeeds for Alice and fails (returncode != 0) for Bob, `process_meeting` invokes `contacts.py talked-to --name Alice` but not `talked-to --name Bob`, and the Discord summary's CRM line lists only Alice.
**Interface under test:** `process_meeting`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append new test class:

```python
class TestCrmTalkedTo(unittest.TestCase):
    def test_talked_to_only_for_attendees_found_in_crm(self):
        import orchestrate
        row = {
            "recording_id": 80,
            "title": "Quarterly",
            "attendees": (
                '[{"name": "Alice", "email": "alice@ext.com", "is_external": true},'
                ' {"name": "Bob", "email": "bob@ext.com", "is_external": true}]'
            ),
            "summary": "quarterly review",
        }
        calls: list[list[str]] = []
        def runner(argv, **_):
            calls.append(argv)
            if "summarize" in argv and "Alice" in argv:
                return MagicMock(returncode=0, stdout="Alice context", stderr="")
            if "summarize" in argv and "Bob" in argv:
                return MagicMock(returncode=1, stdout="", stderr="not in CRM")
            return MagicMock(returncode=0, stdout="", stderr="")
        captured_post = {}
        def poster(content):
            captured_post["c"] = content
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=MagicMock(return_value="no action items"),
            discord_poster=poster,
            d1_client=MagicMock(),
        )
        talked_to = [c for c in calls if "talked-to" in c]
        self.assertEqual(len(talked_to), 1)
        self.assertIn("Alice", talked_to[0])
        self.assertIn("CRM updated: Alice", captured_post["c"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AssertionError: 0 != 1`. No `talked-to` call is made yet; CRM line is hard-coded "No CRM matches".

- [ ] **Step 3: Implement the minimum to make the test pass**

Add a helper and wire it in:

```python
def _update_crm_last_contact(crm_context: dict[str, str], attendees: list[dict], runner) -> list[str]:
    updated: list[str] = []
    for a in attendees:
        email = (a.get("email") or "").lower()
        name = a.get("name")
        if not name or email not in crm_context:
            continue
        result = runner(
            ["python3", str(Path.home() / ".hermes" / "skills" / "relationship-manager" / "scripts" / "contacts.py"), "talked-to", "--name", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            updated.append(name)
    return updated
```

In `process_meeting`, after `_create_tasks(...)`:

```python
    updated_contacts = _update_crm_last_contact(crm_context, attendees, runner)
    crm_line = (
        f"CRM updated: {', '.join(updated_contacts)}"
        if updated_contacts
        else "CRM updated: No CRM matches"
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 10 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): update CRM last_contact for matched attendees"
```

---

### Task 11: main() processes multiple pending meetings in order
**Group:** A (depends on Task 10)

**Behavior being verified:** When `fetch_pending` returns two rows, `main()` calls `mark_done` twice (once per `recording_id`) and `discord_poster` twice (once per meeting summary).
**Interface under test:** `main`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append new test class:

```python
class TestMainMultipleMeetings(unittest.TestCase):
    def test_each_pending_row_is_processed_independently(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = [
            {"recording_id": 101, "title": "Sync A", "attendees": "[]", "summary": "a"},
            {"recording_id": 102, "title": "Sync B", "attendees": "[]", "summary": "b"},
        ]
        posted: list[str] = []
        def poster(c):
            posted.append(c)
        orchestrate.main(
            argv=[],
            d1_client=d1,
            runner=lambda argv, **_: MagicMock(returncode=0, stdout="", stderr=""),
            llm_caller=MagicMock(return_value="no action items"),
            discord_poster=poster,
        )
        marked = [call.args[0] for call in d1.mark_done.call_args_list]
        self.assertEqual(marked, [101, 102])
        self.assertEqual(len(posted), 2)
        self.assertIn("Sync A", posted[0])
        self.assertIn("Sync B", posted[1])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `main()` from Task 1 has no loop; returns 0 without processing rows.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `main` in `scripts/orchestrate.py`:

```python
def main(argv, *, d1_client, runner, llm_caller, discord_poster) -> int:
    rows = d1_client.fetch_pending()
    if not rows:
        print("NO_WORK")
        return 0
    for row in rows:
        process_meeting(
            row,
            runner=runner,
            llm_caller=llm_caller,
            discord_poster=discord_poster,
            d1_client=d1_client,
        )
    return 0
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 11 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): process all pending meetings in one tick"
```

---

### Task 12: main() surfaces LLM exceptions to Discord and exits non-zero
**Group:** A (depends on Task 11)

**Behavior being verified:** If `llm_caller` raises `RuntimeError("OpenRouter 500")` for a meeting, `main()` posts an error message to Discord naming the meeting title and returns 1. The meeting is NOT marked done (so the next tick retries).
**Interface under test:** `main`.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append new test class:

```python
class TestMainErrorHandling(unittest.TestCase):
    def test_llm_exception_surfaces_to_discord_and_returns_nonzero(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = [
            {"recording_id": 201, "title": "Flaky", "attendees": "[]", "summary": "x"},
        ]
        def bad_llm(_prompt):
            raise RuntimeError("OpenRouter 500")
        posted: list[str] = []
        def poster(c):
            posted.append(c)
        rc = orchestrate.main(
            argv=[],
            d1_client=d1,
            runner=lambda argv, **_: MagicMock(returncode=0, stdout="", stderr=""),
            llm_caller=bad_llm,
            discord_poster=poster,
        )
        self.assertEqual(rc, 1)
        self.assertEqual(d1.mark_done.call_count, 0)
        self.assertTrue(any("Flaky" in p and "OpenRouter 500" in p for p in posted))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — the LLM exception propagates out of `main`, test aborts with `RuntimeError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Wrap the loop body in `main` in `scripts/orchestrate.py`:

```python
def main(argv, *, d1_client, runner, llm_caller, discord_poster) -> int:
    rows = d1_client.fetch_pending()
    if not rows:
        print("NO_WORK")
        return 0
    had_error = False
    for row in rows:
        try:
            process_meeting(
                row,
                runner=runner,
                llm_caller=llm_caller,
                discord_poster=discord_poster,
                d1_client=d1_client,
            )
        except Exception as exc:
            had_error = True
            err = f"Meeting processing FAILED for {row.get('title', '?')}: {exc}"
            try:
                discord_poster(err)
            except Exception:
                pass
            print(err)
    return 1 if had_error else 0
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 12 tests ok. All prior tests still pass.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): surface per-meeting errors to Discord and retry next tick"
```

---

### Task 12.5 — Production entry point: `__main__` wiring with real deps
**Group:** A (depends on Task 12)

**Behavior being verified:** Running `python3 orchestrate.py` from the CLI constructs real `D1Client`, a real subprocess runner, a real `_call_llm`, and a real `post_discord` poster, then invokes `main()`. With an empty D1 queue it exits 0 and prints `NO_WORK`.
**Interface under test:** `orchestrate.py` executed as a script.

**Files:**
- Modify: `scripts/orchestrate.py`
- Modify: `tests/test_orchestrate.py`

- [ ] **Step 1: Write the failing test**

Append new test class:

```python
class TestCliEntryPoint(unittest.TestCase):
    def test_cli_entry_point_with_empty_queue_returns_zero(self):
        import orchestrate
        fake_d1 = MagicMock()
        fake_d1.fetch_pending.return_value = []
        captured = io.StringIO()
        env = {
            "CF_ACCOUNT_ID": "acct",
            "CF_D1_DATABASE_ID": "db",
            "CF_API_TOKEN": "tok",
            "OPENROUTER_API_KEY": "k",
            "DISCORD_TRIAGE_WEBHOOK": "https://discord.com/api/webhooks/x/y",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch("orchestrate.D1Client", return_value=fake_d1),
            patch("sys.stdout", captured),
        ):
            rc = orchestrate.cli_main()
        self.assertEqual(rc, 0)
        self.assertIn("NO_WORK", captured.getvalue())
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: FAIL — `AttributeError: module 'orchestrate' has no attribute 'cli_main'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `scripts/orchestrate.py`:

```python
import sys
import ssl
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).parent))
from d1_client import D1Client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_openrouter(prompt: str) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    return data["choices"][0]["message"]["content"]


def _post_discord(content: str) -> None:
    post_discord_py = Path(__file__).parent / "post_discord.py"
    result = subprocess.run(
        ["python3", str(post_discord_py)],
        input=content,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"post_discord.py failed: {result.stderr}")


def cli_main() -> int:
    _load_hermes_env()
    d1 = D1Client(
        account_id=os.environ["CF_ACCOUNT_ID"],
        database_id=os.environ["CF_D1_DATABASE_ID"],
        api_token=os.environ["CF_API_TOKEN"],
    )
    d1.ensure_queue_table()
    return main(
        argv=sys.argv[1:],
        d1_client=d1,
        runner=subprocess.run,
        llm_caller=_call_openrouter,
        discord_poster=_post_discord,
    )


if __name__ == "__main__":
    sys.exit(cli_main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-followthrough && python3 -m unittest tests.test_orchestrate -v
```
Expected: PASS — 13 tests ok.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/scripts/orchestrate.py config/skills/meeting-followthrough/tests/test_orchestrate.py && git commit -m "feat(meeting-followthrough): wire cli_main with real D1/OpenRouter/Discord deps"
```

---

### Task 13: SKILL.md rewrite — thin pass-through to orchestrate.py
**Group:** B (depends on Group A)

**Behavior being verified:** `SKILL.md` body invokes `orchestrate.py` as a single explicit bash command and instructs the LLM to report its stdout and any non-zero exit to Discord.
**Interface under test:** grep-level contract on the file.

**Files:**
- Modify: `assistant/config/skills/meeting-followthrough/SKILL.md`

- [ ] **Step 1: Write the verification check (no unit test — doc change)**

```bash
grep -c "orchestrate.py" assistant/config/skills/meeting-followthrough/SKILL.md
grep -c "scripts/poll.py" assistant/config/skills/meeting-followthrough/SKILL.md
grep -c "Step 1 — Check for pending meetings" assistant/config/skills/meeting-followthrough/SKILL.md
```
Expected after the rewrite: first command returns ≥ 1, second returns 0, third returns 0.

- [ ] **Step 2: Verify current state fails the contract**

The above greps currently return: 0, 1, 1. Confirm.

- [ ] **Step 3: Rewrite SKILL.md body**

Replace everything below the frontmatter (line 17 onward) with:

```markdown
# Meeting Follow-Through

Closes the loop after any recorded meeting. Triggered by cron every 15 minutes.
Everything is done by `orchestrate.py` — this skill just invokes it and reports the result.

## Procedure

Run the orchestrator:

```bash
python3 ~/.hermes/skills/meeting-followthrough/scripts/orchestrate.py
```

Report whatever it prints to stdout verbatim. If the exit code is non-zero, the orchestrator will have already posted a per-meeting error message to the Discord triage channel; repeat the error in your reply so the user sees it in the agent log too.

If stdout is `NO_WORK`, stop — no pending meetings this tick.

## Failure modes

- OpenRouter returns 5xx while generating action items → orchestrator catches, posts error to Discord for the affected meeting, leaves the meeting UN-marked (next tick retries).
- `contacts.py summarize` returns non-zero → attendee is treated as "not in CRM" and skipped for CRM fan-out.
- `tasks.py create` returns non-zero for one action item → that item is dropped from the Discord summary; other items proceed.
- `post_discord.py` fails → orchestrator exits non-zero; SessionStop hook surfaces the error.
```

- [ ] **Step 4: Re-run the verification greps**

```bash
grep -c "orchestrate.py" assistant/config/skills/meeting-followthrough/SKILL.md
grep -c "scripts/poll.py" assistant/config/skills/meeting-followthrough/SKILL.md
grep -c "Step 1 — Check for pending meetings" assistant/config/skills/meeting-followthrough/SKILL.md
```
Expected: ≥ 1, 0, 0.

- [ ] **Step 5: Commit**

```bash
cd assistant && git add config/skills/meeting-followthrough/SKILL.md && git commit -m "docs(meeting-followthrough): rewrite SKILL body as thin orchestrate.py pass-through"
```

---

### Task 14: Deploy, update cron cadence, smoke-test in production
**Group:** C (depends on Task 13)

**Behavior being verified:** After `flyctl deploy` and `hermes cron edit`, a real Fathom test call results in a Discord triage message within 15 minutes.
**Interface under test:** end-to-end production path.

**Files:**
- No git-tracked file changes in this task.

- [ ] **Step 1: Deploy**

```bash
cd assistant && flyctl deploy --remote-only
```
Expected: build succeeds, 2/2 machines reach good state.

- [ ] **Step 2: Verify the new script is on the machine**

```bash
flyctl ssh console -a mahler-agent --user hermes -C "ls /home/hermes/.hermes/skills/meeting-followthrough/scripts/"
```
Expected output contains: `orchestrate.py`, `post_discord.py`, `poll.py`, `d1_client.py`.

- [ ] **Step 3: Update cron cadence from */5 to */15**

Find the job id:

```bash
flyctl ssh console -a mahler-agent --user hermes -C "hermes cron list" | grep -B1 "meeting-followthrough" | head
```

Edit it (replace `<JOB_ID>` with the id from above):

```bash
flyctl ssh console -a mahler-agent --user hermes -C "hermes cron edit <JOB_ID> --schedule '*/15 * * * *'"
```

Re-list to confirm:

```bash
flyctl ssh console -a mahler-agent --user hermes -C "hermes cron list" | grep -A4 "meeting-followthrough"
```
Expected: `Schedule:  */15 * * * *`.

- [ ] **Step 4: Trigger a Fathom test call**

From the Fathom UI, record a short test call. Within ~30 seconds the `fathom-webhook` Worker logs should show `enqueued recording_id=…`. Within the next 15 min (next `*/15` tick), the Discord triage channel should receive a `Post-meeting: …` message.

Tail while waiting:

```bash
flyctl ssh console -a mahler-agent --user hermes -C "hermes logs -f"
```
Look for: `cron.scheduler: Running job 'meeting-followthrough'` and absence of any `process_meeting` exception.

- [ ] **Step 5: Confirm D1 row is marked processed**

```bash
# From local; CF_* vars loaded from ~/.mahler.env
source ~/.mahler.env && curl -sS -X POST "https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/d1/database/${CF_D1_DATABASE_ID}/query" -H "Authorization: Bearer ${CF_API_TOKEN}" -H "Content-Type: application/json" --data '{"sql":"SELECT recording_id, title, created_at, processed_at FROM fathom_meeting_queue ORDER BY created_at DESC LIMIT 3;"}'
```
Expected: most recent row has a non-null `processed_at`.

This task has no git commit — it is runtime verification only. Do not proceed to Task 15 unless Steps 4 and 5 both pass.

---

### Task 15: Delete poll.py
**Group:** D (depends on Task 14)

**Behavior being verified:** `poll.py` is removed; no surviving file in `assistant/` imports or invokes it.
**Interface under test:** grep-level contract.

**Files:**
- Delete: `assistant/config/skills/meeting-followthrough/scripts/poll.py`

- [ ] **Step 1: Write the verification check**

```bash
grep -rn "poll.py\|from poll\|import poll" assistant/config/skills/meeting-followthrough/ | grep -v ".worktrees"
```
Expected after deletion: no matches.

- [ ] **Step 2: Verify no other skill references it**

```bash
grep -rn "meeting-followthrough/scripts/poll" assistant/ | grep -v ".worktrees" | grep -v node_modules
```
Expected: no matches.

- [ ] **Step 3: Delete the file**

```bash
git rm assistant/config/skills/meeting-followthrough/scripts/poll.py
```

- [ ] **Step 4: Re-run the grep to confirm nothing references the deleted path**

```bash
grep -rn "poll.py\|from poll\|import poll" assistant/config/skills/meeting-followthrough/ | grep -v ".worktrees"
```
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
cd assistant && git commit -m "chore(meeting-followthrough): remove poll.py now that orchestrate.py is live"
```
