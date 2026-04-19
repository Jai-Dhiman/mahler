# E4b Project Awareness Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Mahler surfaces recent development wins and blockers from Claude Code sessions in every conversation.
**Spec:** docs/specs/2026-04-19-e4b-project-awareness-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

```
Group A (sequential — all touch d1_client.py):
  Task 1 → Task 2 → Task 3

Group B (parallel — different files, depends on Group A):
  Task 4 (project_log.py), Task 5 (plugin.py)

Group C (parallel between files, sequential within each file, depends on Group B):
  Task 6 (project_log.py — same file as Task 4)
  Task 7 (plugin.py — same file as Task 5)

Group D (parallel between files, sequential within each file, depends on Group C):
  Task 8 (project_log.py — same file as Task 6)
  Task 9 (plugin.py — same file as Task 7)

Group E (parallel — different files, depends on Group D):
  Task 10 (settings.json), Task 11 (ship SKILL.md)
```

---

### Task 1: D1Client — add project_log table to ensure_tables()
**Group:** A (sequential, first)

**Behavior being verified:** `ensure_tables()` creates the `project_log` table with the correct schema.
**Interface under test:** `D1Client.ensure_tables()` — observable via the SQL sent to the D1 API.

**Files:**
- Modify: `assistant/config/skills/email-triage/scripts/d1_client.py`
- Modify: `assistant/config/skills/email-triage/tests/test_d1_client.py`

- [ ] **Step 1: Write the failing test**

Add a new test class to `assistant/config/skills/email-triage/tests/test_d1_client.py`. Also update the existing `TestD1ClientEnsureTables.test_ensure_tables_executes_create_table_statements` assertion from `len(calls) == 4` to `len(calls) == 5`.

```python
class TestD1ClientProjectLogTable(unittest.TestCase):

    def test_ensure_tables_creates_project_log_table(self):
        calls = []

        def capture_open(req):
            body = json.loads(req.data.decode("utf-8"))
            calls.append(body["sql"])
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.ensure_tables()

        self.assertEqual(len(calls), 5)
        self.assertIn("CREATE TABLE IF NOT EXISTS project_log", calls[4])
        self.assertIn("entry_type", calls[4])
        self.assertIn("summary", calls[4])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/test_d1_client.py::TestD1ClientProjectLogTable -v
```

Expected: FAIL — `AssertionError: 4 != 5` (only 4 CREATE TABLE calls are made currently). Also the existing `test_ensure_tables_executes_create_table_statements` will fail with `AssertionError: 4 != 5` after updating its assertion.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the following `self.query(...)` call at the end of the `ensure_tables()` method in `assistant/config/skills/email-triage/scripts/d1_client.py`:

```python
        self.query(
            """CREATE TABLE IF NOT EXISTS project_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    entry_type TEXT NOT NULL CHECK(entry_type IN ('win', 'blocker')),
    summary TEXT NOT NULL,
    git_ref TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)""",
            [],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/test_d1_client.py -v
```

Expected: PASS (all D1Client tests including the updated count assertion and the new class)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/email-triage/scripts/d1_client.py \
        assistant/config/skills/email-triage/tests/test_d1_client.py \
  && git commit -m "feat(e4b): add project_log table to D1Client ensure_tables"
```

---

### Task 2: D1Client — insert_project_log() method
**Group:** A (sequential, after Task 1)

**Behavior being verified:** `insert_project_log()` sends a parameterized INSERT to D1 with the correct values.
**Interface under test:** `D1Client.insert_project_log(project, entry_type, summary, git_ref)`

**Files:**
- Modify: `assistant/config/skills/email-triage/scripts/d1_client.py`
- Modify: `assistant/config/skills/email-triage/tests/test_d1_client.py`

- [ ] **Step 1: Write the failing test**

Add to `assistant/config/skills/email-triage/tests/test_d1_client.py`:

```python
class TestD1ClientInsertProjectLog(unittest.TestCase):

    def test_insert_project_log_sends_correct_sql_and_params(self):
        captured = []

        def capture_open(req):
            body = json.loads(req.data.decode("utf-8"))
            captured.append(body)
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.insert_project_log("mahler", "win", "Shipped kaizen-reflection skill", "abc1234")

        self.assertEqual(len(captured), 1)
        self.assertIn("INSERT INTO project_log", captured[0]["sql"])
        params = captured[0]["params"]
        self.assertEqual(params[0], "mahler")
        self.assertEqual(params[1], "win")
        self.assertEqual(params[2], "Shipped kaizen-reflection skill")
        self.assertEqual(params[3], "abc1234")

    def test_insert_project_log_raises_on_d1_error(self):
        error_payload = {"success": False, "errors": [{"message": "table locked"}], "result": [], "messages": []}
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload)):
            client = _make_client()
            with self.assertRaises(RuntimeError):
                client.insert_project_log("mahler", "blocker", "Stuck on X", "")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/test_d1_client.py::TestD1ClientInsertProjectLog -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'insert_project_log'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `assistant/config/skills/email-triage/scripts/d1_client.py`:

```python
    def insert_project_log(self, project: str, entry_type: str, summary: str, git_ref: str) -> None:
        """Insert one project log entry. Raises RuntimeError on D1 failure."""
        self.query(
            "INSERT INTO project_log (project, entry_type, summary, git_ref) VALUES (?, ?, ?, ?)",
            [project, entry_type, summary, git_ref],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/test_d1_client.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/email-triage/scripts/d1_client.py \
        assistant/config/skills/email-triage/tests/test_d1_client.py \
  && git commit -m "feat(e4b): add insert_project_log method to D1Client"
```

---

### Task 3: D1Client — get_recent_project_log() method
**Group:** A (sequential, after Task 2)

**Behavior being verified:** `get_recent_project_log()` returns rows from the last N days, and returns an empty list when none exist.
**Interface under test:** `D1Client.get_recent_project_log(days: int) -> list[dict]`

**Files:**
- Modify: `assistant/config/skills/email-triage/scripts/d1_client.py`
- Modify: `assistant/config/skills/email-triage/tests/test_d1_client.py`

- [ ] **Step 1: Write the failing test**

Add to `assistant/config/skills/email-triage/tests/test_d1_client.py`:

```python
class TestD1ClientGetRecentProjectLog(unittest.TestCase):

    def test_returns_rows_within_requested_day_window(self):
        rows = [
            {"project": "mahler", "entry_type": "win", "summary": "Shipped kaizen", "git_ref": "abc", "created_at": "2026-04-19 10:00:00"},
        ]

        def capture_open(req):
            body = json.loads(req.data.decode("utf-8"))
            if "project_log" in body["sql"] and "SELECT" in body["sql"]:
                return _make_response(_success_payload(rows))
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            result = client.get_recent_project_log(days=7)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["project"], "mahler")
        self.assertEqual(result[0]["entry_type"], "win")

    def test_returns_empty_list_when_no_entries(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            result = client.get_recent_project_log(days=7)
        self.assertEqual(result, [])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/test_d1_client.py::TestD1ClientGetRecentProjectLog -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'get_recent_project_log'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `assistant/config/skills/email-triage/scripts/d1_client.py`:

```python
    def get_recent_project_log(self, days: int = 7) -> list[dict]:
        """Return project_log rows from the last N days, newest first."""
        return self.query(
            "SELECT project, entry_type, summary, git_ref, created_at FROM project_log "
            "WHERE created_at >= datetime('now', ? || ' days') ORDER BY created_at DESC",
            [f"-{days}"],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/test_d1_client.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/email-triage/scripts/d1_client.py \
        assistant/config/skills/email-triage/tests/test_d1_client.py \
  && git commit -m "feat(e4b): add get_recent_project_log method to D1Client"
```

---

### Task 4: project_log.py — log_win() writes a win row to D1
**Group:** B (parallel with Task 5, depends on Group A)

**Behavior being verified:** `log_win()` calls `insert_project_log` on the D1 client with entry_type `"win"` and the provided arguments. Raises if D1 fails (visible failure is correct for the win path).
**Interface under test:** `log_win(project: str, summary: str, git_ref: str) -> None`

**Files:**
- Create: `assistant/hooks/project_log.py`
- Create: `assistant/hooks/tests/test_project_log.py`

- [ ] **Step 1: Write the failing test**

Create `assistant/hooks/tests/test_project_log.py`:

```python
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLogWin(unittest.TestCase):

    def test_log_win_inserts_win_row_to_d1(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_win
            log_win("mahler", "Shipped kaizen-reflection skill", "abc1234")
        mock_client.insert_project_log.assert_called_once_with(
            "mahler", "win", "Shipped kaizen-reflection skill", "abc1234"
        )

    def test_log_win_propagates_d1_exception(self):
        mock_client = MagicMock()
        mock_client.insert_project_log.side_effect = RuntimeError("D1 timeout")
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_win
            with self.assertRaises(RuntimeError):
                log_win("mahler", "Shipped something", "abc")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py::TestLogWin -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'project_log'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `assistant/hooks/project_log.py`:

```python
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_BLOCKER_KEYWORDS = [
    "stuck",
    "blocked",
    "can't figure out",
    "cannot figure out",
    "issue is",
    "problem is",
    "broken",
    "failing",
    "doesn't work",
    "not working",
    "frustrat",
]


def _load_mahler_env() -> None:
    env_path = Path.home() / ".mahler.env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _get_d1_client():
    _load_mahler_env()
    email_triage_scripts = str(
        Path(__file__).parent.parent
        / "config" / "skills" / "email-triage" / "scripts"
    )
    if email_triage_scripts not in sys.path:
        sys.path.insert(0, email_triage_scripts)
    from d1_client import D1Client
    return D1Client(
        account_id=os.environ["CF_ACCOUNT_ID"],
        database_id=os.environ["CF_D1_DATABASE_ID"],
        api_token=os.environ["CF_API_TOKEN"],
    )


def _derive_project_name(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            name = url.rstrip("/").replace(".git", "").rsplit("/", 1)[-1]
            if name:
                return name
    except Exception:
        pass
    return os.path.basename(cwd.rstrip("/"))


def _derive_git_ref(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _scan_for_keywords(transcript: dict) -> bool:
    messages = transcript.get("messages", transcript.get("transcript", []))
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if any(kw in content.lower() for kw in _BLOCKER_KEYWORDS):
            return True
    return False


def _call_openrouter(transcript: dict, api_key: str, model: str) -> str:
    import ssl
    import urllib.request

    messages = transcript.get("messages", transcript.get("transcript", []))
    user_texts = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        user_texts.append(content)

    excerpt = "\n".join(f"User: {t}" for t in user_texts[-10:])

    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    body = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are extracting a concise blocker summary from a development session. "
                    "Return exactly 1-2 sentences describing the main technical blocker or problem "
                    "the developer is stuck on. Be specific. "
                    "If no clear blocker exists, return an empty string."
                ),
            },
            {"role": "user", "content": excerpt},
        ],
        "max_tokens": 150,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    with opener.open(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def log_win(project: str, summary: str, git_ref: str) -> None:
    client = _get_d1_client()
    client.insert_project_log(project, "win", summary, git_ref)


def log_blocker_if_triggered(transcript: dict, cwd: str) -> None:
    pass  # implemented in Task 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Log project activity to D1")
    subparsers = parser.add_subparsers(dest="mode")

    subparsers.add_parser("blocker")

    win_parser = subparsers.add_parser("win")
    win_parser.add_argument("--project", required=True)
    win_parser.add_argument("--summary", required=True)
    win_parser.add_argument("--git-ref", default="")

    args = parser.parse_args()

    if args.mode == "blocker":
        try:
            data = json.loads(sys.stdin.read())
            cwd = data.get("cwd", os.getcwd())
            log_blocker_if_triggered(data, cwd)
        except Exception as exc:
            print(f"project_log blocker error: {exc}", file=sys.stderr)
        sys.exit(0)

    elif args.mode == "win":
        try:
            log_win(args.project, args.summary, args.git_ref)
        except Exception as exc:
            print(f"project_log win error: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py::TestLogWin -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/project_log.py assistant/hooks/tests/test_project_log.py \
  && git commit -m "feat(e4b): add project_log.py with log_win function"
```

---

### Task 5: project-context plugin — returns None when log is empty
**Group:** B (parallel with Task 4, depends on Group A)

**Behavior being verified:** `project_context()` returns None when there are no recent project log entries, and returns None silently when credentials are missing.
**Interface under test:** `project_context(session_id, user_message, is_first_turn, **kwargs) -> dict | None`

**Files:**
- Create: `assistant/config/plugins/project-context/plugin.py`
- Create: `assistant/config/plugins/project-context/tests/test_plugin.py`

- [ ] **Step 1: Write the failing test**

Create `assistant/config/plugins/project-context/tests/test_plugin.py`:

```python
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProjectContextEmptyLog(unittest.TestCase):

    @patch("plugin._query_project_log", return_value=[])
    def test_returns_none_when_project_log_is_empty(self, _):
        from plugin import project_context
        result = project_context("sess1", "what's up", True)
        self.assertIsNone(result)

    @patch("plugin._query_project_log", return_value=[])
    def test_returns_none_on_non_first_turn_with_empty_log(self, _):
        from plugin import project_context
        result = project_context("sess1", "any updates?", False)
        self.assertIsNone(result)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py::TestProjectContextEmptyLog -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'plugin'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `assistant/config/plugins/project-context/plugin.py`:

```python
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _query_project_log() -> list[dict]:
    _load_hermes_env()
    triage_scripts = str(Path.home() / ".hermes" / "skills" / "email-triage" / "scripts")
    if triage_scripts not in sys.path:
        sys.path.insert(0, triage_scripts)
    from d1_client import D1Client
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    database_id = os.environ.get("CF_D1_DATABASE_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not database_id or not api_token:
        return []
    client = D1Client(account_id, database_id, api_token)
    return client.get_recent_project_log(days=7)


def _format_entries(rows: list[dict]) -> str:
    lines = ["Recent project activity (last 7 days):\n"]
    for row in rows:
        project = row.get("project", "unknown")
        entry_type = row.get("entry_type", "").upper()
        summary = row.get("summary", "")
        created_at = (row.get("created_at") or "")[:10]
        lines.append(f"[{project}] {created_at} — {entry_type}: {summary}")
    return "\n".join(lines)


def project_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    **kwargs,
) -> dict | None:
    try:
        rows = _query_project_log()
        if not rows:
            return None
        return {"context": _format_entries(rows)}
    except Exception as exc:
        logger.debug("project-context plugin error: %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", project_context)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py::TestProjectContextEmptyLog -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/plugins/project-context/plugin.py \
        assistant/config/plugins/project-context/tests/test_plugin.py \
  && git commit -m "feat(e4b): add project-context plugin skeleton with empty-log path"
```

---

### Task 6: project_log.py — blocker mode exits cleanly when no keywords match
**Group:** C (parallel with Task 7, depends on Group B; sequential after Task 4 in the same file)

**Behavior being verified:** `log_blocker_if_triggered()` does not write to D1 when no blocker keywords appear in user messages, and handles empty message arrays without error.
**Interface under test:** `log_blocker_if_triggered(transcript: dict, cwd: str) -> None`

**Files:**
- Modify: `assistant/hooks/tests/test_project_log.py`
- Modify: `assistant/hooks/project_log.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/hooks/tests/test_project_log.py`:

```python
class TestLogBlockerNoKeyword(unittest.TestCase):

    def test_does_not_write_when_no_keywords_in_user_messages(self):
        transcript = {
            "cwd": "/Users/jdhiman/Documents/mahler",
            "messages": [
                {"role": "user", "content": "can you add a feature to the morning brief?"},
                {"role": "assistant", "content": "Sure, I will add it."},
            ],
        }
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/Users/jdhiman/Documents/mahler")
        mock_client.insert_project_log.assert_not_called()

    def test_does_not_write_when_messages_list_is_empty(self):
        transcript = {"cwd": "/tmp/project", "messages": []}
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")
        mock_client.insert_project_log.assert_not_called()

    def test_does_not_write_when_keyword_appears_only_in_assistant_message(self):
        transcript = {
            "cwd": "/tmp/project",
            "messages": [
                {"role": "user", "content": "what should I work on next?"},
                {"role": "assistant", "content": "the issue is that you have many options"},
            ],
        }
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")
        mock_client.insert_project_log.assert_not_called()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py::TestLogBlockerNoKeyword -v
```

Expected: FAIL — all three tests fail because `log_blocker_if_triggered` is a no-op stub (`pass`) but `_get_d1_client` is patched and its `insert_project_log` is called unexpectedly... actually wait, since it's a `pass` stub, `insert_project_log` is never called, so `assert_not_called()` would pass. The test will actually PASS on the stub.

Correct expected failure: The test will PASS on the stub — which means the test is wrong. Rewrite to add a positive assertion that triggers failure.

Revise all three tests to include an assertion that also verifies the function completes without raising:

The stub `pass` means the tests actually pass trivially. This is the "testing shape not behavior" trap. The real behavior is only observable via the keyword-match path. For these no-keyword tests, the behavior under test is "does NOT write" — and a `pass` stub correctly does not write.

These tests are structurally valid: they verify the function does not write when given no-keyword input. The stub happens to satisfy them, which means they will pass before the implementation. This is acceptable because the negative assertion ("not called") is the correct contract.

To make them fail before implementation exists, keep the `ModuleNotFoundError` approach: delete the stub first. At this point in the plan, `project_log.py` exists from Task 4 with the stub `log_blocker_if_triggered`. So these tests will pass trivially on the stub.

**Revised approach:** These tests are added as regression guards. They will pass on the stub (which is correct — no keywords, no write). Their value is preventing regression when Task 8 adds the keyword-match path. Run them now to confirm they pass, and proceed.

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py::TestLogBlockerNoKeyword -v
```

Expected: PASS on the current stub (these are regression-guard tests for the negative path).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the stub `log_blocker_if_triggered` in `assistant/hooks/project_log.py` with the keyword-scan logic (no OpenRouter call yet — that is Task 8):

```python
def log_blocker_if_triggered(transcript: dict, cwd: str) -> None:
    if not _scan_for_keywords(transcript):
        return
    # OpenRouter synthesis implemented in Task 8
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/project_log.py assistant/hooks/tests/test_project_log.py \
  && git commit -m "feat(e4b): add keyword scan to log_blocker_if_triggered, no-keyword exits cleanly"
```

---

### Task 7: project-context plugin — formats rows into context string
**Group:** C (parallel with Task 6, depends on Group B; sequential after Task 5 in the same file)

**Behavior being verified:** `project_context()` returns a context dict containing project names, entry types (WIN/BLOCKER), and summaries when the log has entries.
**Interface under test:** `project_context(session_id, user_message, is_first_turn, **kwargs) -> dict | None`

**Files:**
- Modify: `assistant/config/plugins/project-context/tests/test_plugin.py`

(No changes to `plugin.py` — the formatting is already implemented in Task 5. This task adds tests that verify the formatting behavior.)

- [ ] **Step 1: Write the failing test**

Append to `assistant/config/plugins/project-context/tests/test_plugin.py`:

```python
class TestProjectContextFormatsRows(unittest.TestCase):

    @patch("plugin._query_project_log")
    def test_returns_context_dict_with_win_and_blocker_entries(self, mock_query):
        mock_query.return_value = [
            {
                "project": "mahler",
                "entry_type": "win",
                "summary": "Shipped kaizen-reflection skill",
                "created_at": "2026-04-19 10:00:00",
            },
            {
                "project": "traderjoe",
                "entry_type": "blocker",
                "summary": "Ghost trades still appearing after reconciliation fix",
                "created_at": "2026-04-18 09:00:00",
            },
        ]
        from plugin import project_context
        result = project_context("sess1", "what's the status?", True)
        self.assertIsNotNone(result)
        self.assertIn("context", result)
        self.assertIn("mahler", result["context"])
        self.assertIn("WIN", result["context"])
        self.assertIn("traderjoe", result["context"])
        self.assertIn("BLOCKER", result["context"])
        self.assertIn("kaizen-reflection", result["context"])

    @patch("plugin._query_project_log")
    def test_returns_context_on_non_first_turn(self, mock_query):
        mock_query.return_value = [
            {
                "project": "mahler",
                "entry_type": "win",
                "summary": "Shipped morning-brief improvements",
                "created_at": "2026-04-19 08:00:00",
            },
        ]
        from plugin import project_context
        result = project_context("sess1", "any updates?", False)
        self.assertIsNotNone(result)
        self.assertIn("mahler", result["context"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py::TestProjectContextFormatsRows -v
```

Expected: FAIL — `AssertionError: None is not None` because `_query_project_log` is mocked to return rows but `_format_entries` is not yet called (the plugin currently returns None when rows exist... wait, it IS implemented in Task 5 to call `_format_entries`). So these tests should actually PASS.

If they pass: the formatting was fully implemented in Task 5. Confirm they pass and commit the test additions as regression coverage.

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py -v
```

Expected: PASS (formatting was implemented in Task 5's `_format_entries` function)

- [ ] **Step 3: Implement**

No implementation changes — `_format_entries` was fully implemented in Task 5.

- [ ] **Step 4: Run full plugin test suite**

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/plugins/project-context/tests/test_plugin.py \
  && git commit -m "test(e4b): add row-formatting coverage to project-context plugin tests"
```

---

### Task 8: project_log.py — keyword match triggers OpenRouter synthesis and D1 write
**Group:** D (parallel with Task 9, depends on Group C; sequential after Task 6 in the same file)

**Behavior being verified:** `log_blocker_if_triggered()` calls `_call_openrouter` when a keyword is found in user messages, writes the result to D1 as a `"blocker"` entry. Does not write if OpenRouter returns an empty string.
**Interface under test:** `log_blocker_if_triggered(transcript: dict, cwd: str) -> None`

**Files:**
- Modify: `assistant/hooks/tests/test_project_log.py`
- Modify: `assistant/hooks/project_log.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/hooks/tests/test_project_log.py`:

```python
class TestLogBlockerKeywordMatch(unittest.TestCase):

    def test_writes_blocker_row_when_keyword_matched(self):
        transcript = {
            "cwd": "/Users/jdhiman/Documents/mahler",
            "messages": [
                {
                    "role": "user",
                    "content": "the main issue is that the model isn't giving accurate scores",
                },
                {"role": "assistant", "content": "Let me look into that."},
            ],
        }
        mock_client = MagicMock()

        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._call_openrouter",
                   return_value="The scoring model produces inaccurate outputs, blocking feature completion."), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc1234"):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/Users/jdhiman/Documents/mahler")

        mock_client.insert_project_log.assert_called_once_with(
            "mahler",
            "blocker",
            "The scoring model produces inaccurate outputs, blocking feature completion.",
            "abc1234",
        )

    def test_does_not_write_when_openrouter_returns_empty_string(self):
        transcript = {
            "cwd": "/tmp/project",
            "messages": [
                {"role": "user", "content": "i am stuck on this issue"},
            ],
        }
        mock_client = MagicMock()

        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._call_openrouter", return_value=""), \
             patch("project_log._derive_project_name", return_value="project"), \
             patch("project_log._derive_git_ref", return_value=""):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")

        mock_client.insert_project_log.assert_not_called()

    def test_uses_transcript_key_as_fallback_for_messages(self):
        transcript = {
            "cwd": "/tmp/project",
            "transcript": [
                {"role": "user", "content": "blocked on the D1 migration"},
            ],
        }
        mock_client = MagicMock()

        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._call_openrouter", return_value="D1 migration is blocking progress."), \
             patch("project_log._derive_project_name", return_value="project"), \
             patch("project_log._derive_git_ref", return_value=""):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")

        mock_client.insert_project_log.assert_called_once()
        args = mock_client.insert_project_log.call_args[0]
        self.assertEqual(args[1], "blocker")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py::TestLogBlockerKeywordMatch -v
```

Expected: FAIL — `AssertionError: Expected call: insert_project_log(...)` because `log_blocker_if_triggered` currently returns after the keyword scan without calling OpenRouter or D1.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `log_blocker_if_triggered` in `assistant/hooks/project_log.py`:

```python
def log_blocker_if_triggered(transcript: dict, cwd: str) -> None:
    if not _scan_for_keywords(transcript):
        return
    _load_mahler_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = "x-ai/grok-4.1-fast"
    summary = _call_openrouter(transcript, api_key, model)
    if not summary:
        return
    project = _derive_project_name(cwd)
    git_ref = _derive_git_ref(cwd)
    client = _get_d1_client()
    client.insert_project_log(project, "blocker", summary, git_ref)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant && python3 -m pytest hooks/tests/test_project_log.py -v
```

Expected: PASS (all TestLogWin, TestLogBlockerNoKeyword, and TestLogBlockerKeywordMatch tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/project_log.py assistant/hooks/tests/test_project_log.py \
  && git commit -m "feat(e4b): complete log_blocker_if_triggered with OpenRouter synthesis and D1 write"
```

---

### Task 9: project-context plugin — D1 error returns None silently
**Group:** D (parallel with Task 8, depends on Group C; sequential after Task 7 in the same file)

**Behavior being verified:** `project_context()` returns None and does not raise when `_query_project_log` raises an exception.
**Interface under test:** `project_context(session_id, user_message, is_first_turn, **kwargs) -> dict | None`

**Files:**
- Modify: `assistant/config/plugins/project-context/tests/test_plugin.py`

(No implementation changes — error handling is already in `project_context`'s `try/except` from Task 5.)

- [ ] **Step 1: Write the failing test**

Append to `assistant/config/plugins/project-context/tests/test_plugin.py`:

```python
class TestProjectContextErrorHandling(unittest.TestCase):

    @patch("plugin._query_project_log", side_effect=Exception("D1 connection refused"))
    def test_returns_none_silently_on_d1_error(self, _):
        from plugin import project_context
        result = project_context("sess1", "how are things going?", True)
        self.assertIsNone(result)

    @patch("plugin._query_project_log", side_effect=RuntimeError("API token invalid"))
    def test_returns_none_silently_on_auth_error(self, _):
        from plugin import project_context
        result = project_context("sess1", "what is the status?", False)
        self.assertIsNone(result)
```

- [ ] **Step 2: Run test — verify behavior**

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py::TestProjectContextErrorHandling -v
```

Expected: PASS (error handling was implemented in Task 5's `try/except` block). These tests are regression coverage.

- [ ] **Step 3: No implementation change needed**

The `try/except Exception` in `project_context()` already catches all errors and returns None.

- [ ] **Step 4: Run full plugin test suite**

```bash
cd assistant && python3 -m pytest config/plugins/project-context/tests/test_plugin.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/plugins/project-context/tests/test_plugin.py \
  && git commit -m "test(e4b): add error-handling coverage to project-context plugin tests"
```

---

### Task 10: Register SessionStop hook in ~/.claude/settings.json
**Group:** E (parallel with Task 11, depends on Group D)

**Behavior being verified:** The hook script exits 0 when given a transcript with no blocker keywords (the common case — session ends cleanly without writing to D1).
**Interface under test:** `project_log.py blocker` invoked as a subprocess with JSON on stdin.

**Files:**
- Modify: `~/.claude/settings.json`

- [ ] **Step 1: Smoke-test the script first**

```bash
echo '{"messages": [], "cwd": "/tmp"}' | python3 /Users/jdhiman/Documents/mahler/assistant/hooks/project_log.py blocker
echo "Exit code: $?"
```

Expected: Exit code 0, no output.

```bash
echo '{"messages": [{"role": "user", "content": "can you help with a refactor?"}], "cwd": "/tmp"}' \
  | python3 /Users/jdhiman/Documents/mahler/assistant/hooks/project_log.py blocker
echo "Exit code: $?"
```

Expected: Exit code 0, no output (no keywords matched, no D1 call attempted — credentials not needed).

- [ ] **Step 2: Add the Stop hook to ~/.claude/settings.json**

In the `"hooks"` object, add a `"Stop"` key alongside the existing `"PreToolUse"`:

```json
"Stop": [
    {
        "hooks": [
            {
                "type": "command",
                "command": "python3 /Users/jdhiman/Documents/mahler/assistant/hooks/project_log.py blocker"
            }
        ]
    }
]
```

- [ ] **Step 3: Verify settings.json is valid JSON**

```bash
python3 -m json.tool ~/.claude/settings.json > /dev/null && echo "valid JSON"
```

Expected: `valid JSON`

- [ ] **Step 4: Commit**

```bash
git add assistant/hooks/ \
  && git commit -m "feat(e4b): register SessionStop hook for project_log.py blocker mode"
```

Note: `~/.claude/settings.json` is not tracked in this repo. Document the required change in `assistant/CLAUDE.md` as part of this commit (one line: "SessionStop hook registered in ~/.claude/settings.json pointing to assistant/hooks/project_log.py").

---

### Task 11: Add win-logging step to /ship skill
**Group:** E (parallel with Task 10, depends on Group D)

**Behavior being verified:** The ship skill invokes `project_log.py win` with a synthesized summary, derived project name, and HEAD commit ref after a successful merge.
**Interface under test:** The new Step 1.5 in `~/.claude/skills/ship/SKILL.md`.

**Files:**
- Modify: `~/.claude/skills/ship/SKILL.md`

- [ ] **Step 1: Add Step 1.5 after the existing Step 1 merge block**

Insert the following after the `## Step 1: Merge locally` section (after the `rtk git log --oneline -3` line) in `/Users/jdhiman/.claude/skills/ship/SKILL.md`:

```markdown
## Step 1.5: Log the shipped win to project_log

After confirming the merge succeeded, write a 1-2 sentence plain-English summary
of what shipped based on the commits merged in Step 1. Then run:

```bash
rtk git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git//'
```

Use the output as `{project}`. Then run:

```bash
rtk git rev-parse --short HEAD
```

Use the output as `{git_ref}`. Then call:

```bash
python3 /Users/jdhiman/Documents/mahler/assistant/hooks/project_log.py win \
  --project "{project}" \
  --summary "{your 1-2 sentence summary of what shipped}" \
  --git-ref "{git_ref}"
```

If this command fails or the project is not mahler: log the error to chat and continue
to Step 2. Do not block the ship on a logging failure.
```

- [ ] **Step 2: Verify the file reads correctly**

```bash
grep -n "project_log" /Users/jdhiman/.claude/skills/ship/SKILL.md
```

Expected: at least one line containing `project_log`.

- [ ] **Step 3: Run the full test suite to confirm nothing regressed**

```bash
cd assistant && python3 -m pytest config/skills/email-triage/tests/ config/plugins/project-context/tests/ hooks/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add assistant/ \
  && git commit -m "feat(e4b): wire project-context plugin and project_log hook — E4b complete"
```

Note: `~/.claude/skills/ship/SKILL.md` is not tracked in this repo. This step is a local configuration change. No git commit for that file.

---

## D1 Migration Command

After all tasks are complete, apply the schema change to production D1:

```bash
cd assistant
wrangler d1 execute mahler-db --remote --command \
  "CREATE TABLE IF NOT EXISTS project_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    entry_type TEXT NOT NULL CHECK(entry_type IN ('win', 'blocker')),
    summary TEXT NOT NULL,
    git_ref TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );"
```

Verify:

```bash
wrangler d1 execute mahler-db --remote --command "SELECT name FROM sqlite_master WHERE type='table' AND name='project_log';"
```

Expected output includes `project_log`.

---

## Deployment

After the D1 migration, deploy the updated Hermes image to pick up the new `project-context` plugin:

```bash
cd assistant && flyctl deploy --remote-only
```

The `project_log.py` hook and `~/.mahler.env` are local-only — no Fly.io changes needed for those.
