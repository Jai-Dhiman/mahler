# Kaizen Loop (Phase E3) Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** The email priority map lives in D1 (survives deploys), a weekly reflection skill proposes reclassifications from observed patterns, and a plugin injects the current map on every LLM turn.
**Spec:** assistant/docs/specs/2026-04-18-kaizen-loop-e3-design.md
**Style:** Follow the project's coding standards (assistant/CLAUDE.md)

---

## Task Groups

Group A (sequential baseline): Task 1
Group B (parallel, depends on A): Tasks 2, 3, 4
Group C (sequential, depends on B): Task 5
Group D (sequential, depends on C): Task 6
Group E (parallel, depends on D): Tasks 7, 8

---

### Task 1: Add `priority_map` table and get/set methods to email-triage D1Client

**Group:** A

**Behavior being verified:** Reading the priority map from D1 returns stored content; reading when no row exists raises `RuntimeError`; writing updates the row with an incremented version.
**Interface under test:** `D1Client.get_priority_map()` and `D1Client.set_priority_map(content)`

**Files:**
- Modify: `config/skills/email-triage/scripts/d1_client.py`
- Modify: `config/skills/email-triage/tests/test_d1_client.py`

Run tests from `config/skills/email-triage/` directory.

- [ ] **Step 1: Write the failing tests**

```python
# Add to config/skills/email-triage/tests/test_d1_client.py

class TestGetPriorityMap(unittest.TestCase):

    def test_returns_content_when_row_exists(self):
        rows = [{"content": "## URGENT\nDrop everything.", "version": 1, "updated_at": "2026-04-18T00:00:00Z"}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            result = client.get_priority_map()
        self.assertEqual(result, "## URGENT\nDrop everything.")

    def test_raises_when_no_row_exists(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.get_priority_map()
        self.assertIn("priority_map table is empty", str(ctx.exception))


class TestSetPriorityMap(unittest.TestCase):

    def test_calls_d1_with_insert_sql_and_new_content(self):
        calls = []

        def fake_open(req, timeout=None):
            calls.append(req)
            payload = {"result": [{"results": [], "success": True}], "success": True, "errors": [], "messages": []}
            return _make_response(payload)

        with patch.object(_OPENER, "open", side_effect=fake_open):
            client = _make_client()
            client.set_priority_map("## URGENT\nUpdated content.")

        self.assertEqual(len(calls), 1)
        import json as _json
        body = _json.loads(calls[0].data.decode("utf-8"))
        self.assertIn("priority_map", body["sql"])
        self.assertIn("## URGENT\nUpdated content.", body["params"])
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/skills/email-triage && python -m pytest tests/test_d1_client.py::TestGetPriorityMap tests/test_d1_client.py::TestSetPriorityMap -v
```
Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'get_priority_map'`

- [ ] **Step 3: Implement**

Add to the `D1Client` class in `config/skills/email-triage/scripts/d1_client.py`:

```python
def get_priority_map(self) -> str:
    """Read current priority map content from D1. Raises RuntimeError if no row exists."""
    rows = self.query(
        "SELECT content FROM priority_map ORDER BY version DESC LIMIT 1",
        [],
    )
    if not rows:
        raise RuntimeError(
            "priority_map table is empty — run migrate.py to seed initial content"
        )
    return rows[0]["content"]

def set_priority_map(self, content: str) -> None:
    """Write updated priority map content to D1, incrementing version."""
    self.query(
        "INSERT INTO priority_map (content, version, updated_at) "
        "VALUES (?, COALESCE((SELECT MAX(version) FROM priority_map), 0) + 1, datetime('now'))",
        [content],
    )
```

Add the `priority_map` table to the existing `ensure_tables()` method (append after the `mahler_kv` CREATE):

```python
        self.query(
            """CREATE TABLE IF NOT EXISTS priority_map (
    version INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
            [],
        )
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd config/skills/email-triage && python -m pytest tests/test_d1_client.py -v
```
Expected: PASS (all existing tests plus the two new classes)

- [ ] **Step 5: Commit**

```bash
git add config/skills/email-triage/scripts/d1_client.py config/skills/email-triage/tests/test_d1_client.py && git commit -m "feat(kaizen): add priority_map table and get/set methods to email-triage D1Client"
```

---

### Task 2: Update triage.py to read priority map from D1

**Group:** B (parallel with Tasks 3 and 4)

**Behavior being verified:** `triage.py` reads the priority map from D1 and passes it to the classifier; raises `RuntimeError` propagated from D1 if the table is empty.
**Interface under test:** `triage.main()` CLI

**Files:**
- Modify: `config/skills/email-triage/scripts/triage.py`
- Modify: `config/skills/email-triage/tests/test_triage_integration.py`

Run tests from `config/skills/email-triage/` directory.

- [ ] **Step 1: Write the failing tests**

```python
# Add to config/skills/email-triage/tests/test_triage_integration.py

class TestPriorityMapFromD1(unittest.TestCase):

    def test_triage_uses_priority_map_from_d1(self):
        email = _make_email("msg-pm-001")
        d1 = _make_d1()
        d1.get_priority_map.return_value = "## URGENT\nTest priority map."

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[email]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=[_llm_result("msg-pm-001", "FYI")]) as mock_classify,
        ):
            main(["--dry-run"])

        args, _ = mock_classify.call_args
        self.assertEqual(args[1], "## URGENT\nTest priority map.")

    def test_triage_raises_when_d1_priority_map_unavailable(self):
        d1 = _make_d1()
        d1.get_priority_map.side_effect = RuntimeError("priority_map table is empty")

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                main(["--dry-run"])
        self.assertIn("priority_map table is empty", str(ctx.exception))
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/skills/email-triage && python -m pytest tests/test_triage_integration.py::TestPriorityMapFromD1 -v
```
Expected: FAIL — `d1.get_priority_map` is not called (triage still uses filesystem)

- [ ] **Step 3: Implement**

In `config/skills/email-triage/scripts/triage.py`:

1. Delete the entire `_load_priority_map()` function.

2. In `main()`, replace:
```python
    priority_map = _load_priority_map()
```
with:
```python
    priority_map = d1.get_priority_map()
```

3. In `test_triage_integration.py`, update every existing test that patches `triage._load_priority_map`. Replace:
```python
            patch("triage._load_priority_map", return_value="priority map"),
```
with setting on the existing `d1` mock (the `d1` variable in each test already exists from `_make_d1()`):
```python
            # Before the `with` block:
            d1.get_priority_map.return_value = "priority map"
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd config/skills/email-triage && python -m pytest tests/test_triage_integration.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/email-triage/scripts/triage.py config/skills/email-triage/tests/test_triage_integration.py && git commit -m "feat(kaizen): triage.py reads priority map from D1 instead of filesystem"
```

---

### Task 3: kaizen-context plugin injects priority map on every LLM turn

**Group:** B (parallel with Tasks 2 and 4)

**Behavior being verified:** `priority_map_context()` returns a context dict containing the priority map content; returns `None` silently on any D1 error or when the map is unavailable.
**Interface under test:** `priority_map_context(session_id, user_message, is_first_turn, **kwargs)`

**Files:**
- Create: `config/plugins/kaizen-context/plugin.py`
- Create: `config/plugins/kaizen-context/tests/test_plugin.py`

Run tests from `config/plugins/kaizen-context/` directory.

- [ ] **Step 1: Write the failing tests**

```python
# config/plugins/kaizen-context/tests/test_plugin.py

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPriorityMapContextReturnsNone(unittest.TestCase):

    @patch("plugin._query_priority_map", return_value=None)
    def test_returns_none_when_priority_map_unavailable(self, _):
        from plugin import priority_map_context
        result = priority_map_context("sess1", "check my email", True)
        self.assertIsNone(result)

    @patch("plugin._query_priority_map", side_effect=Exception("D1 connection refused"))
    def test_returns_none_silently_on_any_exception(self, _):
        from plugin import priority_map_context
        result = priority_map_context("sess1", "hello", True)
        self.assertIsNone(result)


class TestPriorityMapContextReturnsContent(unittest.TestCase):

    @patch("plugin._query_priority_map")
    def test_returns_context_dict_with_priority_map_content(self, mock_query):
        mock_query.return_value = "## URGENT\nDrop everything.\n## NEEDS_ACTION\n..."
        from plugin import priority_map_context
        result = priority_map_context("sess1", "triage my email", True)
        self.assertIsNotNone(result)
        self.assertIn("URGENT", result["context"])
        self.assertIn("NEEDS_ACTION", result["context"])

    @patch("plugin._query_priority_map")
    def test_returns_context_on_non_first_turn(self, mock_query):
        mock_query.return_value = "## URGENT\nTest."
        from plugin import priority_map_context
        result = priority_map_context("sess1", "follow up", False)
        self.assertIsNotNone(result)
        self.assertIn("URGENT", result["context"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/plugins/kaizen-context && python -m pytest tests/test_plugin.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'plugin'`

- [ ] **Step 3: Implement**

Create `config/plugins/kaizen-context/plugin.py`:

```python
"""
Kaizen-context pre_llm_call plugin for Mahler.
Injects the current email priority map into every chat turn.
Returns None silently on any failure — must never break a chat turn.
"""
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


def _query_priority_map() -> str | None:
    """Read current priority map content from D1. Returns None on any error."""
    _load_hermes_env()
    triage_scripts = str(Path.home() / ".hermes" / "skills" / "email-triage" / "scripts")
    if triage_scripts not in sys.path:
        sys.path.insert(0, triage_scripts)
    from d1_client import D1Client
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    database_id = os.environ.get("CF_D1_DATABASE_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not database_id or not api_token:
        return None
    client = D1Client(account_id, database_id, api_token)
    return client.get_priority_map()


def priority_map_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    **kwargs,
) -> dict | None:
    """Called before each LLM turn. Injects email priority map or returns None."""
    try:
        content = _query_priority_map()
        if not content:
            return None
        return {"context": f"Email priority map (active classification rules):\n\n{content}"}
    except Exception as exc:
        logger.debug("kaizen-context plugin error: %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", priority_map_context)
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd config/plugins/kaizen-context && python -m pytest tests/test_plugin.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/plugins/kaizen-context/ && git commit -m "feat(kaizen): add kaizen-context plugin to inject priority map on every LLM turn"
```

---

### Task 4: kaizen-reflection D1Client with triage pattern query

**Group:** B (parallel with Tasks 2 and 3)

**Behavior being verified:** `get_triage_patterns()` returns senders appearing at or above `min_count` at the same tier; returns empty list when no patterns qualify. `get_priority_map()` raises on empty table.
**Interface under test:** `D1Client.get_triage_patterns()`, `D1Client.get_priority_map()`

**Files:**
- Create: `config/skills/kaizen-reflection/scripts/d1_client.py`
- Create: `config/skills/kaizen-reflection/tests/test_d1_client.py`

Run tests from `config/skills/kaizen-reflection/` directory.

- [ ] **Step 1: Write the failing tests**

```python
# config/skills/kaizen-reflection/tests/test_d1_client.py

import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from d1_client import D1Client, _OPENER


def _make_response(payload: dict, status: int = 200):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _success_payload(rows: list) -> dict:
    return {
        "result": [{"results": rows, "success": True}],
        "success": True,
        "errors": [],
        "messages": [],
    }


def _make_client() -> D1Client:
    return D1Client(
        account_id="test-account-123",
        database_id="test-db-456",
        api_token="test-token-abc",
    )


class TestGetTriagePatterns(unittest.TestCase):

    def test_returns_senders_meeting_min_count_threshold(self):
        rows = [
            {"from_addr": "news@acme.com", "classification": "NEEDS_ACTION", "occurrence_count": 5}
        ]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            patterns = client.get_triage_patterns(since_days=7, min_count=3)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["from_addr"], "news@acme.com")
        self.assertEqual(patterns[0]["classification"], "NEEDS_ACTION")
        self.assertEqual(patterns[0]["occurrence_count"], 5)

    def test_returns_empty_list_when_no_patterns_meet_threshold(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            patterns = client.get_triage_patterns(since_days=7, min_count=3)
        self.assertEqual(patterns, [])


class TestGetPriorityMapKaizen(unittest.TestCase):

    def test_returns_content_when_row_exists(self):
        rows = [{"content": "## URGENT\nDrop everything."}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            result = client.get_priority_map()
        self.assertEqual(result, "## URGENT\nDrop everything.")

    def test_raises_when_no_row_exists(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.get_priority_map()
        self.assertIn("priority_map table is empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_d1_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'd1_client'`

- [ ] **Step 3: Implement**

Create `config/skills/kaizen-reflection/scripts/d1_client.py`:

```python
import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Optional

_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


class D1Client:
    def __init__(self, account_id: str, database_id: str, api_token: str):
        if not _ID_RE.match(account_id):
            raise ValueError(f"Invalid account_id: {account_id!r}")
        if not _ID_RE.match(database_id):
            raise ValueError(f"Invalid database_id: {database_id!r}")
        self.account_id = account_id
        self.database_id = database_id
        self.api_token = api_token
        self._url = _URL_TEMPLATE.format(account_id=account_id, database_id=database_id)

    def query(self, sql: str, params: Optional[list] = None) -> list[dict]:
        """Execute SQL against D1. Returns list of row dicts. Raises on error."""
        body = json.dumps({"sql": sql, "params": params or []}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with _OPENER.open(req) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"D1 API error (connection failed): {exc.reason}") from exc
        if status != 200:
            raise RuntimeError(f"D1 API error {status}: {raw.decode('utf-8', errors='replace')}")
        data = json.loads(raw)
        if not data.get("success") or data.get("errors"):
            raise RuntimeError(f"D1 query failed: {data.get('errors', [])}")
        results = data.get("result", [])
        if not results:
            return []
        return results[0].get("results") or []

    def get_triage_patterns(self, since_days: int = 7, min_count: int = 3) -> list[dict]:
        """Return senders appearing >= min_count times at the same tier in the last since_days days."""
        return self.query(
            """SELECT from_addr, classification, COUNT(*) AS occurrence_count
               FROM email_triage_log
               WHERE processed_at >= datetime('now', ? || ' days')
               GROUP BY from_addr, classification
               HAVING COUNT(*) >= ?
               ORDER BY occurrence_count DESC""",
            [f"-{since_days}", min_count],
        )

    def get_priority_map(self) -> str:
        """Read the current priority map content from D1. Raises RuntimeError if no row exists."""
        rows = self.query(
            "SELECT content FROM priority_map ORDER BY version DESC LIMIT 1",
            [],
        )
        if not rows:
            raise RuntimeError(
                "priority_map table is empty — run migrate.py to seed initial content"
            )
        return rows[0]["content"]

    def set_priority_map(self, content: str) -> None:
        """Write updated priority map content to D1, incrementing version."""
        self.query(
            "INSERT INTO priority_map (content, version, updated_at) "
            "VALUES (?, COALESCE((SELECT MAX(version) FROM priority_map), 0) + 1, datetime('now'))",
            [content],
        )

    def ensure_priority_map_table(self) -> None:
        """Create priority_map table if it does not exist. Does not seed initial content."""
        self.query(
            """CREATE TABLE IF NOT EXISTS priority_map (
    version INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
            [],
        )
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_d1_client.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/kaizen-reflection/scripts/d1_client.py config/skills/kaizen-reflection/tests/test_d1_client.py && git commit -m "feat(kaizen): add kaizen-reflection D1Client with triage pattern query"
```

---

### Task 5: reflect.py — `--run` mode detects patterns and outputs JSON proposals

**Group:** C (depends on Group B)

**Behavior being verified:** `reflect.py --run` outputs a JSON array of proposals for detected patterns; outputs `"No proposals this week."` and exits 0 when no patterns qualify.
**Interface under test:** `reflect.main(["--run"])` stdout

**Files:**
- Create: `config/skills/kaizen-reflection/scripts/reflect.py`
- Create: `config/skills/kaizen-reflection/tests/test_reflect.py`

Run tests from `config/skills/kaizen-reflection/` directory.

- [ ] **Step 1: Write the failing tests**

```python
# config/skills/kaizen-reflection/tests/test_reflect.py

import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct123",
    "CF_D1_DATABASE_ID": "db123",
    "CF_API_TOKEN": "cftoken",
    "OPENROUTER_API_KEY": "orkey",
}


def _patch_env(extra: dict | None = None):
    env = dict(_BASE_ENV)
    if extra:
        env.update(extra)
    return patch.dict("os.environ", env, clear=True)


class TestReflectRunWithProposals(unittest.TestCase):

    def test_outputs_json_proposals_for_detected_patterns(self):
        patterns = [
            {"from_addr": "news@acme.com", "classification": "NEEDS_ACTION", "occurrence_count": 5}
        ]
        llm_response = json.dumps([
            {
                "sender": "news@acme.com",
                "current_tier": "NEEDS_ACTION",
                "proposed_tier": "FYI",
                "evidence": "5 occurrences in 7 days with no follow-up",
            }
        ])
        mock_d1 = MagicMock()
        mock_d1.get_triage_patterns.return_value = patterns
        mock_d1.get_priority_map.return_value = "## URGENT\nDrop everything."

        captured = io.StringIO()
        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm", return_value=llm_response),
            patch("sys.stdout", captured),
        ):
            import reflect
            reflect.main(["--run"])

        proposals = json.loads(captured.getvalue())
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["sender"], "news@acme.com")
        self.assertEqual(proposals[0]["proposed_tier"], "FYI")
        self.assertIn("evidence", proposals[0])


class TestReflectRunNoProposals(unittest.TestCase):

    def test_prints_no_proposals_message_when_no_patterns(self):
        mock_d1 = MagicMock()
        mock_d1.get_triage_patterns.return_value = []

        captured = io.StringIO()
        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("sys.stdout", captured),
        ):
            import reflect
            reflect.main(["--run"])

        self.assertIn("No proposals this week", captured.getvalue())

    def test_does_not_call_llm_when_no_patterns(self):
        mock_d1 = MagicMock()
        mock_d1.get_triage_patterns.return_value = []

        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm") as mock_llm,
            patch("sys.stdout", io.StringIO()),
        ):
            import reflect
            reflect.main(["--run"])

        mock_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_reflect.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'reflect'`

- [ ] **Step 3: Implement**

Create `config/skills/kaizen-reflection/scripts/reflect.py`:

```python
"""
Kaizen reflection script.

Usage:
    python3 reflect.py --run [--since-days N] [--dry-run]
    python3 reflect.py --apply PROPOSAL_JSON

--run         Detect patterns in email_triage_log, generate proposals, print JSON to stdout.
--apply JSON  Apply a single approved proposal to the priority map in D1.
--since-days N  Lookback window in days (default: 7).
--dry-run     (--run only) Print proposals without writing to D1.
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from d1_client import D1Client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "x-ai/grok-4.1-fast"
_REQUIRED_ENV = ["CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN", "OPENROUTER_API_KEY"]


def _supplement_env_from_hermes() -> None:
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


_supplement_env_from_hermes()


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


def _call_llm(prompt: str, api_key: str, model: str) -> str:
    """Call OpenRouter with a single user message. Returns the response content string."""
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
        if exc.code in (401, 403):
            raise RuntimeError(
                f"OpenRouter auth failure: HTTP {exc.code} — check OPENROUTER_API_KEY"
            ) from exc
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    return data["choices"][0]["message"]["content"]


_PROPOSAL_PROMPT = """\
You are analyzing email triage patterns for a personal chief-of-staff assistant.

Current priority map:
{priority_map}

Detected patterns (senders appearing frequently at the same classification tier):
{patterns}

For each pattern, propose a reclassification only if the current tier appears wrong.
A sender appearing many times as NEEDS_ACTION with no escalation signal likely belongs at FYI.

Return a JSON array (no other text). Each element must have exactly these fields:
{{"sender": "<email address>", "current_tier": "<tier>", "proposed_tier": "<tier>", "evidence": "<one sentence>"}}

If no reclassifications are warranted, return an empty JSON array: []\
"""

_APPLY_PROMPT = """\
You are editing an email classification priority map in markdown format.

Current priority map:
{priority_map}

Apply this change: move the sender "{sender}" from {current_tier} to {proposed_tier}.
Evidence: {evidence}

If {sender} appears as an example under {current_tier}, move that line to {proposed_tier}.
If it does not appear explicitly, add it as a new example under {proposed_tier}.
Return the complete updated priority map as a markdown document. No other text.\
"""


def _load_env() -> dict:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _run(since_days: int, dry_run: bool, env: dict) -> None:
    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    patterns = d1.get_triage_patterns(since_days=since_days, min_count=3)
    if not patterns:
        print("No proposals this week.")
        return

    priority_map = d1.get_priority_map()
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)

    patterns_text = "\n".join(
        f"- {p['from_addr']}: {p['occurrence_count']} times as {p['classification']}"
        for p in patterns
    )
    prompt = _PROPOSAL_PROMPT.format(priority_map=priority_map, patterns=patterns_text)
    raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)

    try:
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            raise ValueError("LLM did not return a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"LLM returned unparseable proposals: {exc}\nRaw: {raw}") from exc

    if not proposals:
        print("No proposals this week.")
        return

    print(json.dumps(proposals))


def _apply(proposal_json: str, env: dict) -> None:
    try:
        proposal = json.loads(proposal_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid proposal JSON: {exc}") from exc

    required_keys = {"sender", "current_tier", "proposed_tier", "evidence"}
    missing = required_keys - proposal.keys()
    if missing:
        raise RuntimeError(f"Proposal missing required keys: {missing}")

    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    priority_map = d1.get_priority_map()
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)

    prompt = _APPLY_PROMPT.format(
        priority_map=priority_map,
        sender=proposal["sender"],
        current_tier=proposal["current_tier"],
        proposed_tier=proposal["proposed_tier"],
        evidence=proposal["evidence"],
    )
    updated_map = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)
    d1.set_priority_map(updated_map)
    print(f"Priority map updated. Moved {proposal['sender']} to {proposal['proposed_tier']}.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mahler kaizen reflection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--apply", metavar="PROPOSAL_JSON")
    parser.add_argument("--since-days", type=int, default=7, metavar="N")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env = _load_env()
    if args.run:
        _run(since_days=args.since_days, dry_run=args.dry_run, env=env)
    else:
        _apply(proposal_json=args.apply, env=env)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_reflect.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/kaizen-reflection/scripts/reflect.py config/skills/kaizen-reflection/tests/test_reflect.py && git commit -m "feat(kaizen): add reflect.py --run mode with LLM proposal generation"
```

---

### Task 6: reflect.py — `--apply` mode writes updated priority map to D1

**Group:** D (depends on Group C — same file as Task 5)

**Behavior being verified:** `reflect.py --apply PROPOSAL_JSON` reads the current priority map, calls LLM to apply the change, and writes the result to D1; raises `RuntimeError` on invalid JSON or missing proposal keys.
**Interface under test:** `reflect.main(["--apply", proposal_json])` side effect on D1

**Files:**
- Modify: `config/skills/kaizen-reflection/tests/test_reflect.py`

(The `_apply()` function is already implemented in Task 5 — these tests confirm correct behavior.)

- [ ] **Step 1: Write the failing tests**

```python
# Add to config/skills/kaizen-reflection/tests/test_reflect.py

class TestReflectApply(unittest.TestCase):

    def _make_proposal(self, **overrides) -> str:
        base = {
            "sender": "news@acme.com",
            "current_tier": "NEEDS_ACTION",
            "proposed_tier": "FYI",
            "evidence": "5 occurrences in 7 days with no follow-up",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_writes_updated_map_to_d1(self):
        current_map = "## NEEDS_ACTION\n\n**Examples:**\n- Direct questions"
        updated_map = "## FYI\n\n**Examples:**\n- news@acme.com"

        mock_d1 = MagicMock()
        mock_d1.get_priority_map.return_value = current_map

        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm", return_value=updated_map),
        ):
            import reflect
            reflect.main(["--apply", self._make_proposal()])

        mock_d1.set_priority_map.assert_called_once_with(updated_map)

    def test_raises_on_invalid_proposal_json(self):
        with (
            _patch_env(),
            patch("reflect.D1Client"),
        ):
            import reflect
            with self.assertRaises(RuntimeError) as ctx:
                reflect.main(["--apply", "not-valid-json"])
        self.assertIn("Invalid proposal JSON", str(ctx.exception))

    def test_raises_on_proposal_missing_required_keys(self):
        incomplete = json.dumps({"sender": "news@acme.com"})
        with (
            _patch_env(),
            patch("reflect.D1Client"),
        ):
            import reflect
            with self.assertRaises(RuntimeError) as ctx:
                reflect.main(["--apply", incomplete])
        self.assertIn("missing required keys", str(ctx.exception))
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_reflect.py::TestReflectApply -v
```
Expected: FAIL on `test_writes_updated_map_to_d1` — if `_apply` is not yet fully wired, `set_priority_map` will not be called. The validation tests may also fail on missing error messages.

- [ ] **Step 3: Implement (if needed)**

The `_apply()` function in `reflect.py` is already written in Task 5. Run the tests. If any fail, fix the specific gap:
- If `set_priority_map` is not called: verify `_apply()` calls `d1.set_priority_map(updated_map)` after the LLM call
- If error message doesn't match: adjust the `raise RuntimeError(...)` message to match `"Invalid proposal JSON"` and `"missing required keys"`

- [ ] **Step 4: Run all reflect tests — verify they PASS**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_reflect.py -v
```
Expected: PASS (all tests from Tasks 5 and 6)

- [ ] **Step 5: Commit**

```bash
git add config/skills/kaizen-reflection/tests/test_reflect.py && git commit -m "test(kaizen): verify reflect.py --apply writes updated priority map to D1"
```

---

### Task 7: migrate.py seeds priority_map table from file

**Group:** E (parallel with Task 8)

**Behavior being verified:** `migrate.py --file PATH` inserts file content as the initial D1 priority_map row; raises `RuntimeError` if a row already exists.
**Interface under test:** `migrate.main(["--file", path])` side effect on D1

**Files:**
- Create: `config/skills/kaizen-reflection/scripts/migrate.py`
- Create: `config/skills/kaizen-reflection/tests/test_migrate.py`

Run tests from `config/skills/kaizen-reflection/` directory.

- [ ] **Step 1: Write the failing tests**

```python
# config/skills/kaizen-reflection/tests/test_migrate.py

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct123",
    "CF_D1_DATABASE_ID": "db123",
    "CF_API_TOKEN": "cftoken",
}


def _patch_env():
    return patch.dict("os.environ", _BASE_ENV, clear=True)


class TestMigrate(unittest.TestCase):

    def _write_temp_map(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_inserts_file_content_into_d1_when_table_is_empty(self):
        tmp = self._write_temp_map("## URGENT\nTest priority map content.")
        mock_d1 = MagicMock()
        mock_d1.get_priority_map.side_effect = RuntimeError("priority_map table is empty")

        with (
            _patch_env(),
            patch("migrate.D1Client", return_value=mock_d1),
        ):
            import migrate
            migrate.main(["--file", tmp])

        mock_d1.set_priority_map.assert_called_once()
        inserted_content = mock_d1.set_priority_map.call_args[0][0]
        self.assertIn("URGENT", inserted_content)
        self.assertIn("Test priority map content", inserted_content)

    def test_raises_if_priority_map_already_exists_in_d1(self):
        tmp = self._write_temp_map("## URGENT\nContent.")
        mock_d1 = MagicMock()
        mock_d1.get_priority_map.return_value = "existing content"

        with (
            _patch_env(),
            patch("migrate.D1Client", return_value=mock_d1),
        ):
            import migrate
            with self.assertRaises(RuntimeError) as ctx:
                migrate.main(["--file", tmp])
        self.assertIn("already seeded", str(ctx.exception))
        mock_d1.set_priority_map.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_migrate.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'migrate'`

- [ ] **Step 3: Implement**

Create `config/skills/kaizen-reflection/scripts/migrate.py`:

```python
"""
One-time migration: seed the priority_map table in D1 from a local file.

Usage:
    python3 migrate.py --file PATH

Raises RuntimeError if the table already has a row (prevents double-seeding).
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from d1_client import D1Client

_REQUIRED_ENV = ["CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN"]


def _supplement_env_from_hermes() -> None:
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


_supplement_env_from_hermes()


def _load_env() -> dict:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed priority_map table in D1")
    parser.add_argument("--file", required=True, metavar="PATH", help="Path to priority-map.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env = _load_env()

    map_path = Path(args.file).expanduser()
    with open(map_path, "r", encoding="utf-8") as f:
        content = f.read()

    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )

    d1.ensure_priority_map_table()

    try:
        d1.get_priority_map()
        raise RuntimeError(
            "priority_map table is already seeded — use reflect.py --apply to make changes"
        )
    except RuntimeError as exc:
        if "already seeded" in str(exc):
            raise

    d1.set_priority_map(content)
    print(f"Priority map seeded from {map_path} (version 1).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd config/skills/kaizen-reflection && python -m pytest tests/test_migrate.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/kaizen-reflection/scripts/migrate.py config/skills/kaizen-reflection/tests/test_migrate.py && git commit -m "feat(kaizen): add migrate.py to seed priority_map table from file"
```

---

### Task 8: SKILL.md, Dockerfile, and entrypoint.sh wiring

**Group:** E (parallel with Task 7)

**Behavior being verified:** After deploy, the kaizen-reflection skill is available to Mahler, the kaizen-context plugin loads on startup, and the weekly Sunday 18:00 UTC cron job is registered in `~/.hermes/cron/jobs.json`.
**Interface under test:** File contents — verify by inspection.

**Files:**
- Create: `config/skills/kaizen-reflection/SKILL.md`
- Modify: `Dockerfile`
- Modify: `entrypoint.sh`

- [ ] **Step 1: Create `config/skills/kaizen-reflection/SKILL.md`**

```markdown
---
name: kaizen-reflection
description: Weekly email triage reflection. Analyzes email_triage_log patterns from the past 7 days, proposes priority-map reclassifications, and applies approved changes to the D1 priority_map table.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [email, triage, kaizen, priority-map, reflection, productivity]
    related_skills: [email-triage, morning-brief]
---

## When to use

- Invoked automatically by Hermes cron every Sunday at 18:00 UTC
- When the user asks any of: "run kaizen reflection", "check triage patterns", "update priority map", "what should we reclassify"

## Prerequisites

| Variable | Purpose |
|---|---|
| `CF_ACCOUNT_ID` | Cloudflare account ID for D1 API calls |
| `CF_D1_DATABASE_ID` | D1 database ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 read/write permission |
| `OPENROUTER_API_KEY` | API key for LLM proposal generation |

The `priority_map` table must be seeded before first use. Run this once after the first deploy:

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/migrate.py \
    --file ~/.hermes/workspace/priority-map.md
```

## Procedure

### Generate weekly proposals

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/reflect.py --run
```

Output is a JSON array of proposals, or `"No proposals this week."` if no patterns qualify. Dry run:

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/reflect.py --run --dry-run
```

### Apply an approved proposal

When the user approves a proposal, call --apply with the JSON object from the --run output:

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/reflect.py \
    --apply '{"sender": "news@acme.com", "current_tier": "NEEDS_ACTION", "proposed_tier": "FYI", "evidence": "5 occurrences in 7 days with no follow-up"}'
```

Prints `"Priority map updated. Moved <sender> to <tier>."` on success. Raises `RuntimeError` on D1 write failure.

## Cron flow

1. Every Sunday at 18:00 UTC, Mahler runs `reflect.py --run`
2. If proposals are returned, Mahler posts each one to Discord as a separate message with approve/deny buttons
3. On approval of a specific proposal, Mahler calls `reflect.py --apply PROPOSAL_JSON` for that proposal
4. If `"No proposals this week."` is returned, Mahler reports this to Discord and takes no further action
```

- [ ] **Step 2: Add COPY line to `Dockerfile`**

In `Dockerfile`, after the existing `COPY` line for `meeting-prep`, add:

```dockerfile
COPY --chown=hermes:hermes config/skills/kaizen-reflection /home/hermes/.hermes/skills/kaizen-reflection
```

The existing line `COPY --chown=hermes:hermes config/plugins /home/hermes/.hermes/plugins` already covers the new `kaizen-context` plugin directory — no additional change needed for the plugin.

- [ ] **Step 3: Add cron job to `entrypoint.sh`**

In `entrypoint.sh`, inside the Python block that registers cron jobs, add after the `meeting-prep` block and before `with open(jobs_file, 'w') as f:`:

```python
if 'kaizen-reflection' not in existing_skills:
    jobs.append(make_job(
        ['kaizen-reflection'],
        'Run the weekly kaizen reflection: analyze email triage patterns from the past 7 days, generate reclassification proposals, and present each to Discord with approve/deny buttons.',
        '0 18 * * 0',
    ))
    added.append('kaizen-reflection (Sundays 18:00 UTC)')
```

The cron expression `0 18 * * 0` means: minute 0, hour 18 UTC, any day of month, any month, weekday 0 (Sunday).

- [ ] **Step 4: Verify the changes**

```bash
grep -n "kaizen" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && grep -n "kaizen" /Users/jdhiman/Documents/mahler/assistant/entrypoint.sh
```
Expected: Dockerfile shows the COPY line for `kaizen-reflection`; entrypoint.sh shows the cron job registration block.

- [ ] **Step 5: Commit**

```bash
git add config/skills/kaizen-reflection/SKILL.md /Users/jdhiman/Documents/mahler/assistant/Dockerfile /Users/jdhiman/Documents/mahler/assistant/entrypoint.sh && git commit -m "feat(kaizen): wire kaizen-reflection skill and kaizen-context plugin into Dockerfile and entrypoint cron"
```

---

## Post-deploy one-time step

After `flyctl deploy --remote-only`, SSH in and seed the priority_map table:

```bash
flyctl ssh console --user hermes -C \
    "python3 ~/.hermes/skills/kaizen-reflection/scripts/migrate.py --file ~/.hermes/workspace/priority-map.md"
```

The `priority-map.md` is still copied to the workspace at build time by the unchanged Dockerfile line — it is available for this one-time seed. After seeding, the filesystem copy is no longer used; D1 is authoritative.
