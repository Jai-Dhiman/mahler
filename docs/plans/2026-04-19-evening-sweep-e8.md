# Evening Task Sweep (E8) Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** At 6pm Pacific daily, Mahler queries Notion for today's completed, past-due, and open tasks, posts a short summary to Discord, picks tomorrow's top 3 priorities, and checks in on overdue items.
**Spec:** docs/specs/2026-04-19-evening-sweep-e8-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

```
Group A (sequential): Task 1, Task 2
Group B (sequential, depends on A): Task 3, Task 4, Task 5, Task 6, Task 7
Group C (parallel, depends on B): Task 8, Task 9
```

---

### Task 1: notion_client.py — last_edited_after filter and last_edited_time extraction

**Group:** A

**Behavior being verified:** `list_tasks(last_edited_after="2026-04-19")` sends a Notion timestamp filter in the request body, and the returned task dict includes `last_edited_time` from the page's top-level field.

**Interface under test:** `NotionClient.list_tasks(last_edited_after=...)` and the dict it returns.

**Files:**
- Create: `assistant/config/skills/evening-sweep/scripts/notion_client.py`
- Create: `assistant/config/skills/evening-sweep/tests/__init__.py` (empty)
- Create: `assistant/config/skills/evening-sweep/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Create `assistant/config/skills/evening-sweep/tests/test_notion_client.py`:

```python
import sys
sys.path.insert(0, 'scripts')

import json
import unittest
from unittest.mock import MagicMock, patch


def _make_response(payload: dict, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _page_fixture(
    page_id: str = "page-abc123",
    title: str = "Test task",
    status: str = "Not started",
    due: str | None = None,
    priority: str | None = None,
    last_edited_time: str | None = None,
) -> dict:
    return {
        "object": "page",
        "id": page_id,
        "last_edited_time": last_edited_time,
        "properties": {
            "Task name": {"title": [{"plain_text": title}]},
            "Status": {"status": {"name": status}},
            "Due date": {"date": {"start": due} if due else None},
            "Priority": {"select": {"name": priority} if priority else None},
        },
    }


def _list_response(
    pages: list,
    has_more: bool = False,
    next_cursor: str | None = None,
) -> dict:
    return {
        "object": "list",
        "results": pages,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


def _make_client():
    from notion_client import NotionClient
    return NotionClient(api_token="test-token", database_id="test-db-id")


class TestLastEditedAfterFilter(unittest.TestCase):

    def _capture_body(self, **kwargs) -> dict:
        from notion_client import _OPENER
        captured = []

        def side_effect(req):
            captured.append(req)
            return _make_response(_list_response([]))

        with patch.object(_OPENER, "open", side_effect=side_effect):
            client = _make_client()
            client.list_tasks(**kwargs)

        return json.loads(captured[0].data.decode("utf-8"))

    def test_last_edited_after_generates_timestamp_filter(self):
        body = self._capture_body(
            status=None, priority=None, due_before=None, last_edited_after="2026-04-19"
        )
        self.assertEqual(
            body["filter"],
            {
                "timestamp": "last_edited_time",
                "last_edited_time": {"on_or_after": "2026-04-19"},
            },
        )

    def test_last_edited_time_included_in_extracted_task(self):
        from notion_client import _OPENER
        page = _page_fixture(
            page_id="p1",
            title="Done task",
            status="Done",
            last_edited_time="2026-04-19T20:00:00.000Z",
        )
        with patch.object(_OPENER, "open", return_value=_make_response(_list_response([page]))):
            client = _make_client()
            result = client.list_tasks(status="Done", last_edited_after="2026-04-19")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["last_edited_time"], "2026-04-19T20:00:00.000Z")

    def test_no_filter_key_when_no_params_passed(self):
        body = self._capture_body(status=None, priority=None, due_before=None, last_edited_after=None)
        self.assertNotIn("filter", body)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_notion_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'notion_client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `assistant/config/skills/evening-sweep/scripts/notion_client.py`:

```python
import json
import ssl
import urllib.request
from typing import Optional


_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _extract_task(page: dict) -> dict:
    props = page.get("properties", {})
    title_list = props.get("Task name", {}).get("title", [])
    title = title_list[0]["plain_text"] if title_list else ""
    status_sel = props.get("Status", {}).get("status")
    status = status_sel["name"] if status_sel else "Todo"
    due_obj = props.get("Due date", {}).get("date")
    due = due_obj["start"] if due_obj else None
    priority_sel = props.get("Priority", {}).get("select")
    priority = priority_sel["name"] if priority_sel else None
    last_edited_time = page.get("last_edited_time")
    return {
        "id": page["id"],
        "title": title,
        "status": status,
        "due": due,
        "priority": priority,
        "last_edited_time": last_edited_time,
    }


class NotionClient:
    def __init__(self, api_token: str, database_id: str):
        if not api_token:
            raise RuntimeError("NOTION_API_TOKEN is required")
        if not database_id:
            raise RuntimeError("NOTION_DATABASE_ID is required")
        self._token = api_token
        self._database_id = database_id

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        due_before: Optional[str] = None,
        last_edited_after: Optional[str] = None,
    ) -> list[dict]:
        filters = []
        if status is not None:
            filters.append({"property": "Status", "status": {"equals": status}})
        if priority is not None:
            filters.append({"property": "Priority", "select": {"equals": priority}})
        if due_before is not None:
            filters.append({"property": "Due date", "date": {"on_or_before": due_before}})
        if last_edited_after is not None:
            filters.append({
                "timestamp": "last_edited_time",
                "last_edited_time": {"on_or_after": last_edited_after},
            })

        body: dict = {}
        if len(filters) == 1:
            body["filter"] = filters[0]
        elif len(filters) > 1:
            body["filter"] = {"and": filters}

        results = []
        cursor = None
        while True:
            if cursor is not None:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{self._database_id}/query", body)
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                raise RuntimeError(
                    "Notion API returned has_more=True but no next_cursor"
                )
        return results

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
        if status not in (200,):
            raise RuntimeError(f"Notion API error {status}: {raw.decode('utf-8', errors='replace')}")
        return json.loads(raw)
```

Also create `assistant/config/skills/evening-sweep/tests/__init__.py` (empty file).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_notion_client.py -v
```

Expected: PASS — all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/scripts/notion_client.py assistant/config/skills/evening-sweep/tests/__init__.py assistant/config/skills/evening-sweep/tests/test_notion_client.py && git commit -m "feat(evening-sweep): add notion_client with last_edited_after filter and last_edited_time extraction"
```

---

### Task 2: notion_client.py — combined last_edited_after + status uses AND clause

**Group:** A (sequential after Task 1)

**Behavior being verified:** `list_tasks(status="Done", last_edited_after="2026-04-19")` sends an AND filter combining the property filter and the timestamp filter.

**Interface under test:** `NotionClient.list_tasks(status=..., last_edited_after=...)` request body.

**Files:**
- Modify: `assistant/config/skills/evening-sweep/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Add the following class to `assistant/config/skills/evening-sweep/tests/test_notion_client.py` (after the existing `TestLastEditedAfterFilter` class):

```python
class TestCombinedFilters(unittest.TestCase):

    def _capture_body(self, **kwargs) -> dict:
        from notion_client import _OPENER
        captured = []

        def side_effect(req):
            captured.append(req)
            return _make_response(_list_response([]))

        with patch.object(_OPENER, "open", side_effect=side_effect):
            client = _make_client()
            client.list_tasks(**kwargs)

        return json.loads(captured[0].data.decode("utf-8"))

    def test_status_and_last_edited_after_combined_with_and(self):
        body = self._capture_body(
            status="Done", priority=None, due_before=None, last_edited_after="2026-04-19"
        )
        self.assertIn("and", body["filter"])
        and_clause = body["filter"]["and"]
        self.assertEqual(len(and_clause), 2)
        self.assertIn(
            {"property": "Status", "status": {"equals": "Done"}},
            and_clause,
        )
        self.assertIn(
            {
                "timestamp": "last_edited_time",
                "last_edited_time": {"on_or_after": "2026-04-19"},
            },
            and_clause,
        )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_notion_client.py -v
```

Expected: FAIL — `AssertionError: 'and' not found in ...` (because the existing implementation does not yet have the combined filter logic from Task 1 applied)

Note: If Task 1 is already committed and passing, this test should fail only if the AND combination is broken. With Task 1's implementation in place, this test will actually PASS immediately — which is the expected outcome (the implementation already handles it). Verify the test passes. If it passes without any code change, that confirms the filter-building logic is correct.

- [ ] **Step 3: Implement**

No code change required. The filter list accumulation in Task 1's `list_tasks` implementation handles this: both filters are appended to `filters`, and `len(filters) > 1` triggers the `{"and": filters}` branch.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_notion_client.py -v
```

Expected: PASS — all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/tests/test_notion_client.py && git commit -m "test(evening-sweep): verify combined status+timestamp filter uses AND clause"
```

---

### Task 3: sweep.py — COMPLETED TODAY section

**Group:** B (sequential, depends on Group A)

**Behavior being verified:** When Notion returns tasks with `status="Done"` and a `last_edited_time` from today, `sweep.py` outputs a `=== COMPLETED TODAY ===` section listing those task titles. The query uses `status="Done"` and `last_edited_after=today_str`.

**Interface under test:** `sweep.main(_today=date(2026, 4, 19))` stdout.

**Files:**
- Create: `assistant/config/skills/evening-sweep/scripts/sweep.py`
- Create: `assistant/config/skills/evening-sweep/tests/test_sweep.py`

---

- [ ] **Step 1: Write the failing test**

Create `assistant/config/skills/evening-sweep/tests/test_sweep.py`:

```python
import sys
sys.path.insert(0, 'scripts')

import os
import unittest
from datetime import date
from io import StringIO
from unittest.mock import MagicMock, patch

import sweep


def _make_task(
    page_id="page-abc",
    title="Test task",
    status="Not started",
    due=None,
    priority=None,
    last_edited_time=None,
):
    return {
        "id": page_id,
        "title": title,
        "status": status,
        "due": due,
        "priority": priority,
        "last_edited_time": last_edited_time,
    }


class TestCompletedTodaySection(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("sweep.NotionClient")
    def test_completed_task_appears_in_completed_today_section(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.side_effect = [
            [_make_task(title="Write tests", status="Done", priority="High",
                        last_edited_time="2026-04-19T20:00:00.000Z")],
            [],
            [],
        ]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            sweep.main(_today=date(2026, 4, 19))

        output = mock_out.getvalue()
        self.assertIn("=== COMPLETED TODAY ===", output)
        self.assertIn("Write tests", output)
        mock_client.list_tasks.assert_any_call(status="Done", last_edited_after="2026-04-19")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'sweep'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `assistant/config/skills/evening-sweep/scripts/sweep.py`:

```python
import os
import sys
from datetime import date, timedelta
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))


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

from notion_client import NotionClient  # noqa: E402


def _get_client() -> NotionClient:
    api_token = os.environ.get("NOTION_API_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")
    if not api_token:
        raise RuntimeError("NOTION_API_TOKEN environment variable is not set")
    if not database_id:
        raise RuntimeError("NOTION_DATABASE_ID environment variable is not set")
    return NotionClient(api_token, database_id)


def main(argv=None, _today=None):
    today = _today if _today is not None else date.today()
    yesterday = (today - timedelta(days=1)).isoformat()
    today_str = today.isoformat()

    client = _get_client()

    completed = client.list_tasks(status="Done", last_edited_after=today_str)
    overdue_raw = client.list_tasks(due_before=yesterday)
    open_tasks = client.list_tasks()

    print("=== COMPLETED TODAY ===")
    if completed:
        for t in completed:
            priority_part = f", priority={t['priority']}" if t["priority"] else ""
            print(f"- {t['title']}{priority_part}")
    else:
        print("none")

    print("\n=== PAST DUE (not done) ===")
    past_due = [t for t in overdue_raw if t["status"] != "Done"]
    if past_due:
        for t in past_due:
            days_over = (today - date.fromisoformat(t["due"])).days
            day_word = "day" if days_over == 1 else "days"
            print(f"- {t['title']} (due={t['due']}, {days_over} {day_word} overdue)")
    else:
        print("none")

    print("\n=== OPEN TASKS ===")
    if open_tasks:
        for t in open_tasks:
            parts = [t["title"]]
            if t["due"]:
                parts.append(f"due={t['due']}")
            if t["priority"]:
                parts.append(f"priority={t['priority']}")
            print(f"- {', '.join(parts)}")
    else:
        print("none")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/scripts/sweep.py assistant/config/skills/evening-sweep/tests/test_sweep.py && git commit -m "feat(evening-sweep): add sweep.py with COMPLETED TODAY section"
```

---

### Task 4: sweep.py — PAST DUE section

**Group:** B (sequential after Task 3)

**Behavior being verified:** Tasks with a due date before today and `status != Done` appear in the `=== PAST DUE (not done) ===` section with the correct days-overdue count. Tasks with `status="Done"` in the overdue query result are excluded.

**Interface under test:** `sweep.main(_today=date(2026, 4, 19))` stdout.

**Files:**
- Modify: `assistant/config/skills/evening-sweep/tests/test_sweep.py`

---

- [ ] **Step 1: Write the failing test**

Add the following class to `assistant/config/skills/evening-sweep/tests/test_sweep.py`:

```python
class TestPastDueSection(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("sweep.NotionClient")
    def test_past_due_task_shows_days_overdue_and_done_tasks_excluded(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.side_effect = [
            [],
            [
                _make_task(title="File taxes", status="Not started", due="2026-04-16"),
                _make_task(title="Already done", status="Done", due="2026-04-15"),
            ],
            [],
        ]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            sweep.main(_today=date(2026, 4, 19))

        output = mock_out.getvalue()
        self.assertIn("=== PAST DUE (not done) ===", output)
        self.assertIn("File taxes", output)
        self.assertIn("3 days overdue", output)
        self.assertNotIn("Already done", output)
        mock_client.list_tasks.assert_any_call(due_before="2026-04-18")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: FAIL — `AssertionError: '3 days overdue' not found in ...` (sweep.py exists but may not yet format days-overdue correctly, or PAST DUE filtering needs adjustment)

Note: If Task 3's implementation already includes the full `sweep.py` with PAST DUE logic, this test may pass immediately. Verify. If it passes, confirm the assertions all hold and proceed to Step 5.

- [ ] **Step 3: Implement**

The full `sweep.py` written in Task 3 already includes the PAST DUE section with `date.fromisoformat` days calculation and Done-task exclusion. No additional code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/tests/test_sweep.py && git commit -m "test(evening-sweep): verify PAST DUE section shows days overdue and excludes Done tasks"
```

---

### Task 5: sweep.py — OPEN TASKS section

**Group:** B (sequential after Task 4)

**Behavior being verified:** All open tasks from Notion appear in the `=== OPEN TASKS ===` section with title, due date, and priority on each line.

**Interface under test:** `sweep.main(_today=date(2026, 4, 19))` stdout.

**Files:**
- Modify: `assistant/config/skills/evening-sweep/tests/test_sweep.py`

---

- [ ] **Step 1: Write the failing test**

Add the following class to `assistant/config/skills/evening-sweep/tests/test_sweep.py`:

```python
class TestOpenTasksSection(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("sweep.NotionClient")
    def test_open_tasks_appear_with_title_due_and_priority(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.side_effect = [
            [],
            [],
            [
                _make_task(title="Refactor auth", due="2026-04-21", priority="High"),
                _make_task(title="Write docs", due=None, priority="Low"),
            ],
        ]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            sweep.main(_today=date(2026, 4, 19))

        output = mock_out.getvalue()
        self.assertIn("=== OPEN TASKS ===", output)
        self.assertIn("Refactor auth", output)
        self.assertIn("due=2026-04-21", output)
        self.assertIn("priority=High", output)
        self.assertIn("Write docs", output)
        self.assertIn("priority=Low", output)
        mock_client.list_tasks.assert_any_call()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: FAIL — `AssertionError: 'due=2026-04-21' not found in ...` or similar formatting mismatch.

Note: If Task 3's full implementation already passes this, proceed to Step 5.

- [ ] **Step 3: Implement**

The full `sweep.py` written in Task 3 already includes the OPEN TASKS section with `due=` and `priority=` formatting. No additional code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/tests/test_sweep.py && git commit -m "test(evening-sweep): verify OPEN TASKS section lists tasks with due and priority"
```

---

### Task 6: sweep.py — empty buckets print "none"

**Group:** B (sequential after Task 5)

**Behavior being verified:** When all three Notion queries return empty lists, each section still appears with "none" rather than being omitted.

**Interface under test:** `sweep.main(_today=date(2026, 4, 19))` stdout when all queries return `[]`.

**Files:**
- Modify: `assistant/config/skills/evening-sweep/tests/test_sweep.py`

---

- [ ] **Step 1: Write the failing test**

Add the following class to `assistant/config/skills/evening-sweep/tests/test_sweep.py`:

```python
class TestEmptyBuckets(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("sweep.NotionClient")
    def test_all_empty_buckets_print_none_for_each_section(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.side_effect = [[], [], []]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            sweep.main(_today=date(2026, 4, 19))

        output = mock_out.getvalue()
        self.assertIn("=== COMPLETED TODAY ===", output)
        self.assertIn("=== PAST DUE (not done) ===", output)
        self.assertIn("=== OPEN TASKS ===", output)
        self.assertEqual(output.count("none"), 3)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: FAIL — `AssertionError: 3 != ...` or section missing.

Note: If Task 3's implementation already passes this, proceed to Step 5.

- [ ] **Step 3: Implement**

The `sweep.py` from Task 3 already handles empty lists with `print("none")` in each section's `else` branch. No additional code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/tests/test_sweep.py && git commit -m "test(evening-sweep): verify empty buckets print none for each section"
```

---

### Task 7: sweep.py — missing env vars raise RuntimeError

**Group:** B (sequential after Task 6)

**Behavior being verified:** When `NOTION_API_TOKEN` is absent from the environment, `sweep.main()` raises `RuntimeError` with a message containing `"NOTION_API_TOKEN"`.

**Interface under test:** `sweep.main()` with empty environment.

**Files:**
- Modify: `assistant/config/skills/evening-sweep/tests/test_sweep.py`

---

- [ ] **Step 1: Write the failing test**

Add the following class to `assistant/config/skills/evening-sweep/tests/test_sweep.py`:

```python
class TestMissingEnvVars(unittest.TestCase):

    def test_missing_notion_api_token_raises_runtime_error(self):
        env_without_token = {k: v for k, v in os.environ.items()
                             if k not in ("NOTION_API_TOKEN", "NOTION_DATABASE_ID")}
        with patch.dict(os.environ, env_without_token, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                sweep.main(_today=date(2026, 4, 19))
        self.assertIn("NOTION_API_TOKEN", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: FAIL — either no `RuntimeError` is raised, or the wrong error message.

Note: If Task 3's `_get_client()` already raises with the correct message, this test may pass immediately. Verify and proceed to Step 5 if so.

- [ ] **Step 3: Implement**

The `_get_client()` function in Task 3's `sweep.py` already raises `RuntimeError("NOTION_API_TOKEN environment variable is not set")` when the token is absent. No code change required.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/evening-sweep && python3 -m unittest tests/test_sweep.py -v
```

Expected: PASS — all tests pass.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/tests/test_sweep.py && git commit -m "test(evening-sweep): verify missing NOTION_API_TOKEN raises RuntimeError"
```

---

### Task 8: SKILL.md — create evening-sweep skill definition

**Group:** C (parallel with Task 9, depends on Group B)

**Behavior being verified:** The skill file exists at the correct path with required YAML frontmatter fields and the `sweep.py` invocation procedure.

**Interface under test:** File content verification.

**Files:**
- Create: `assistant/config/skills/evening-sweep/SKILL.md`

---

- [ ] **Step 1: Write the failing test**

```bash
test -f assistant/config/skills/evening-sweep/SKILL.md && echo "PASS" || echo "FAIL: SKILL.md missing"
```

Expected: `FAIL: SKILL.md missing`

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && test -f assistant/config/skills/evening-sweep/SKILL.md && echo "PASS" || echo "FAIL: SKILL.md missing"
```

Expected: `FAIL: SKILL.md missing`

- [ ] **Step 3: Create SKILL.md**

Create `assistant/config/skills/evening-sweep/SKILL.md`:

```markdown
---
name: evening-sweep
description: Run the evening task sweep — query today's completed, past-due, and open tasks from Notion, pick tomorrow's top 3 priorities, and check in on overdue items.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [tasks, notion, evening, productivity, cron]
    related_skills: [notion-tasks]
---

## When to use

- Cron-triggered at 01:00 UTC (6pm Pacific) daily
- When the user asks for an evening summary, daily close, or "what's on for tomorrow"

## Prerequisites

| Variable | Purpose |
|---|---|
| `NOTION_API_TOKEN` | Notion internal integration token |
| `NOTION_DATABASE_ID` | ID of the Notion tasks database |

Both must be set as Fly.io secrets. The script raises `RuntimeError` if either is missing.

## Procedure

Run the task sweep:

\`\`\`bash
python3 ~/.hermes/skills/evening-sweep/scripts/sweep.py
\`\`\`

The script prints three sections to stdout:
- `=== COMPLETED TODAY ===` — tasks with status Done and last_edited_time today
- `=== PAST DUE (not done) ===` — tasks with a past due date and status not Done, with days overdue
- `=== OPEN TASKS ===` — all tasks, with title, due date, and priority

## After running sweep.py

1. From `=== OPEN TASKS ===`, select the **top 3 tasks for tomorrow** by: High priority first, then soonest due date, then Medium priority. State a one-line reason for each pick.
2. For each task in `=== PAST DUE (not done) ===`, ask the user: "Is [task title] done? If so, say `mark [task title] done` and I will complete it."
3. Post a single Discord message with three parts:
   - Completed today: count and task titles
   - Tomorrow's focus: the top 3 picks with one-line reasons
   - Past-due check-in: the question(s) from step 2, or "No overdue tasks." if the section is empty

If the sweep script fails, surface the error message to the user directly. Do not retry silently.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && test -f assistant/config/skills/evening-sweep/SKILL.md && echo "PASS" || echo "FAIL: SKILL.md missing"
```

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/evening-sweep/SKILL.md && git commit -m "feat(evening-sweep): add SKILL.md with cron prompt contract"
```

---

### Task 9: Dockerfile + entrypoint.sh — wire evening-sweep into image and cron

**Group:** C (parallel with Task 8, depends on Group B)

**Behavior being verified:** The Dockerfile includes a COPY line for evening-sweep, and the entrypoint.sh cron registration block produces a job entry with `skill="evening-sweep"` and cron expression `0 1 * * *`.

**Interface under test:** Dockerfile grep + Python execution of the entrypoint cron snippet.

**Files:**
- Modify: `assistant/Dockerfile`
- Modify: `assistant/entrypoint.sh`

---

- [ ] **Step 1: Write the failing test**

```bash
grep -q "evening-sweep" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "Dockerfile: PASS" || echo "Dockerfile: FAIL"
grep -q "evening-sweep" /Users/jdhiman/Documents/mahler/assistant/entrypoint.sh && echo "entrypoint: PASS" || echo "entrypoint: FAIL"
```

Expected: both lines print `FAIL`

- [ ] **Step 2: Run test — verify it FAILS**

```bash
grep -q "evening-sweep" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "Dockerfile: PASS" || echo "Dockerfile: FAIL"
grep -q "evening-sweep" /Users/jdhiman/Documents/mahler/assistant/entrypoint.sh && echo "entrypoint: PASS" || echo "entrypoint: FAIL"
```

Expected: `Dockerfile: FAIL` and `entrypoint: FAIL`

- [ ] **Step 3: Implement**

In `assistant/Dockerfile`, add the following line after the `meeting-prep` COPY line (line 38):

```dockerfile
COPY --chown=hermes:hermes config/skills/evening-sweep /home/hermes/.hermes/skills/evening-sweep
```

In `assistant/entrypoint.sh`, add the following block after the `kaizen-reflection` block (after line 147, before the closing `with open(jobs_file` block):

```python
if 'evening-sweep' not in existing_skills:
    jobs.append(make_job(
        ['evening-sweep'],
        'Run the evening task sweep: query today\'s completed, past-due, and open tasks from Notion, pick the top 3 priorities for tomorrow, post a summary to Discord, and check in on any overdue items.',
        '0 1 * * *',
    ))
    added.append('evening-sweep (01:00 UTC / 6pm Pacific)')
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
grep -q "evening-sweep" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "Dockerfile: PASS" || echo "Dockerfile: FAIL"
grep -q "evening-sweep" /Users/jdhiman/Documents/mahler/assistant/entrypoint.sh && echo "entrypoint: PASS" || echo "entrypoint: FAIL"
```

Expected: `Dockerfile: PASS` and `entrypoint: PASS`

Also verify the cron expression is correct:

```bash
grep "0 1 \* \* \*" /Users/jdhiman/Documents/mahler/assistant/entrypoint.sh && echo "cron expr: PASS" || echo "cron expr: FAIL"
```

Expected: `cron expr: PASS`

- [ ] **Step 5: Commit**

```bash
git add assistant/Dockerfile assistant/entrypoint.sh && git commit -m "feat(evening-sweep): wire skill into Dockerfile and entrypoint cron at 01:00 UTC"
```
