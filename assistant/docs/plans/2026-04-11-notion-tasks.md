# Notion Tasks Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Manage personal tasks in Notion via natural language through Discord — create, list, update, complete, and delete tasks without leaving the chat.
**Spec:** docs/specs/2026-04-11-notion-tasks-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

```
Group A (parallel): Task 1
Group B (sequential, depends on A — all modify notion_client.py): Tasks 2, 3, 4, 5, 6, 7
Group C (sequential, depends on B — all modify tasks.py): Tasks 8, 9, 10
Group D (sequential, depends on C): Task 11
```

---

### Task 1: NotionClient init and HTTPS enforcement

**Group:** A

**Behavior being verified:** NotionClient raises RuntimeError on missing credentials and accepts valid credentials without making network calls.

**Interface under test:** `NotionClient.__init__`

**Files:**
- Create: `config/skills/notion-tasks/scripts/notion_client.py`
- Create: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Create `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
import sys
sys.path.insert(0, 'scripts')

import json
import unittest
from unittest.mock import MagicMock, patch

from notion_client import NotionClient, _OPENER


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
    status: str = "Todo",
    due: str | None = None,
    priority: str | None = None,
) -> dict:
    return {
        "object": "page",
        "id": page_id,
        "properties": {
            "Name": {"title": [{"plain_text": title}]},
            "Status": {"select": {"name": status}},
            "Due": {"date": {"start": due} if due else None},
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


def _make_client() -> NotionClient:
    return NotionClient(api_token="test-token", database_id="test-db-id")


class TestNotionClientInit(unittest.TestCase):

    def test_empty_api_token_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionClient(api_token="", database_id="test-db-id")
        self.assertIn("NOTION_API_TOKEN", str(ctx.exception))

    def test_empty_database_id_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionClient(api_token="test-token", database_id="")
        self.assertIn("NOTION_DATABASE_ID", str(ctx.exception))

    def test_valid_credentials_succeed_without_network_calls(self):
        with patch.object(_OPENER, "open") as mock_open:
            client = NotionClient(api_token="tok", database_id="db-id")
            mock_open.assert_not_called()
        self.assertIsNotNone(client)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestNotionClientInit -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'notion_client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `config/skills/notion-tasks/scripts/notion_client.py`:

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
    title_list = props.get("Name", {}).get("title", [])
    title = title_list[0]["plain_text"] if title_list else ""
    status_sel = props.get("Status", {}).get("select")
    status = status_sel["name"] if status_sel else "Todo"
    due_obj = props.get("Due", {}).get("date")
    due = due_obj["start"] if due_obj else None
    priority_sel = props.get("Priority", {}).get("select")
    priority = priority_sel["name"] if priority_sel else None
    return {
        "id": page["id"],
        "title": title,
        "status": status,
        "due": due,
        "priority": priority,
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

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
        return status, json.loads(raw)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestNotionClientInit -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/notion_client.py config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add NotionClient init and HTTPS-only enforcement"
```

---

### Task 2: create_task

**Group:** B (sequential after Task 1)

**Behavior being verified:** `create_task` sends a correctly structured Notion API payload and returns a flat task dict.

**Interface under test:** `NotionClient.create_task`

**Files:**
- Modify: `config/skills/notion-tasks/scripts/notion_client.py`
- Modify: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
class TestCreateTask(unittest.TestCase):

    def test_create_task_sends_correct_payload_and_returns_task_dict(self):
        page = _page_fixture(page_id="new-page-id", title="Buy groceries")
        captured = []

        def capture_open(req):
            captured.append(req)
            return _make_response(page)

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            result = client.create_task(title="Buy groceries", due=None, priority=None)

        self.assertEqual(len(captured), 1)
        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(body["parent"]["database_id"], "test-db-id")
        self.assertEqual(body["properties"]["Name"]["title"][0]["text"]["content"], "Buy groceries")
        self.assertEqual(body["properties"]["Status"]["select"]["name"], "Todo")
        self.assertNotIn("Due", body["properties"])
        self.assertNotIn("Priority", body["properties"])
        self.assertEqual(result["id"], "new-page-id")
        self.assertEqual(result["title"], "Buy groceries")
        self.assertEqual(result["status"], "Todo")
        self.assertIsNone(result["due"])
        self.assertIsNone(result["priority"])

    def test_create_task_includes_due_and_priority_when_provided(self):
        page = _page_fixture(
            page_id="page-xyz",
            title="Fix bug",
            status="Todo",
            due="2026-04-17",
            priority="High",
        )
        with patch.object(_OPENER, "open", return_value=_make_response(page)) as mock_open:
            client = _make_client()
            result = client.create_task(title="Fix bug", due="2026-04-17", priority="High")

        body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(body["properties"]["Due"]["date"]["start"], "2026-04-17")
        self.assertEqual(body["properties"]["Priority"]["select"]["name"], "High")
        self.assertEqual(result["due"], "2026-04-17")
        self.assertEqual(result["priority"], "High")

    def test_create_task_raises_on_api_error(self):
        error_payload = {"object": "error", "status": 400, "message": "bad request"}
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload, status=400)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.create_task(title="Bad task", due=None, priority=None)
        self.assertIn("400", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestCreateTask -v
```

Expected: FAIL — `AttributeError: 'NotionClient' object has no attribute 'create_task'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `NotionClient` in `config/skills/notion-tasks/scripts/notion_client.py`:

```python
    def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> dict:
        properties: dict = {
            "Name": {"title": [{"text": {"content": title}}]},
            "Status": {"select": {"name": "Todo"}},
        }
        if due is not None:
            properties["Due"] = {"date": {"start": due}}
        if priority is not None:
            properties["Priority"] = {"select": {"name": priority}}

        body = {
            "parent": {"database_id": self._database_id},
            "properties": properties,
        }
        status, data = self._request("POST", "/pages", body)
        if status != 200:
            raise RuntimeError(f"Notion API error {status}: {data}")
        return _extract_task(data)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestCreateTask -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/notion_client.py config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add create_task"
```

---

### Task 3: list_tasks (no filters, pagination)

**Group:** B (sequential after Task 2)

**Behavior being verified:** `list_tasks` with no filters returns all tasks and follows pagination until `has_more` is false.

**Interface under test:** `NotionClient.list_tasks`

**Files:**
- Modify: `config/skills/notion-tasks/scripts/notion_client.py`
- Modify: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
class TestListTasksNoFilter(unittest.TestCase):

    def test_list_tasks_with_no_filters_returns_all_tasks(self):
        pages = [
            _page_fixture(page_id="page-1", title="Task one"),
            _page_fixture(page_id="page-2", title="Task two"),
        ]
        with patch.object(_OPENER, "open", return_value=_make_response(_list_response(pages))):
            client = _make_client()
            result = client.list_tasks(status=None, priority=None, due_before=None)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "page-1")
        self.assertEqual(result[0]["title"], "Task one")
        self.assertEqual(result[1]["id"], "page-2")

    def test_list_tasks_no_filter_sends_no_filter_key_in_body(self):
        captured = []

        def capture_open(req):
            captured.append(req)
            return _make_response(_list_response([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.list_tasks(status=None, priority=None, due_before=None)

        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertNotIn("filter", body)

    def test_list_tasks_follows_pagination_cursor(self):
        page1 = _page_fixture(page_id="page-1", title="First")
        page2 = _page_fixture(page_id="page-2", title="Second")
        responses = [
            _make_response(_list_response([page1], has_more=True, next_cursor="cursor-abc")),
            _make_response(_list_response([page2], has_more=False)),
        ]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            client = _make_client()
            result = client.list_tasks(status=None, priority=None, due_before=None)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "page-1")
        self.assertEqual(result[1]["id"], "page-2")
        second_body = json.loads(calls[1].data.decode("utf-8"))
        self.assertEqual(second_body["start_cursor"], "cursor-abc")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestListTasksNoFilter -v
```

Expected: FAIL — `AttributeError: 'NotionClient' object has no attribute 'list_tasks'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `NotionClient` in `config/skills/notion-tasks/scripts/notion_client.py`:

```python
    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        due_before: Optional[str] = None,
    ) -> list[dict]:
        filters = []
        if status is not None:
            filters.append({"property": "Status", "select": {"equals": status}})
        if priority is not None:
            filters.append({"property": "Priority", "select": {"equals": priority}})
        if due_before is not None:
            filters.append({"property": "Due", "date": {"on_or_before": due_before}})

        body: dict = {}
        if len(filters) == 1:
            body["filter"] = filters[0]
        elif len(filters) > 1:
            body["filter"] = {"and": filters}

        results = []
        cursor = None
        while True:
            if cursor:
                body["start_cursor"] = cursor
            status_code, data = self._request(
                "POST", f"/databases/{self._database_id}/query", body
            )
            if status_code != 200:
                raise RuntimeError(f"Notion API error {status_code}: {data}")
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return results
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestListTasksNoFilter -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/notion_client.py config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add list_tasks with pagination"
```

---

### Task 4: list_tasks (with filters)

**Group:** B (sequential after Task 3)

**Behavior being verified:** `list_tasks` produces correct Notion filter JSON for status, priority, and due_before — individually and combined.

**Interface under test:** `NotionClient.list_tasks`

**Files:**
- Modify: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
class TestListTasksFilters(unittest.TestCase):

    def _capture_body(self, **kwargs) -> dict:
        captured = []

        def side_effect(req):
            captured.append(req)
            return _make_response(_list_response([]))

        with patch.object(_OPENER, "open", side_effect=side_effect):
            client = _make_client()
            client.list_tasks(**kwargs)

        return json.loads(captured[0].data.decode("utf-8"))

    def test_status_filter_generates_correct_notion_filter(self):
        body = self._capture_body(status="Todo", priority=None, due_before=None)
        self.assertEqual(
            body["filter"],
            {"property": "Status", "select": {"equals": "Todo"}},
        )

    def test_priority_filter_generates_correct_notion_filter(self):
        body = self._capture_body(status=None, priority="High", due_before=None)
        self.assertEqual(
            body["filter"],
            {"property": "Priority", "select": {"equals": "High"}},
        )

    def test_due_before_filter_generates_correct_notion_filter(self):
        body = self._capture_body(status=None, priority=None, due_before="2026-04-17")
        self.assertEqual(
            body["filter"],
            {"property": "Due", "date": {"on_or_before": "2026-04-17"}},
        )

    def test_multiple_filters_combined_with_and(self):
        body = self._capture_body(status="Todo", priority="High", due_before=None)
        self.assertIn("and", body["filter"])
        and_clause = body["filter"]["and"]
        self.assertEqual(len(and_clause), 2)
        self.assertIn({"property": "Status", "select": {"equals": "Todo"}}, and_clause)
        self.assertIn({"property": "Priority", "select": {"equals": "High"}}, and_clause)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestListTasksFilters -v
```

Expected: FAIL — `AssertionError: 'filter' not found in body` (the current implementation exists but these specific filter assertions have not been run against it yet; if tests pass, the implementation from Task 3 already handles this correctly — see note below)

> Note: If all tests pass immediately, verify by temporarily breaking the filter logic (e.g., change `"equals"` to `"eq"`) and confirming tests fail, then restore.

- [ ] **Step 3: Implement the minimum to make the test pass**

No new implementation required — the filter logic was written in Task 3. If tests fail, check that `list_tasks` in `notion_client.py` matches the implementation from Task 3 exactly.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestListTasksFilters -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add list_tasks filter support"
```

---

### Task 5: update_task

**Group:** B (sequential after Task 4)

**Behavior being verified:** `update_task` sends only the fields provided as keyword arguments and returns the updated task dict.

**Interface under test:** `NotionClient.update_task`

**Files:**
- Modify: `config/skills/notion-tasks/scripts/notion_client.py`
- Modify: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
class TestUpdateTask(unittest.TestCase):

    def test_update_task_sends_only_provided_fields(self):
        updated_page = _page_fixture(page_id="page-abc123", title="Test task", status="In Progress")
        captured = []

        def capture_open(req):
            captured.append(req)
            return _make_response(updated_page)

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            result = client.update_task("page-abc123", status="In Progress")

        body = json.loads(captured[0].data.decode("utf-8"))
        props = body["properties"]
        self.assertIn("Status", props)
        self.assertEqual(props["Status"]["select"]["name"], "In Progress")
        self.assertNotIn("Name", props)
        self.assertNotIn("Due", props)
        self.assertNotIn("Priority", props)
        self.assertEqual(result["status"], "In Progress")

    def test_update_task_with_title_sends_title_property(self):
        updated_page = _page_fixture(page_id="page-abc123", title="New title")
        with patch.object(_OPENER, "open", return_value=_make_response(updated_page)) as mock_open:
            client = _make_client()
            client.update_task("page-abc123", title="New title")

        body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(
            body["properties"]["Name"]["title"][0]["text"]["content"], "New title"
        )

    def test_update_task_raises_on_api_error(self):
        error_payload = {"object": "error", "status": 500, "message": "internal error"}
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload, status=500)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.update_task("page-abc123", status="Done")
        self.assertIn("500", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestUpdateTask -v
```

Expected: FAIL — `AttributeError: 'NotionClient' object has no attribute 'update_task'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `NotionClient` in `config/skills/notion-tasks/scripts/notion_client.py`:

```python
    def update_task(self, page_id: str, **fields) -> dict:
        properties: dict = {}
        if "title" in fields:
            properties["Name"] = {"title": [{"text": {"content": fields["title"]}}]}
        if "status" in fields:
            properties["Status"] = {"select": {"name": fields["status"]}}
        if "due" in fields:
            if fields["due"]:
                properties["Due"] = {"date": {"start": fields["due"]}}
            else:
                properties["Due"] = {"date": None}
        if "priority" in fields:
            if fields["priority"]:
                properties["Priority"] = {"select": {"name": fields["priority"]}}
            else:
                properties["Priority"] = {"select": None}

        status, data = self._request("PATCH", f"/pages/{page_id}", {"properties": properties})
        if status == 404:
            raise RuntimeError(f"Task not found: {page_id}")
        if status != 200:
            raise RuntimeError(f"Notion API error {status}: {data}")
        return _extract_task(data)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestUpdateTask -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/notion_client.py config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add update_task"
```

---

### Task 6: complete_task

**Group:** B (sequential after Task 5)

**Behavior being verified:** `complete_task` sets Status to "Done" and returns the updated task dict.

**Interface under test:** `NotionClient.complete_task`

**Files:**
- Modify: `config/skills/notion-tasks/scripts/notion_client.py`
- Modify: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
class TestCompleteTask(unittest.TestCase):

    def test_complete_task_sets_status_to_done(self):
        done_page = _page_fixture(page_id="page-abc123", title="Test task", status="Done")
        captured = []

        def capture_open(req):
            captured.append(req)
            return _make_response(done_page)

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            result = client.complete_task("page-abc123")

        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(body["properties"]["Status"]["select"]["name"], "Done")
        self.assertEqual(result["status"], "Done")
        self.assertEqual(result["id"], "page-abc123")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestCompleteTask -v
```

Expected: FAIL — `AttributeError: 'NotionClient' object has no attribute 'complete_task'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `NotionClient` in `config/skills/notion-tasks/scripts/notion_client.py`:

```python
    def complete_task(self, page_id: str) -> dict:
        return self.update_task(page_id, status="Done")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestCompleteTask -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/notion_client.py config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add complete_task"
```

---

### Task 7: delete_task and 404 handling

**Group:** B (sequential after Task 6)

**Behavior being verified:** `delete_task` archives the Notion page; a 404 response raises `RuntimeError("Task not found: {page_id}")`.

**Interface under test:** `NotionClient.delete_task`

**Files:**
- Modify: `config/skills/notion-tasks/scripts/notion_client.py`
- Modify: `config/skills/notion-tasks/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_notion_client.py`:

```python
class TestDeleteTask(unittest.TestCase):

    def test_delete_task_sends_archived_true(self):
        archived_page = _page_fixture(page_id="page-abc123", title="Test task")
        captured = []

        def capture_open(req):
            captured.append(req)
            return _make_response(archived_page)

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.delete_task("page-abc123")

        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertTrue(body["archived"])
        self.assertIn("/pages/page-abc123", captured[0].full_url)

    def test_delete_task_raises_task_not_found_on_404(self):
        error_payload = {"object": "error", "status": 404, "message": "Could not find page"}
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload, status=404)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.delete_task("page-missing")
        self.assertIn("Task not found", str(ctx.exception))
        self.assertIn("page-missing", str(ctx.exception))

    def test_update_task_raises_task_not_found_on_404(self):
        error_payload = {"object": "error", "status": 404, "message": "Could not find page"}
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload, status=404)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.update_task("page-missing", status="Done")
        self.assertIn("Task not found", str(ctx.exception))
        self.assertIn("page-missing", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py::TestDeleteTask -v
```

Expected: FAIL — `AttributeError: 'NotionClient' object has no attribute 'delete_task'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `NotionClient` in `config/skills/notion-tasks/scripts/notion_client.py`:

```python
    def delete_task(self, page_id: str) -> None:
        status, data = self._request("PATCH", f"/pages/{page_id}", {"archived": True})
        if status == 404:
            raise RuntimeError(f"Task not found: {page_id}")
        if status != 200:
            raise RuntimeError(f"Notion API error {status}: {data}")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_notion_client.py -v
```

Expected: All tests PASS (run full suite to confirm nothing regressed)

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/notion_client.py config/skills/notion-tasks/tests/test_notion_client.py && git commit -m "feat(notion-tasks): add delete_task with 404 handling"
```

---

### Task 8: tasks.py create subcommand

**Group:** C (sequential after Group B)

**Behavior being verified:** `tasks.py create --title "Buy groceries"` calls `NotionClient.create_task` with the correct args and prints a confirmation line containing the page ID and title.

**Interface under test:** `tasks.main` CLI

**Files:**
- Create: `config/skills/notion-tasks/scripts/tasks.py`
- Create: `config/skills/notion-tasks/tests/test_tasks.py`

---

- [ ] **Step 1: Write the failing test**

Create `config/skills/notion-tasks/tests/test_tasks.py`:

```python
import sys
sys.path.insert(0, 'scripts')

import os
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import tasks


def _make_task(
    page_id: str = "page-abc",
    title: str = "Test task",
    status: str = "Todo",
    due: str | None = None,
    priority: str | None = None,
) -> dict:
    return {"id": page_id, "title": title, "status": status, "due": due, "priority": priority}


class TestCreateSubcommand(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_create_calls_create_task_with_title_and_optional_nones(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.create_task.return_value = _make_task(page_id="page-new", title="Buy groceries")

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            tasks.main(["create", "--title", "Buy groceries"])

        mock_client.create_task.assert_called_once_with(
            title="Buy groceries", due=None, priority=None
        )
        output = mock_out.getvalue()
        self.assertIn("page-new", output)
        self.assertIn("Buy groceries", output)

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_create_with_due_and_priority_passes_them_through(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.create_task.return_value = _make_task(
            page_id="page-xyz", title="Fix bug", due="2026-04-17", priority="High"
        )

        tasks.main(["create", "--title", "Fix bug", "--due", "2026-04-17", "--priority", "High"])

        mock_client.create_task.assert_called_once_with(
            title="Fix bug", due="2026-04-17", priority="High"
        )

    def test_create_without_title_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            tasks.main(["create"])
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_tasks.py::TestCreateSubcommand -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tasks'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `config/skills/notion-tasks/scripts/tasks.py`:

```python
"""
Notion task management CLI for Mahler.

Usage:
    python3 tasks.py create --title TITLE [--due YYYY-MM-DD] [--priority High|Medium|Low]
    python3 tasks.py list [--status STATUS] [--priority PRIORITY] [--due-before YYYY-MM-DD]
    python3 tasks.py update --id PAGE_ID [--title TITLE] [--status STATUS] [--due DATE] [--priority PRIORITY]
    python3 tasks.py complete --id PAGE_ID
    python3 tasks.py delete --id PAGE_ID
"""

import argparse
import os
import sys
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


def _format_task(task: dict) -> str:
    meta = []
    if task.get("status"):
        meta.append(f"status={task['status']}")
    if task.get("priority"):
        meta.append(f"priority={task['priority']}")
    if task.get("due"):
        meta.append(f"due={task['due']}")
    line = f"[{task['id']}] {task['title']}"
    if meta:
        line += f"\n  ({', '.join(meta)})"
    return line


def cmd_create(args: argparse.Namespace) -> None:
    client = _get_client()
    task = client.create_task(title=args.title, due=args.due, priority=args.priority)
    print(f"Created: {task['id']} — {task['title']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler Notion task manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--due", default=None)
    p_create.add_argument("--priority", choices=["High", "Medium", "Low"], default=None)

    args = parser.parse_args(argv)
    if args.command == "create":
        cmd_create(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_tasks.py::TestCreateSubcommand -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/tasks.py config/skills/notion-tasks/tests/test_tasks.py && git commit -m "feat(notion-tasks): add tasks.py create subcommand"
```

---

### Task 9: tasks.py list subcommand

**Group:** C (sequential after Task 8)

**Behavior being verified:** `tasks.py list` returns formatted output with page IDs visible, passes filter args through, and prints "No tasks found." on empty results.

**Interface under test:** `tasks.main` CLI

**Files:**
- Modify: `config/skills/notion-tasks/scripts/tasks.py`
- Modify: `config/skills/notion-tasks/tests/test_tasks.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_tasks.py`:

```python
class TestListSubcommand(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_list_output_includes_page_id_for_each_task(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.return_value = [
            _make_task(page_id="page-001", title="Task one", status="Todo"),
            _make_task(page_id="page-002", title="Task two", status="In Progress"),
        ]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            tasks.main(["list"])

        output = mock_out.getvalue()
        self.assertIn("page-001", output)
        self.assertIn("Task one", output)
        self.assertIn("page-002", output)
        self.assertIn("Task two", output)
        mock_client.list_tasks.assert_called_once_with(status=None, priority=None, due_before=None)

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_list_with_status_filter_passes_it_through(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.return_value = []

        tasks.main(["list", "--status", "Todo"])

        mock_client.list_tasks.assert_called_once_with(status="Todo", priority=None, due_before=None)

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_list_empty_result_prints_no_tasks_found(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.return_value = []

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            tasks.main(["list"])

        self.assertEqual(mock_out.getvalue().strip(), "No tasks found.")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_tasks.py::TestListSubcommand -v
```

Expected: FAIL — `SystemExit: 2` (argparse error — `list` subcommand not registered)

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `cmd_list` function and `list` subparser to `main()` in `config/skills/notion-tasks/scripts/tasks.py`:

```python
def cmd_list(args: argparse.Namespace) -> None:
    client = _get_client()
    task_list = client.list_tasks(
        status=args.status,
        priority=args.priority,
        due_before=args.due_before,
    )
    if not task_list:
        print("No tasks found.")
        return
    for task in task_list:
        print(_format_task(task))
```

In `main()`, add after the `p_create` block and update the dispatch:

```python
    p_list = sub.add_parser("list")
    p_list.add_argument("--status", choices=["Todo", "In Progress", "Done"], default=None)
    p_list.add_argument("--priority", choices=["High", "Medium", "Low"], default=None)
    p_list.add_argument("--due-before", dest="due_before", default=None)
```

Update the dispatch block in `main()`:

```python
    dispatch = {"create": cmd_create, "list": cmd_list}
    dispatch[args.command](args)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_tasks.py::TestListSubcommand -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/tasks.py config/skills/notion-tasks/tests/test_tasks.py && git commit -m "feat(notion-tasks): add tasks.py list subcommand"
```

---

### Task 10: tasks.py update, complete, and delete subcommands

**Group:** C (sequential after Task 9)

**Behavior being verified:** `update`, `complete`, and `delete` subcommands dispatch to the correct `NotionClient` methods with the correct arguments.

**Interface under test:** `tasks.main` CLI

**Files:**
- Modify: `config/skills/notion-tasks/scripts/tasks.py`
- Modify: `config/skills/notion-tasks/tests/test_tasks.py`

---

- [ ] **Step 1: Write the failing test**

Append to `config/skills/notion-tasks/tests/test_tasks.py`:

```python
class TestUpdateCompleteDeleteSubcommands(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_update_calls_update_task_with_provided_fields_only(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.update_task.return_value = _make_task(
            page_id="page-abc", title="Test task", status="In Progress"
        )

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            tasks.main(["update", "--id", "page-abc", "--status", "In Progress"])

        mock_client.update_task.assert_called_once_with("page-abc", status="In Progress")
        output = mock_out.getvalue()
        self.assertIn("page-abc", output)

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_complete_calls_complete_task_with_page_id(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.complete_task.return_value = _make_task(
            page_id="page-abc", title="Test task", status="Done"
        )

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            tasks.main(["complete", "--id", "page-abc"])

        mock_client.complete_task.assert_called_once_with("page-abc")
        self.assertIn("page-abc", mock_out.getvalue())

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("tasks.NotionClient")
    def test_delete_calls_delete_task_and_prints_confirmation(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.delete_task.return_value = None

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            tasks.main(["delete", "--id", "page-abc"])

        mock_client.delete_task.assert_called_once_with("page-abc")
        self.assertIn("page-abc", mock_out.getvalue())

    def test_update_without_id_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            tasks.main(["update", "--status", "Done"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_complete_without_id_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            tasks.main(["complete"])
        self.assertNotEqual(ctx.exception.code, 0)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_tasks.py::TestUpdateCompleteDeleteSubcommands -v
```

Expected: FAIL — `SystemExit: 2` (`update`, `complete`, `delete` subcommands not registered)

- [ ] **Step 3: Implement the minimum to make the test pass**

Add handler functions and subparsers to `config/skills/notion-tasks/scripts/tasks.py`.

Add these three functions before `main()`:

```python
def cmd_update(args: argparse.Namespace) -> None:
    client = _get_client()
    fields = {}
    if args.title is not None:
        fields["title"] = args.title
    if args.status is not None:
        fields["status"] = args.status
    if args.due is not None:
        fields["due"] = args.due
    if args.priority is not None:
        fields["priority"] = args.priority
    task = client.update_task(args.id, **fields)
    print(f"Updated: {task['id']} — {task['title']}")


def cmd_complete(args: argparse.Namespace) -> None:
    client = _get_client()
    task = client.complete_task(args.id)
    print(f"Completed: {task['id']} — {task['title']}")


def cmd_delete(args: argparse.Namespace) -> None:
    client = _get_client()
    client.delete_task(args.id)
    print(f"Deleted: {args.id}")
```

In `main()`, add after the `p_list` block:

```python
    p_update = sub.add_parser("update")
    p_update.add_argument("--id", required=True)
    p_update.add_argument("--title", default=None)
    p_update.add_argument("--status", choices=["Todo", "In Progress", "Done"], default=None)
    p_update.add_argument("--due", default=None)
    p_update.add_argument("--priority", choices=["High", "Medium", "Low"], default=None)

    p_complete = sub.add_parser("complete")
    p_complete.add_argument("--id", required=True)

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--id", required=True)
```

Update the dispatch block in `main()`:

```python
    dispatch = {
        "create": cmd_create,
        "list": cmd_list,
        "update": cmd_update,
        "complete": cmd_complete,
        "delete": cmd_delete,
    }
    dispatch[args.command](args)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/test_tasks.py -v
```

Expected: All tests PASS (run full suite to confirm nothing regressed)

- [ ] **Step 5: Commit**

```bash
git add config/skills/notion-tasks/scripts/tasks.py config/skills/notion-tasks/tests/test_tasks.py && git commit -m "feat(notion-tasks): add tasks.py update, complete, delete subcommands"
```

---

### Task 11: SKILL.md and Dockerfile integration

**Group:** D (sequential after Group C)

**Behavior being verified:** Hermes can read the skill and invoke the correct CLI command; Dockerfile builds with the skill copied.

**Interface under test:** N/A — artifact creation only. No automated test; deployment to Fly.io verifies this.

**Files:**
- Create: `config/skills/notion-tasks/SKILL.md`
- Modify: `Dockerfile`

---

- [ ] **Step 1: Create SKILL.md**

Create `config/skills/notion-tasks/SKILL.md`:

```markdown
---
name: notion-tasks
description: Create, list, update, complete, and delete tasks in the user's Notion task database. Full CRUD for personal task management.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [tasks, todo, notion, productivity]
    related_skills: []
---

## When to use

- When the user asks to add, create, or log a new task or todo
- When the user asks what tasks or todos they have, wants to see their list, or asks "what do I need to do?"
- When the user asks to update, modify, or change a task's title, status, priority, or due date
- When the user says a task is done, finished, or complete
- When the user asks to remove or delete a task

## Prerequisites

| Variable | Purpose |
|---|---|
| `NOTION_API_TOKEN` | Notion internal integration token |
| `NOTION_DATABASE_ID` | ID of the Notion tasks database |

Both must be set as Fly.io secrets. The script raises `RuntimeError` if either is missing.

## Date handling

All dates must be in ISO 8601 format: `YYYY-MM-DD`. Before invoking any command, convert relative dates from the user's message to absolute dates using today's date. Example: "Friday" → the date of the upcoming Friday, "next week" → the Monday of next week.

## Operations

### Create a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py create \
  --title "TITLE" \
  [--due YYYY-MM-DD] \
  [--priority High|Medium|Low]
```

If the user does not state a priority, infer it:
- Deadlines within 2 days, blocking other work, or urgent language → `High`
- Clear action items without urgency → `Medium`
- Nice-to-have, someday, or low-stakes tasks → `Low`

### List tasks

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  [--status Todo|"In Progress"|Done] \
  [--priority High|Medium|Low] \
  [--due-before YYYY-MM-DD]
```

Output format per task:
```
[page-id] Task title
  (status=Todo, priority=High, due=2026-04-17)
```

The page ID on each task's first line is required for follow-up update, complete, or delete operations. If the user asks to act on a task and you do not have its page ID, run `list` first.

### Update a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py update \
  --id PAGE_ID \
  [--title "NEW TITLE"] \
  [--status Todo|"In Progress"|Done] \
  [--due YYYY-MM-DD] \
  [--priority High|Medium|Low]
```

Include only the flags for fields the user wants to change.

### Complete a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py complete --id PAGE_ID
```

Use this when the user says a task is done, finished, or complete. Sets status to Done.

### Delete a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py delete --id PAGE_ID
```

**Always confirm with the user before running delete.** Ask: "Are you sure you want to delete [task title]?" and only proceed if they confirm. Deleted tasks are archived in Notion (recoverable from the Notion UI) but treated as permanently removed in this interface.

## Output

Each command prints a single confirmation line to stdout:
- `create` → `Created: {page_id} — {title}`
- `list` → one formatted entry per task, or `No tasks found.`
- `update` → `Updated: {page_id} — {title}`
- `complete` → `Completed: {page_id} — {title}`
- `delete` → `Deleted: {page_id}`

Any failure raises `RuntimeError` and exits non-zero. Surface the error message to the user directly.
```

- [ ] **Step 2: Add COPY line to Dockerfile**

In `Dockerfile`, after the existing skill COPY lines, add:

```dockerfile
COPY --chown=hermes:hermes config/skills/notion-tasks /home/hermes/.hermes/skills/notion-tasks
```

The correct location is after this block:
```
COPY --chown=hermes:hermes config/skills/morning-brief /home/hermes/.hermes/skills/morning-brief
```

- [ ] **Step 3: Run the full test suite to confirm nothing broken**

```bash
cd config/skills/notion-tasks && python3 -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add config/skills/notion-tasks/SKILL.md Dockerfile && git commit -m "feat(notion-tasks): add SKILL.md and Dockerfile integration"
```

---

## Challenge Review

### CEO Pass

**Premise:** Sound. No Notion task management exists today — every task requires leaving Discord. This is a real daily friction point, not a speculative feature. The "Hermes orchestrates NL → CLI" decision is justified: Hermes is already an LLM doing exactly this for other skills.

**Scope:** Tight. 6 files, 2 new scripts, no new services. The decision to exclude subtasks, calendar sync, and recurring tasks is correct — those are Phase 4/5 territory.

**12-Month Alignment:**
```
CURRENT STATE                THIS PLAN                    12-MONTH IDEAL
No Notion integration  →  Tasks CRUD via Discord  →  Full Notion surface
                                                        (tasks + wiki + CRM)
```

This plan lays the Notion client foundation (auth pattern, HTTP opener, response parsing) that wiki and CRM skills will reuse. Good alignment — no tech debt introduced.

**Alternatives:** The spec documents the NL-in-script vs. Hermes-orchestrates tradeoff explicitly. Alternative rejected on sound grounds (extra LLM hop, no complexity gain). Covered.

---

### Engineering Pass

**Architecture:** Clean. `notion_client.py` owns all HTTP, serialization, and parsing. `tasks.py` is a pure dispatch layer. Data flow is straightforward:

```
Discord NL → Hermes reads SKILL.md → tasks.py main(argv)
  → _get_client() reads env vars
  → NotionClient._request() → HTTPS → Notion API
  → _extract_task() → flat dict → stdout
  → Hermes reads stdout → Discord reply
```

Error path: RuntimeError propagates from `NotionClient` → `cmd_*` → `main()` → Python traceback on stderr. Hermes reads exit code and stderr to surface error to user.

**Module Depth:**
- `notion_client.py` — DEEP: 5-method public interface hides block format JSON, pagination cursor loop, HTTPS opener, response parsing, HTTP status mapping.
- `tasks.py` — SHALLOW by design: justified, depth lives in `notion_client.py`.

**Deviation from established `d1_client.py` pattern — confirmed RISK:**

`d1_client.query()` checks `status != 200` BEFORE calling `json.loads(raw)`, and raises `RuntimeError` with the raw bytes on non-200. The plan's `_request` does the opposite — it ALWAYS calls `json.loads(raw)` on line 198 regardless of status, then returns `(status, dict)` to the caller. If Notion's CDN returns a 502 or 503 with an HTML error page, `json.loads(raw)` raises `json.JSONDecodeError` — an uncaught exception that propagates as a confusing internal error rather than a clean `RuntimeError("Notion API error 502: ...")`.

Fix: structure `_request` like `d1_client.query()`:
```python
with _OPENER.open(req) as resp:
    status = resp.status
    raw = resp.read()
if status not in (200,):
    raise RuntimeError(f"Notion API error {status}: {raw.decode('utf-8', errors='replace')}")
return json.loads(raw)
```

This means callers no longer need to check the returned status — `_request` either succeeds or raises. The 404 case becomes a caught pattern: callers catch `RuntimeError` and re-raise with "Task not found". Alternatively, keep the `(status, dict)` return and add a try/except around `json.loads`. Either fix works; the first is cleaner and matches the existing codebase.

**Test Philosophy:**

All tests mock at the correct boundaries: `_OPENER.open` for `notion_client.py` tests (HTTP boundary), `tasks.NotionClient` for `tasks.py` tests (class boundary). No internal collaborators mocked. Behavior-focused assertions. Philosophy is sound.

**Vertical Slice Audit:**

Tasks 1, 2, 3, 5, 6, 7, 8, 9, 10: clean one-test → one-impl → one-commit slices.

Task 4 is not a genuine vertical slice. The plan's own note states: "No new implementation required — the filter logic was written in Task 3." This means Task 4 writes tests for code that already exists. The plan instructs the build agent to run Step 2 ("verify it FAILS") with an expected failure message, then immediately notes the tests will likely PASS. The build agent's TDD discipline check — "if the test PASSES without the implementation, the test is wrong: rewrite" — will fire incorrectly on Task 4 and cause the agent to either skip valid tests or rewrite them. This is a real execution risk.

**Test Coverage Gaps:**

```
[+] notion_client.py
    ├── __init__()
    │   ├── [TESTED ★★] empty token raises — Task 1
    │   ├── [TESTED ★★] empty database_id raises — Task 1
    │   └── [TESTED ★★] valid init, no network calls — Task 1
    ├── _request()
    │   ├── [GAP] json.JSONDecodeError on non-JSON response — no test
    │   └── (tested indirectly through all method tests)
    ├── create_task()
    │   ├── [TESTED ★★★] correct payload, returns flat dict — Task 2
    │   ├── [TESTED ★★★] due/priority included when provided — Task 2
    │   └── [TESTED ★★] non-200 raises RuntimeError — Task 2
    ├── list_tasks()
    │   ├── [TESTED ★★★] no filters, returns all — Task 3
    │   ├── [TESTED ★★★] no filter key in body when no filters — Task 3
    │   ├── [TESTED ★★★] follows pagination cursor — Task 3
    │   ├── [TESTED ★★★] status filter correct JSON — Task 4
    │   ├── [TESTED ★★★] priority filter correct JSON — Task 4
    │   ├── [TESTED ★★★] due_before filter correct JSON — Task 4
    │   └── [TESTED ★★★] multiple filters use "and" — Task 4
    ├── update_task()
    │   ├── [TESTED ★★★] only provided fields sent — Task 5
    │   ├── [TESTED ★★] title field maps to Name property — Task 5
    │   ├── [TESTED ★★] non-200 raises RuntimeError — Task 5
    │   └── [TESTED ★★★] 404 raises "Task not found" — Task 7
    ├── complete_task()
    │   └── [TESTED ★★★] sets Status to Done — Task 6
    └── delete_task()
        ├── [TESTED ★★★] sends archived=true, correct URL — Task 7
        └── [TESTED ★★★] 404 raises "Task not found" — Task 7

[+] tasks.py
    ├── _get_client()
    │   ├── [GAP] NOTION_API_TOKEN missing → RuntimeError propagates as traceback
    │   └── [GAP] NOTION_DATABASE_ID missing → same
    ├── cmd_create()
    │   ├── [TESTED ★★★] calls create_task with correct args — Task 8
    │   ├── [TESTED ★★] due/priority passed through — Task 8
    │   └── [TESTED ★★] missing --title exits nonzero — Task 8
    ├── cmd_list()
    │   ├── [TESTED ★★★] output includes page IDs — Task 9
    │   ├── [TESTED ★★★] status filter passed through — Task 9
    │   └── [TESTED ★★★] empty result prints "No tasks found." — Task 9
    ├── cmd_update()
    │   ├── [TESTED ★★★] provided fields only passed — Task 10
    │   └── [TESTED ★★] missing --id exits nonzero — Task 10
    ├── cmd_complete()
    │   ├── [TESTED ★★★] calls complete_task with ID — Task 10
    │   └── [TESTED ★★] missing --id exits nonzero — Task 10
    └── cmd_delete()
        ├── [TESTED ★★★] calls delete_task, prints confirmation — Task 10
        └── [GAP] missing --id — not tested
```

The `_get_client()` env-var-missing gap is low severity — the error propagates to Hermes as a non-zero exit with a `RuntimeError` message on stderr, which Hermes surfaces to the user. Not a silent failure.

**Failure Modes:**

All failures propagate as `RuntimeError` and exit non-zero. Hermes reads stderr and exit code. No silent failures. The one concern is the `json.JSONDecodeError` on non-JSON responses (documented above as RISK).

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| Notion API returns HTTP 200 (not 201) for POST /pages | SAFE | Notion API v1 docs confirm 200 for page creation |
| `urllib.request.Request.full_url` attribute exists | SAFE | Python 3.4+ property, Dockerfile uses Python 3.11 |
| Mocking `_OPENER.open` via `patch.object` captures the Request object as first positional arg | SAFE | Verified against `d1_client.py` test pattern which does the same |
| `patch("tasks.NotionClient")` patches the right binding | SAFE | `tasks.py` does `from notion_client import NotionClient` — `tasks.NotionClient` is the correct patch target |
| Notion API 404 response body is valid JSON | VALIDATE | Notion's error responses are always JSON per their docs, but a proxy-level 404 may not be |
| Tests run from `config/skills/notion-tasks/` (not from repo root) | SAFE | All `cd config/skills/notion-tasks && python3 -m pytest` commands enforce this |
| `_supplement_env_from_hermes()` at import time doesn't break tests | SAFE | Function handles missing file gracefully; test `@patch.dict` overrides any injected values |
| `OpenerDirector` without `HTTPDefaultErrorHandler` returns response on non-200 (not raises) | SAFE | Verified against `d1_client.py` which uses the same opener construction and manually checks status |

---

### Summary

```
[RISK] (confidence: 9/10) — _request() calls json.loads(raw) unconditionally before
       checking status. d1_client.py checks status != 200 first and raises RuntimeError
       with raw bytes on non-200. A Notion 502/503 with HTML body will raise
       json.JSONDecodeError instead of RuntimeError, violating the project's explicit
       exception handling convention. Fix: check status before json.loads, or wrap
       json.loads in try/except and re-raise as RuntimeError. Affects Tasks 1-7.

[RISK] (confidence: 9/10) — Task 4 "Step 2: verify it FAILS" contradicts the plan's
       own note that the filter implementation already exists from Task 3. The build
       agent's TDD discipline check will fire on passing tests and may incorrectly
       rewrite valid tests. Fix: restructure Task 4 as a pure test-addition step,
       remove the "verify it FAILS" instruction, and note explicitly that this task
       adds coverage for already-implemented filter paths.

[OBS] — _get_client() env-var-missing path in tasks.py is not tested. RuntimeError
       propagates as a Python traceback on stderr. Hermes surfaces this as an error
       to the user (non-silent). Low severity.

[OBS] — No timeout on _OPENER.open() calls. Consistent with d1_client.py and
       gmail_client.py patterns. Notion API could hang indefinitely. Acceptable given
       Hermes's own timeout handling.
```

[RISK] count: 2
[OBS] count: 2

---

## Post-build verification

Before deploying, set the two new secrets:

```bash
flyctl secrets set NOTION_API_TOKEN=<your-notion-integration-token> NOTION_DATABASE_ID=<your-database-id>
```

The Notion database must be created manually in Notion with these exact property names and types before deploying:
- `Name` — Title property
- `Status` — Select property with options: `Todo`, `In Progress`, `Done`
- `Due` — Date property
- `Priority` — Select property with options: `High`, `Medium`, `Low`

The integration token must have read/write access to the database (share the database with the integration in Notion's Share menu).

---

VERDICT: PROCEED_WITH_CAUTION — Fix `_request()` to check status before calling `json.loads(raw)` (match the `d1_client.py` pattern). Reword Task 4's Step 2 to remove the misleading "verify it FAILS" instruction. Both are small, in-task corrections the build agent can apply as it works.
