# E5 Relationship CRM Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Track professional and personal relationships via Discord commands, with `last_contact` auto-updated from Google Calendar attendee emails and open commitments surfaced as Notion tasks.
**Spec:** docs/specs/2026-04-19-e5-relationship-crm-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

```
Group A (parallel): Task 1, Task 7
Group B (sequential, depends on Task 1): Task 2 → Task 3 → Task 4 → Task 5 → Task 6
Group C (sequential, depends on Group B + Task 7): Task 8 → Task 9 → Task 10 → Task 11
Group D (parallel, depends on Group C): Task 12, Task 13, Task 14
```

---

### Task 1: d1_client.py — skeleton, ensure_table, upsert_contact

**Group:** A (parallel with Task 7)

**Behavior being verified:** A contact can be added (or re-added by email) without error; the correct INSERT … ON CONFLICT SQL and params are sent to D1.

**Interface under test:** `D1Client.upsert_contact(name, email, type, context)`

**Files:**
- Create: `assistant/config/skills/relationship-manager/scripts/d1_client.py`
- Create: `assistant/config/skills/relationship-manager/tests/conftest.py`
- Create: `assistant/config/skills/relationship-manager/tests/test_d1_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/relationship-manager/tests/conftest.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
```

```python
# assistant/config/skills/relationship-manager/tests/test_d1_client.py
import json
import pytest
from unittest.mock import patch, MagicMock


def _make_d1_response(rows=None, status=200):
    body = {"success": True, "errors": [], "result": [{"results": rows or []}]}
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_upsert_contact_sends_correct_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        client.upsert_contact("Alice Chen", "alice@example.com", "professional", "Works at Sequoia")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "INSERT INTO contacts" in body["sql"]
        assert "ON CONFLICT" in body["sql"]
        assert body["params"] == ["Alice Chen", "alice@example.com", "professional", "Works at Sequoia"]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_upsert_contact_sends_correct_sql -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'd1_client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/relationship-manager/scripts/d1_client.py
import json
import re
import ssl
import urllib.request
from typing import Optional

_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"

_ALLOWED_UPDATE_FIELDS = frozenset({"name", "email", "type", "context"})


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
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
        if status != 200:
            raise RuntimeError(f"D1 API error {status}: {raw.decode('utf-8', errors='replace')}")
        data = json.loads(raw)
        if not data.get("success") or data.get("errors"):
            raise RuntimeError(f"D1 query failed: {data.get('errors', [])}")
        results = data.get("result", [])
        if not results:
            return []
        return results[0].get("results") or []

    def ensure_table(self) -> None:
        self.query("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL CHECK(type IN ('professional', 'personal')),
                last_contact TEXT,
                context TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

    def upsert_contact(self, name: str, email: str, type: str, context: str) -> None:
        self.query(
            """
            INSERT INTO contacts (name, email, type, context, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                context = excluded.context
            """,
            [name, email, type, context],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_upsert_contact_sends_correct_sql -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/d1_client.py \
        assistant/config/skills/relationship-manager/tests/conftest.py \
        assistant/config/skills/relationship-manager/tests/test_d1_client.py && \
git commit -m "feat(e5): add d1_client skeleton with ensure_table and upsert_contact"
```

---

### Task 2: d1_client — get_contact

**Group:** B (sequential after Task 1)

**Behavior being verified:** `get_contact` returns the contact row for a known name; raises `RuntimeError` when the name is not found.

**Interface under test:** `D1Client.get_contact(name) → dict`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/d1_client.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_d1_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_d1_client.py

def test_get_contact_returns_row():
    row = {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
           "type": "professional", "last_contact": None, "context": "Works at Sequoia",
           "created_at": "2026-04-19"}
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=[row])
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        result = client.get_contact("Alice Chen")
        assert result["name"] == "Alice Chen"
        assert result["email"] == "alice@example.com"
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "WHERE lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["Alice Chen"]


def test_get_contact_raises_when_not_found():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=[])
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        with pytest.raises(RuntimeError, match="Contact not found"):
            client.get_contact("Nobody")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_get_contact_returns_row tests/test_d1_client.py::test_get_contact_raises_when_not_found -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'get_contact'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `d1_client.py`:

```python
    def get_contact(self, name: str) -> dict:
        rows = self.query(
            "SELECT id, name, email, type, last_contact, context, created_at "
            "FROM contacts WHERE lower(name) = lower(?) LIMIT 1",
            [name],
        )
        if not rows:
            raise RuntimeError(f"Contact not found: {name!r}")
        return rows[0]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_get_contact_returns_row tests/test_d1_client.py::test_get_contact_raises_when_not_found -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/d1_client.py \
        assistant/config/skills/relationship-manager/tests/test_d1_client.py && \
git commit -m "feat(e5): add d1_client.get_contact"
```

---

### Task 3: d1_client — list_contacts

**Group:** B (sequential after Task 2)

**Behavior being verified:** `list_contacts` returns all contacts when called without arguments; passes a type filter to D1 when `type` is specified.

**Interface under test:** `D1Client.list_contacts(type=None) → list[dict]`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/d1_client.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_d1_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_d1_client.py

def test_list_contacts_returns_all():
    rows = [
        {"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
         "last_contact": None, "context": "", "created_at": "2026-04-19"},
        {"id": 2, "name": "Bob Smith", "email": "b@x.com", "type": "personal",
         "last_contact": None, "context": "", "created_at": "2026-04-19"},
    ]
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=rows)
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        result = client.list_contacts()
        assert len(result) == 2
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "WHERE type" not in body["sql"]


def test_list_contacts_filters_by_type():
    rows = [{"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
             "last_contact": None, "context": "", "created_at": "2026-04-19"}]
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=rows)
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        result = client.list_contacts(type="professional")
        assert len(result) == 1
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "WHERE type = ?" in body["sql"]
        assert body["params"] == ["professional"]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_list_contacts_returns_all tests/test_d1_client.py::test_list_contacts_filters_by_type -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'list_contacts'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `d1_client.py`:

```python
    def list_contacts(self, type: Optional[str] = None) -> list[dict]:
        if type is not None:
            return self.query(
                "SELECT id, name, email, type, last_contact, context, created_at "
                "FROM contacts WHERE type = ? ORDER BY name",
                [type],
            )
        return self.query(
            "SELECT id, name, email, type, last_contact, context, created_at "
            "FROM contacts ORDER BY name"
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_list_contacts_returns_all tests/test_d1_client.py::test_list_contacts_filters_by_type -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/d1_client.py \
        assistant/config/skills/relationship-manager/tests/test_d1_client.py && \
git commit -m "feat(e5): add d1_client.list_contacts"
```

---

### Task 4: d1_client — touch_last_contact

**Group:** B (sequential after Task 3)

**Behavior being verified:** `touch_last_contact` sends an UPDATE SQL that sets `last_contact` to the given date for the named contact.

**Interface under test:** `D1Client.touch_last_contact(name, date) → None`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/d1_client.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_d1_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_d1_client.py

def test_touch_last_contact_sends_update_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        client.touch_last_contact("Alice Chen", "2026-04-19")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "UPDATE contacts SET last_contact = ?" in body["sql"]
        assert "lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["2026-04-19", "Alice Chen"]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_touch_last_contact_sends_update_sql -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'touch_last_contact'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `d1_client.py`:

```python
    def touch_last_contact(self, name: str, date: str) -> None:
        self.query(
            "UPDATE contacts SET last_contact = ? WHERE lower(name) = lower(?)",
            [date, name],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_touch_last_contact_sends_update_sql -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/d1_client.py \
        assistant/config/skills/relationship-manager/tests/test_d1_client.py && \
git commit -m "feat(e5): add d1_client.touch_last_contact"
```

---

### Task 5: d1_client — update_contact

**Group:** B (sequential after Task 4)

**Behavior being verified:** `update_contact` sends an UPDATE SQL for the specified allowlisted field; raises `ValueError` for unknown fields.

**Interface under test:** `D1Client.update_contact(name, field, value) → None`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/d1_client.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_d1_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_d1_client.py

def test_update_contact_sends_correct_field_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        client.update_contact("Alice Chen", "context", "Partner at Sequoia now")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "UPDATE contacts SET context = ?" in body["sql"]
        assert "lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["Partner at Sequoia now", "Alice Chen"]


def test_update_contact_rejects_unknown_field():
    from d1_client import D1Client
    client = D1Client("acct1", "db1", "tok1")
    with pytest.raises(ValueError, match="Cannot update field"):
        client.update_contact("Alice Chen", "password", "hack")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_update_contact_sends_correct_field_sql tests/test_d1_client.py::test_update_contact_rejects_unknown_field -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'update_contact'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `d1_client.py`:

```python
    def update_contact(self, name: str, field: str, value: str) -> None:
        if field not in _ALLOWED_UPDATE_FIELDS:
            raise ValueError(f"Cannot update field {field!r}. Allowed: {sorted(_ALLOWED_UPDATE_FIELDS)}")
        self.query(
            f"UPDATE contacts SET {field} = ? WHERE lower(name) = lower(?)",
            [value, name],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_update_contact_sends_correct_field_sql tests/test_d1_client.py::test_update_contact_rejects_unknown_field -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/d1_client.py \
        assistant/config/skills/relationship-manager/tests/test_d1_client.py && \
git commit -m "feat(e5): add d1_client.update_contact with field allowlist"
```

---

### Task 6: d1_client — delete_contact

**Group:** B (sequential after Task 5)

**Behavior being verified:** `delete_contact` sends a DELETE SQL for the named contact.

**Interface under test:** `D1Client.delete_contact(name) → None`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/d1_client.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_d1_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_d1_client.py

def test_delete_contact_sends_delete_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        client.delete_contact("Alice Chen")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "DELETE FROM contacts" in body["sql"]
        assert "lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["Alice Chen"]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py::test_delete_contact_sends_delete_sql -v
```

Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'delete_contact'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `D1Client` in `d1_client.py`:

```python
    def delete_contact(self, name: str) -> None:
        self.query(
            "DELETE FROM contacts WHERE lower(name) = lower(?)",
            [name],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_d1_client.py -v
```

Expected: PASS (all 9 d1_client tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/d1_client.py \
        assistant/config/skills/relationship-manager/tests/test_d1_client.py && \
git commit -m "feat(e5): add d1_client.delete_contact — d1_client complete"
```

---

### Task 7: notion_client — list_tasks_for_contact

**Group:** A (parallel with Task 1)

**Behavior being verified:** `list_tasks_for_contact` queries Notion with a title-contains + status-not-Done filter and returns normalized task dicts.

**Interface under test:** `NotionClient.list_tasks_for_contact(name) → list[dict]`

**Files:**
- Create: `assistant/config/skills/relationship-manager/scripts/notion_client.py`
- Create: `assistant/config/skills/relationship-manager/tests/test_notion_client.py`

---

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/relationship-manager/tests/test_notion_client.py
import json
import pytest
from unittest.mock import patch, MagicMock


def _make_notion_response(results, has_more=False, next_cursor=None, status=200):
    body = {"results": results, "has_more": has_more}
    if next_cursor:
        body["next_cursor"] = next_cursor
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _page_fixture(title="[Alice Chen] Send IC memo", status="In progress", due="2026-04-25", priority="High"):
    return {
        "id": "page-abc123",
        "properties": {
            "Task name": {"title": [{"plain_text": title}]},
            "Status": {"status": {"name": status}},
            "Due date": {"date": {"start": due} if due else None},
            "Priority": {"select": {"name": priority} if priority else None},
        },
    }


def test_list_tasks_for_contact_sends_correct_filter():
    with patch("notion_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_notion_response(results=[_page_fixture()])
        from notion_client import NotionClient
        client = NotionClient(api_token="ntok", database_id="ndb1")
        tasks = client.list_tasks_for_contact("Alice Chen")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "[Alice Chen] Send IC memo"
        assert tasks[0]["id"] == "page-abc123"
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        filt = body["filter"]
        assert filt["and"][0]["title"]["contains"] == "[Alice Chen]"
        assert filt["and"][1]["status"]["does_not_equal"] == "Done"


def test_list_tasks_for_contact_returns_empty_when_none():
    with patch("notion_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_notion_response(results=[])
        from notion_client import NotionClient
        client = NotionClient(api_token="ntok", database_id="ndb1")
        tasks = client.list_tasks_for_contact("Unknown Person")
        assert tasks == []


def test_list_tasks_for_contact_paginates():
    page1 = _page_fixture(title="[Alice Chen] Task 1")
    page2 = _page_fixture(title="[Alice Chen] Task 2")
    with patch("notion_client._OPENER") as mock_opener:
        mock_opener.open.side_effect = [
            _make_notion_response(results=[page1], has_more=True, next_cursor="cur1"),
            _make_notion_response(results=[page2], has_more=False),
        ]
        from notion_client import NotionClient
        client = NotionClient(api_token="ntok", database_id="ndb1")
        tasks = client.list_tasks_for_contact("Alice Chen")
        assert len(tasks) == 2
        assert tasks[0]["title"] == "[Alice Chen] Task 1"
        assert tasks[1]["title"] == "[Alice Chen] Task 2"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_notion_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'notion_client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/relationship-manager/scripts/notion_client.py
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
    return {"id": page["id"], "title": title, "status": status, "due": due, "priority": priority}


class NotionClient:
    def __init__(self, api_token: str, database_id: str):
        self.api_token = api_token
        self.database_id = database_id

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Notion-Version": _NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
        if status not in (200, 201):
            raise RuntimeError(f"Notion API error {status}: {raw.decode('utf-8', errors='replace')}")
        return json.loads(raw)

    def list_tasks_for_contact(self, name: str) -> list[dict]:
        results = []
        cursor = None
        while True:
            body: dict = {
                "filter": {
                    "and": [
                        {"property": "Task name", "title": {"contains": f"[{name}]"}},
                        {"property": "Status", "status": {"does_not_equal": "Done"}},
                    ]
                },
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{self.database_id}/query", body)
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return results
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_notion_client.py -v
```

Expected: PASS (all 3 notion_client tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/notion_client.py \
        assistant/config/skills/relationship-manager/tests/test_notion_client.py && \
git commit -m "feat(e5): add notion_client with list_tasks_for_contact"
```

---

### Task 8: contacts.py — add command

**Group:** C (sequential, depends on Group B + Task 7)

**Behavior being verified:** The `add` subcommand calls `ensure_table` + `upsert_contact` on D1Client and prints `Added: {name} ({type})`.

**Interface under test:** `contacts.main(["add", "--name", ..., "--email", ..., "--type", ..., "--context", ...])`

**Files:**
- Create: `assistant/config/skills/relationship-manager/scripts/contacts.py`
- Create: `assistant/config/skills/relationship-manager/tests/test_contacts.py`

---

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/relationship-manager/tests/test_contacts.py
import io
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

_ENV = {
    "CF_ACCOUNT_ID": "acct1",
    "CF_D1_DATABASE_ID": "db1",
    "CF_API_TOKEN": "tok1",
    "NOTION_API_TOKEN": "ntok",
    "NOTION_DATABASE_ID": "ndb1",
    "GMAIL_CLIENT_ID": "cid",
    "GMAIL_CLIENT_SECRET": "csec",
    "GMAIL_REFRESH_TOKEN": "rtok",
}


def test_add_command_creates_contact_and_prints_confirmation():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main([
                    "add",
                    "--name", "Alice Chen",
                    "--email", "alice@example.com",
                    "--type", "professional",
                    "--context", "Works at Sequoia",
                ])
            mock_d1.ensure_table.assert_called_once()
            mock_d1.upsert_contact.assert_called_once_with(
                "Alice Chen", "alice@example.com", "professional", "Works at Sequoia"
            )
            assert "Added: Alice Chen (professional)" in out.getvalue()


def test_add_command_context_defaults_to_empty_string():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main([
                    "add",
                    "--name", "Bob Smith",
                    "--email", "bob@example.com",
                    "--type", "personal",
                ])
            mock_d1.upsert_contact.assert_called_once_with(
                "Bob Smith", "bob@example.com", "personal", ""
            )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_add_command_creates_contact_and_prints_confirmation tests/test_contacts.py::test_add_command_context_defaults_to_empty_string -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'contacts'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/relationship-manager/scripts/contacts.py
import argparse
import os
import sys
from datetime import date

from d1_client import D1Client
from notion_client import NotionClient
import gcal_client


def _supplement_env_from_hermes() -> None:
    env_path = os.path.expanduser("~/.hermes/.env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key not in os.environ:
                os.environ[key] = val


def _d1_client() -> D1Client:
    return D1Client(
        account_id=os.environ["CF_ACCOUNT_ID"],
        database_id=os.environ["CF_D1_DATABASE_ID"],
        api_token=os.environ["CF_API_TOKEN"],
    )


def _notion_client() -> NotionClient:
    return NotionClient(
        api_token=os.environ["NOTION_API_TOKEN"],
        database_id=os.environ["NOTION_DATABASE_ID"],
    )


def _cmd_add(args: argparse.Namespace) -> None:
    db = _d1_client()
    db.ensure_table()
    db.upsert_contact(args.name, args.email, args.type, args.context or "")
    print(f"Added: {args.name} ({args.type})")


def main(argv=None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(prog="contacts")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--type", required=True, choices=["professional", "personal"])
    p_add.add_argument("--context", default="")

    # placeholders for later subcommands
    sub.add_parser("summarize").add_argument("--name", required=True)
    sub.add_parser("list").add_argument("--type", choices=["professional", "personal"])
    p_tt = sub.add_parser("talked-to")
    p_tt.add_argument("--name", required=True)
    p_up = sub.add_parser("update")
    p_up.add_argument("--name", required=True)
    p_up.add_argument("--field", required=True)
    p_up.add_argument("--value", required=True)
    sub.add_parser("delete").add_argument("--name", required=True)
    sub.add_parser("sync-calendar").add_argument("--days", type=int, default=1)

    args = parser.parse_args(argv)
    dispatch = {
        "add": _cmd_add,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        print(f"Command '{args.command}' not yet implemented", file=sys.stderr)
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_add_command_creates_contact_and_prints_confirmation tests/test_contacts.py::test_add_command_context_defaults_to_empty_string -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/contacts.py \
        assistant/config/skills/relationship-manager/tests/test_contacts.py && \
git commit -m "feat(e5): add contacts.py with add command"
```

---

### Task 9: contacts.py — summarize command

**Group:** C (sequential after Task 8)

**Behavior being verified:** `summarize` prints the contact card (name, type, email, last contact, context) and each open Notion task for that contact.

**Interface under test:** `contacts.main(["summarize", "--name", ...])`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/contacts.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_contacts.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_contacts.py

def test_summarize_shows_contact_card_and_open_tasks():
    contact_row = {
        "id": 1, "name": "Alice Chen", "email": "alice@example.com",
        "type": "professional", "last_contact": "2026-04-15",
        "context": "Works at Sequoia", "created_at": "2026-04-01",
    }
    tasks = [{"id": "p1", "title": "[Alice Chen] Send IC memo", "status": "In progress", "due": "2026-04-25", "priority": "High"}]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.NotionClient") as MockNotion:
            mock_d1 = MagicMock()
            mock_d1.get_contact.return_value = contact_row
            MockD1.return_value = mock_d1
            mock_notion = MagicMock()
            mock_notion.list_tasks_for_contact.return_value = tasks
            MockNotion.return_value = mock_notion
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["summarize", "--name", "Alice Chen"])
            result = out.getvalue()
            assert "Alice Chen (professional)" in result
            assert "alice@example.com" in result
            assert "2026-04-15" in result
            assert "Works at Sequoia" in result
            assert "[Alice Chen] Send IC memo" in result
            mock_notion.list_tasks_for_contact.assert_called_once_with("Alice Chen")


def test_summarize_shows_none_when_no_tasks():
    contact_row = {
        "id": 1, "name": "Bob Smith", "email": "bob@example.com",
        "type": "personal", "last_contact": None,
        "context": "", "created_at": "2026-04-01",
    }
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.NotionClient") as MockNotion:
            mock_d1 = MagicMock()
            mock_d1.get_contact.return_value = contact_row
            MockD1.return_value = mock_d1
            mock_notion = MagicMock()
            mock_notion.list_tasks_for_contact.return_value = []
            MockNotion.return_value = mock_notion
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["summarize", "--name", "Bob Smith"])
            result = out.getvalue()
            assert "never" in result
            assert "none" in result
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_summarize_shows_contact_card_and_open_tasks tests/test_contacts.py::test_summarize_shows_none_when_no_tasks -v
```

Expected: FAIL — `SystemExit` or `Command 'summarize' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `_cmd_summarize` to `contacts.py` and wire it into `dispatch`:

```python
def _cmd_summarize(args: argparse.Namespace) -> None:
    db = _d1_client()
    notion = _notion_client()
    c = db.get_contact(args.name)
    last = c["last_contact"] or "never"
    ctx = c["context"] or "—"
    print(f"{c['name']} ({c['type']})")
    print(f"Email: {c['email']}")
    print(f"Last contact: {last}")
    print(f"Context: {ctx}")
    print()
    tasks = notion.list_tasks_for_contact(c["name"])
    if tasks:
        print("Open tasks:")
        for t in tasks:
            due_str = f" (due: {t['due']})" if t["due"] else ""
            print(f"  {t['title']}{due_str}")
    else:
        print("Open tasks: none")
```

In `dispatch` dict inside `main()`, add:
```python
"summarize": _cmd_summarize,
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_summarize_shows_contact_card_and_open_tasks tests/test_contacts.py::test_summarize_shows_none_when_no_tasks -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/contacts.py \
        assistant/config/skills/relationship-manager/tests/test_contacts.py && \
git commit -m "feat(e5): add contacts.py summarize command"
```

---

### Task 10: contacts.py — list and talked-to commands

**Group:** C (sequential after Task 9)

**Behavior being verified:** `list` prints one line per contact with name, type, and last contact date; `talked-to` calls `touch_last_contact` with today's ISO date and confirms.

**Interface under test:** `contacts.main(["list"])` and `contacts.main(["talked-to", "--name", ...])`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/contacts.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_contacts.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_contacts.py

def test_list_command_prints_all_contacts():
    rows = [
        {"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
         "last_contact": "2026-04-15", "context": "", "created_at": "2026-04-01"},
        {"id": 2, "name": "Bob Smith", "email": "b@x.com", "type": "personal",
         "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = rows
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["list"])
            result = out.getvalue()
            assert "Alice Chen (professional)" in result
            assert "2026-04-15" in result
            assert "Bob Smith (personal)" in result
            assert "never" in result
            mock_d1.list_contacts.assert_called_once_with(type=None)


def test_list_command_filters_by_type():
    rows = [{"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
             "last_contact": None, "context": "", "created_at": "2026-04-01"}]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = rows
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["list", "--type", "professional"])
            mock_d1.list_contacts.assert_called_once_with(type="professional")


def test_talked_to_updates_last_contact_to_today():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                with patch("contacts.date") as mock_date:
                    mock_date.today.return_value.isoformat.return_value = "2026-04-19"
                    import contacts
                    contacts.main(["talked-to", "--name", "Alice Chen"])
            mock_d1.touch_last_contact.assert_called_once_with("Alice Chen", "2026-04-19")
            assert "Noted: talked to Alice Chen on 2026-04-19" in out.getvalue()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_list_command_prints_all_contacts tests/test_contacts.py::test_list_command_filters_by_type tests/test_contacts.py::test_talked_to_updates_last_contact_to_today -v
```

Expected: FAIL — `SystemExit` or `Command '...' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `contacts.py`:

```python
def _cmd_list(args: argparse.Namespace) -> None:
    db = _d1_client()
    rows = db.list_contacts(type=getattr(args, "type", None))
    if not rows:
        print("No contacts found.")
        return
    for c in rows:
        last = c["last_contact"] or "never"
        print(f"{c['name']} ({c['type']}) — last contact: {last}")


def _cmd_talked_to(args: argparse.Namespace) -> None:
    db = _d1_client()
    today = date.today().isoformat()
    db.touch_last_contact(args.name, today)
    print(f"Noted: talked to {args.name} on {today}")
```

Add to `dispatch` dict inside `main()`:
```python
"list": _cmd_list,
"talked-to": _cmd_talked_to,
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_list_command_prints_all_contacts tests/test_contacts.py::test_list_command_filters_by_type tests/test_contacts.py::test_talked_to_updates_last_contact_to_today -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/contacts.py \
        assistant/config/skills/relationship-manager/tests/test_contacts.py && \
git commit -m "feat(e5): add contacts.py list and talked-to commands"
```

---

### Task 11: contacts.py — update and delete commands

**Group:** C (sequential after Task 10)

**Behavior being verified:** `update` calls `update_contact` with the specified field and value; `delete` calls `delete_contact`.

**Interface under test:** `contacts.main(["update", ...])` and `contacts.main(["delete", ...])`

**Files:**
- Modify: `assistant/config/skills/relationship-manager/scripts/contacts.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_contacts.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_contacts.py

def test_update_command_patches_field():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["update", "--name", "Alice Chen", "--field", "context", "--value", "Partner at Sequoia now"])
            mock_d1.update_contact.assert_called_once_with("Alice Chen", "context", "Partner at Sequoia now")
            assert "Updated: Alice Chen" in out.getvalue()


def test_delete_command_removes_contact():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["delete", "--name", "Alice Chen"])
            mock_d1.delete_contact.assert_called_once_with("Alice Chen")
            assert "Deleted: Alice Chen" in out.getvalue()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_update_command_patches_field tests/test_contacts.py::test_delete_command_removes_contact -v
```

Expected: FAIL — `SystemExit` or `Command '...' not yet implemented`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `contacts.py`:

```python
def _cmd_update(args: argparse.Namespace) -> None:
    db = _d1_client()
    db.update_contact(args.name, args.field, args.value)
    print(f"Updated: {args.name}")


def _cmd_delete(args: argparse.Namespace) -> None:
    db = _d1_client()
    db.delete_contact(args.name)
    print(f"Deleted: {args.name}")
```

Add to `dispatch` dict inside `main()`:
```python
"update": _cmd_update,
"delete": _cmd_delete,
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py -v
```

Expected: PASS (all contacts tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/contacts.py \
        assistant/config/skills/relationship-manager/tests/test_contacts.py && \
git commit -m "feat(e5): add contacts.py update and delete commands — core CLI complete"
```

---

### Task 12: gcal_client.py + sync-calendar command

**Group:** D (parallel with Tasks 13 and 14)

**Behavior being verified:** `sync-calendar` fetches recent calendar events, matches attendee emails against contacts, and calls `touch_last_contact` for each match; prints a summary of updated contacts.

**Interface under test:** `contacts.main(["sync-calendar", "--days", "1"])`

**Files:**
- Create: `assistant/config/skills/relationship-manager/scripts/gcal_client.py`
- Modify: `assistant/config/skills/relationship-manager/scripts/contacts.py`
- Modify: `assistant/config/skills/relationship-manager/tests/test_contacts.py`

---

- [ ] **Step 1: Write the failing test**

```python
# append to test_contacts.py

def test_sync_calendar_updates_matching_contacts():
    events = [
        {
            "summary": "Sequoia Partner Meeting",
            "start": "2026-04-18T15:00:00Z",
            "end": "2026-04-18T16:00:00Z",
            "attendees": ["alice@example.com", "other@external.com"],
        }
    ]
    contact_rows = [
        {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
         "type": "professional", "last_contact": None, "context": "", "created_at": "2026-04-01"},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com",
         "type": "personal", "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = contact_rows
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["sync-calendar", "--days", "1"])
            mock_d1.touch_last_contact.assert_called_once_with("Alice Chen", "2026-04-18")
            result = out.getvalue()
            assert "Alice Chen" in result
            assert "1 contact" in result


def test_sync_calendar_prints_no_matches_when_none():
    events = [
        {
            "summary": "Internal sync",
            "start": "2026-04-18T10:00:00Z",
            "end": "2026-04-18T11:00:00Z",
            "attendees": ["stranger@nowhere.com"],
        }
    ]
    contact_rows = [
        {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
         "type": "professional", "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = contact_rows
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["sync-calendar", "--days", "1"])
            mock_d1.touch_last_contact.assert_not_called()
            assert "0 contacts" in out.getvalue()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_sync_calendar_updates_matching_contacts tests/test_contacts.py::test_sync_calendar_prints_no_matches_when_none -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'gcal_client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Copy `gcal_client.py` from the google-calendar skill:

```bash
cp assistant/config/skills/google-calendar/scripts/gcal_client.py \
   assistant/config/skills/relationship-manager/scripts/gcal_client.py
```

Add `_cmd_sync_calendar` to `contacts.py`:

```python
def _cmd_sync_calendar(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone, timedelta
    db = _d1_client()
    access_token = gcal_client.refresh_access_token(
        client_id=os.environ["GMAIL_CLIENT_ID"],
        client_secret=os.environ["GMAIL_CLIENT_SECRET"],
        refresh_token=os.environ["GMAIL_REFRESH_TOKEN"],
    )
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=args.days)).isoformat().replace("+00:00", "Z")
    time_max = now.isoformat().replace("+00:00", "Z")
    events = gcal_client.list_events(access_token, time_min, time_max, max_results=250)
    all_contacts = db.list_contacts()
    email_to_contact = {c["email"].lower(): c for c in all_contacts}
    updated = []
    for event in events:
        event_date = event["start"][:10]
        for attendee_email in event.get("attendees", []):
            contact = email_to_contact.get(attendee_email.lower())
            if contact and contact["name"] not in updated:
                db.touch_last_contact(contact["name"], event_date)
                updated.append(contact["name"])
    count = len(updated)
    noun = "contact" if count == 1 else "contacts"
    if updated:
        names = ", ".join(updated)
        print(f"Synced calendar: updated last_contact for {count} {noun} ({names})")
    else:
        print(f"Synced calendar: 0 contacts matched")
```

Add to `dispatch` dict inside `main()`:
```python
"sync-calendar": _cmd_sync_calendar,
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/test_contacts.py::test_sync_calendar_updates_matching_contacts tests/test_contacts.py::test_sync_calendar_prints_no_matches_when_none -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/relationship-manager/scripts/gcal_client.py \
        assistant/config/skills/relationship-manager/scripts/contacts.py \
        assistant/config/skills/relationship-manager/tests/test_contacts.py && \
git commit -m "feat(e5): add gcal_client copy and sync-calendar command"
```

---

### Task 13: entrypoint.sh — register calendar-sync cron

**Group:** D (parallel with Tasks 12 and 14)

**Behavior being verified:** The relationship-manager calendar sync is registered as a daily Hermes cron at 08:00 UTC so it runs without manual invocation.

**Interface under test:** Hermes cron jobs file at `~/.hermes/cron/jobs.json`

**Files:**
- Modify: `assistant/entrypoint.sh`

*No automated test — this wires into the Hermes cron system which only runs inside the container. Verify by deploying and checking `~/.hermes/cron/jobs.json` via `flyctl ssh console --user hermes -C "cat ~/.hermes/cron/jobs.json"`.*

---

- [ ] **Step 1: Add the cron block to entrypoint.sh**

In `assistant/entrypoint.sh`, after the `evening-sweep` block (around line 155) and before the closing `"` that ends the Python heredoc, add:

```python
if 'relationship-manager' not in existing_skills:
    jobs.append(make_job(
        ['relationship-manager'],
        "Run the daily calendar sync for the relationship CRM: call python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py sync-calendar --days 1 to fetch yesterday's Google Calendar events and update last_contact for any attendees that match known contacts.",
        '0 8 * * *',
    ))
    added.append('relationship-manager (08:00 UTC / midnight Pacific)')
```

The full insertion point in `entrypoint.sh` is between the `evening-sweep` block ending at `added.append('evening-sweep ...')` and the `with open(jobs_file, 'w') as f:` line.

- [ ] **Step 2: Verify entrypoint.sh is syntactically valid**

```bash
bash -n assistant/entrypoint.sh
```

Expected: no output (no syntax errors)

- [ ] **Step 3: Commit**

```bash
git add assistant/entrypoint.sh && \
git commit -m "feat(e5): register relationship-manager calendar-sync cron at 08:00 UTC"
```

---

### Task 14: SKILL.md

**Group:** D (parallel with Tasks 12 and 13)

**Behavior being verified:** Hermes can discover and invoke relationship-manager commands via the SKILL.md definition.

**Files:**
- Create: `assistant/config/skills/relationship-manager/SKILL.md`

*No automated test — SKILL.md is read by the Hermes runtime. Verify by deploying and asking Mahler to add a contact.*

---

- [ ] **Step 1: Create the SKILL.md**

```markdown
---
name: relationship-manager
description: >
  Track professional and personal relationships. Add contacts, update context,
  view summaries with open Notion tasks, log interactions, and sync Google Calendar
  for auto-detected last contact dates.
triggers:
  - add contact
  - update contact
  - summarize contact
  - list contacts
  - talked to
  - delete contact
  - sync calendar contacts
env:
  - CF_ACCOUNT_ID
  - CF_D1_DATABASE_ID
  - CF_API_TOKEN
  - NOTION_API_TOKEN
  - NOTION_DATABASE_ID
  - GMAIL_CLIENT_ID
  - GMAIL_CLIENT_SECRET
  - GMAIL_REFRESH_TOKEN
---

# Relationship Manager

Track professional and personal contacts in D1. Open commitments are Notion tasks
with `[Name]` prefix (e.g., `[Alice Chen] Send IC memo`). Last contact dates
auto-update daily from Google Calendar attendees at 08:00 UTC.

## Commands

All commands run:
```
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py <subcommand> [args]
```

### Add a contact

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py add \
  --name "Alice Chen" \
  --email "alice@example.com" \
  --type professional \
  --context "Partner at Sequoia, intro'd by Marcus"
```

Output: `Added: Alice Chen (professional)`

`--type` must be `professional` or `personal`. `--context` is optional.

### Summarize a contact

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py summarize \
  --name "Alice Chen"
```

Output: contact card (name, type, email, last contact date, context) followed by
open Notion tasks whose title starts with `[Alice Chen]`.

### List all contacts

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py list
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py list --type professional
```

### Log an interaction (manual touch)

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py talked-to \
  --name "Alice Chen"
```

Sets `last_contact` to today's date. Use for phone calls, in-person meetings, or
any interaction not captured by calendar sync.

Output: `Noted: talked to Alice Chen on 2026-04-19`

### Update a contact field

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py update \
  --name "Alice Chen" \
  --field context \
  --value "Now General Partner at Sequoia"
```

`--field` must be one of: `name`, `email`, `type`, `context`.

Output: `Updated: Alice Chen`

### Delete a contact

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py delete \
  --name "Alice Chen"
```

Output: `Deleted: Alice Chen`

### Sync calendar (runs automatically at 08:00 UTC)

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py sync-calendar --days 1
```

Fetches Google Calendar events from the past N days. For each event attendee whose
email matches a known contact, updates `last_contact` to the event date.

Output: `Synced calendar: updated last_contact for 2 contacts (Alice Chen, Bob Smith)`

## Notion task convention

Open commitments to a contact are tracked as Notion tasks with `[Name]` prefix:

```
[Alice Chen] Send Q2 IC memo
[Alice Chen] Intro to Marcus at Benchmark
```

Create these with the `notion-tasks` skill. The `summarize` command surfaces them automatically.

## Notes

- Name matching is case-insensitive for all commands.
- Email must be unique per contact — `add` on a duplicate email updates the existing record.
- The D1 `contacts` table is in `mahler-db` (same database as `email_triage_log`).
```

- [ ] **Step 2: Commit**

```bash
git add assistant/config/skills/relationship-manager/SKILL.md && \
git commit -m "feat(e5): add relationship-manager SKILL.md"
```

---

## Final verification

After all tasks in Group D are committed, run the full test suite:

```bash
cd assistant/config/skills/relationship-manager && python3 -m pytest tests/ -v
```

Expected: all tests pass (9 d1_client + 3 notion_client + 10 contacts = 22 tests).
