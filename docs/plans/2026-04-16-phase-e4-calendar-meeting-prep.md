# Phase E4 — Google Calendar Skill + Intelligent Meeting Prep System

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Mahler can list and create Google Calendar events via Discord, delivers an intelligent prep brief ~1 hour before meetings, and stays passively aware of upcoming meetings during all chat.
**Spec:** docs/specs/2026-04-16-phase-e4-calendar-meeting-prep-design.md
**Style:** Follow project conventions in assistant/CLAUDE.md — `uv` for Python packages, `RuntimeError` on missing env vars, `_supplement_env_from_hermes()` pattern, `_OPENER` urllib pattern, tests in `tests/` subdir with `sys.path.insert(0, "scripts")`.

---

## Task Groups

- **Group A** (parallel): Task 1, Task 6
- **Group B** (parallel, depends on A): Task 2, Task 7
- **Group C** (parallel, depends on B): Task 3, Task 8
- **Group D** (parallel, depends on C): Task 4, Task 9, Task 11
- **Group E** (parallel, depends on D): Task 5, Task 10, Task 12
- **Group F** (parallel, depends on E): Task 13, Task 14
- **Group G** (depends on F): Task 15

---

### Task 1: `gcal_client.py` — token refresh returns access token

**Group:** A (parallel with Task 6)
**Behavior being verified:** `refresh_access_token` exchanges a refresh token for an access token, raises on HTTP error, raises when access_token is absent from response.
**Interface under test:** `refresh_access_token(client_id, client_secret, refresh_token) -> str`

**Files:**
- Create: `assistant/config/skills/google-calendar/scripts/gcal_client.py`
- Create: `assistant/config/skills/google-calendar/tests/test_gcal_client.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/google-calendar/tests/test_gcal_client.py
import sys
import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, "scripts")
from gcal_client import refresh_access_token


def _make_response(body, status=200):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_http_error(code, body_bytes):
    return urllib.error.HTTPError(
        url="https://oauth2.googleapis.com/token",
        code=code, msg="Error", hdrs=MagicMock(), fp=BytesIO(body_bytes),
    )


class TestRefreshAccessToken(unittest.TestCase):

    @patch("gcal_client._OPENER")
    def test_success_returns_access_token(self, mock_opener):
        mock_opener.open.return_value = _make_response({"access_token": "goog_tok_123"})
        result = refresh_access_token("client_id", "client_secret", "refresh_tok")
        self.assertEqual(result, "goog_tok_123")

    @patch("gcal_client._OPENER")
    def test_raises_on_http_401(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(401, b'{"error":"invalid_client"}')
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("bad", "bad", "bad")
        self.assertIn("401", str(ctx.exception))

    @patch("gcal_client._OPENER")
    def test_raises_when_access_token_missing_from_response(self, mock_opener):
        mock_opener.open.return_value = _make_response({"token_type": "Bearer"})
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("cid", "csec", "rtok")
        self.assertIn("access_token", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal_client.py::TestRefreshAccessToken -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'gcal_client'`

- [ ] **Step 3: Implement**

```python
# assistant/config/skills/google-calendar/scripts/gcal_client.py
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange refresh token for access token. Raises on failure."""
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }).encode("utf-8")
    req = urllib.request.Request(
        _TOKEN_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Token refresh failed: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )
    if status != 200:
        raise RuntimeError(
            f"Token refresh failed: HTTP {status} — {raw.decode('utf-8', errors='replace')}"
        )
    data = json.loads(raw)
    if "access_token" not in data:
        raise RuntimeError(f"Token refresh response missing access_token: {data}")
    return data["access_token"]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal_client.py::TestRefreshAccessToken -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/google-calendar/ && git commit -m "feat(google-calendar): add gcal_client token refresh"
```

---

### Task 2: `gcal_client.py` — `list_events` returns normalized event list

**Group:** B (depends on Task 1)
**Behavior being verified:** `list_events` returns normalized dicts with id/summary/start/end/attendees/description; handles all-day events; raises on API error.
**Interface under test:** `list_events(access_token, time_min, time_max, max_results=50) -> list[dict]`

**Files:**
- Modify: `assistant/config/skills/google-calendar/scripts/gcal_client.py`
- Modify: `assistant/config/skills/google-calendar/tests/test_gcal_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test_gcal_client.py` (keep existing imports and helpers, add new import and class):

```python
from gcal_client import refresh_access_token, list_events


class TestListEvents(unittest.TestCase):

    @patch("gcal_client._OPENER")
    def test_returns_empty_list_when_no_items(self, mock_opener):
        mock_opener.open.return_value = _make_response({"kind": "calendar#events", "items": []})
        result = list_events("tok", "2026-04-16T00:00:00Z", "2026-04-16T23:59:00Z")
        self.assertEqual(result, [])

    @patch("gcal_client._OPENER")
    def test_returns_normalized_event_with_all_fields(self, mock_opener):
        item = {
            "id": "evt123",
            "summary": "Team standup",
            "start": {"dateTime": "2026-04-16T15:00:00Z"},
            "end": {"dateTime": "2026-04-16T15:30:00Z"},
            "attendees": [{"email": "a@x.com"}, {"email": "b@x.com"}],
            "description": "Daily sync",
        }
        mock_opener.open.return_value = _make_response({"items": [item]})
        result = list_events("tok", "2026-04-16T00:00:00Z", "2026-04-16T23:59:00Z")
        self.assertEqual(len(result), 1)
        evt = result[0]
        self.assertEqual(evt["id"], "evt123")
        self.assertEqual(evt["summary"], "Team standup")
        self.assertEqual(evt["start"], "2026-04-16T15:00:00Z")
        self.assertEqual(evt["end"], "2026-04-16T15:30:00Z")
        self.assertEqual(evt["attendees"], ["a@x.com", "b@x.com"])
        self.assertEqual(evt["description"], "Daily sync")

    @patch("gcal_client._OPENER")
    def test_all_day_event_uses_date_field_and_empty_attendees(self, mock_opener):
        item = {
            "id": "evt456",
            "summary": "Birthday",
            "start": {"date": "2026-04-20"},
            "end": {"date": "2026-04-21"},
        }
        mock_opener.open.return_value = _make_response({"items": [item]})
        result = list_events("tok", "2026-04-16T00:00:00Z", "2026-04-21T00:00:00Z")
        self.assertEqual(result[0]["start"], "2026-04-20")
        self.assertEqual(result[0]["attendees"], [])

    @patch("gcal_client._OPENER")
    def test_raises_on_403(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(403, b'{"error":"forbidden"}')
        with self.assertRaises(RuntimeError) as ctx:
            list_events("tok", "2026-04-16T00:00:00Z", "2026-04-16T23:59:00Z")
        self.assertIn("403", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal_client.py::TestListEvents -v
```
Expected: FAIL — `ImportError: cannot import name 'list_events' from 'gcal_client'`

- [ ] **Step 3: Implement**

Add to `gcal_client.py`:

```python
def list_events(
    access_token: str,
    time_min: str,
    time_max: str,
    max_results: int = 50,
) -> list[dict]:
    """Fetch calendar events in [time_min, time_max]. Returns normalized event dicts. Raises on API error."""
    params = urllib.parse.urlencode({
        "timeMin": time_min,
        "timeMax": time_max,
        "maxResults": max_results,
        "singleEvents": "true",
        "orderBy": "startTime",
    })
    url = f"{_CALENDAR_API_BASE}/calendars/primary/events?{params}"
    data = _calendar_get(url, access_token)
    return [_normalize_event(item) for item in data.get("items", [])]


def _normalize_event(item: dict) -> dict:
    start_obj = item.get("start", {})
    end_obj = item.get("end", {})
    return {
        "id": item.get("id", ""),
        "summary": item.get("summary", "(no title)"),
        "start": start_obj.get("dateTime") or start_obj.get("date", ""),
        "end": end_obj.get("dateTime") or end_obj.get("date", ""),
        "attendees": [a["email"] for a in item.get("attendees", []) if "email" in a],
        "description": item.get("description", ""),
    }


def _calendar_get(url: str, access_token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    try:
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Calendar API error: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )
    if status != 200:
        raise RuntimeError(
            f"Calendar API error: HTTP {status} — {raw.decode('utf-8', errors='replace')}"
        )
    return json.loads(raw)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal_client.py::TestListEvents -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/google-calendar/ && git commit -m "feat(google-calendar): add list_events to gcal_client"
```

---

### Task 3: `gcal_client.py` — `create_event` returns normalized event dict

**Group:** C (parallel with Task 8; depends on Task 2)
**Behavior being verified:** `create_event` posts to Calendar API and returns a normalized dict; raises on API error.
**Interface under test:** `create_event(access_token, summary, start, end, attendees=None, description=None) -> dict`

**Files:**
- Modify: `assistant/config/skills/google-calendar/scripts/gcal_client.py`
- Modify: `assistant/config/skills/google-calendar/tests/test_gcal_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test_gcal_client.py`:

```python
from gcal_client import refresh_access_token, list_events, create_event


class TestCreateEvent(unittest.TestCase):

    @patch("gcal_client._OPENER")
    def test_returns_normalized_event_with_id_and_summary(self, mock_opener):
        mock_opener.open.return_value = _make_response({
            "id": "new-evt-abc",
            "summary": "Lunch with Alice",
            "start": {"dateTime": "2026-04-17T12:00:00Z"},
            "end": {"dateTime": "2026-04-17T13:00:00Z"},
        })
        result = create_event(
            access_token="tok",
            summary="Lunch with Alice",
            start="2026-04-17T12:00:00Z",
            end="2026-04-17T13:00:00Z",
        )
        self.assertEqual(result["id"], "new-evt-abc")
        self.assertEqual(result["summary"], "Lunch with Alice")
        self.assertEqual(result["start"], "2026-04-17T12:00:00Z")

    @patch("gcal_client._OPENER")
    def test_raises_on_403(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(403, b'{"error":"forbidden"}')
        with self.assertRaises(RuntimeError) as ctx:
            create_event("tok", "Meeting", "2026-04-17T12:00:00Z", "2026-04-17T13:00:00Z")
        self.assertIn("403", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal_client.py::TestCreateEvent -v
```
Expected: FAIL — `ImportError: cannot import name 'create_event' from 'gcal_client'`

- [ ] **Step 3: Implement**

Add to `gcal_client.py`:

```python
def create_event(
    access_token: str,
    summary: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    description: str | None = None,
) -> dict:
    """Create a calendar event. Returns normalized event dict. Raises on API error."""
    body: dict = {
        "summary": summary,
        "start": {"dateTime": start},
        "end": {"dateTime": end},
    }
    if attendees:
        body["attendees"] = [{"email": e} for e in attendees]
    if description:
        body["description"] = description
    url = f"{_CALENDAR_API_BASE}/calendars/primary/events"
    encoded = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Calendar API error: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )
    if status not in (200, 201):
        raise RuntimeError(
            f"Calendar API error: HTTP {status} — {raw.decode('utf-8', errors='replace')}"
        )
    return _normalize_event(json.loads(raw))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal_client.py::TestCreateEvent -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/google-calendar/ && git commit -m "feat(google-calendar): add create_event to gcal_client"
```

---

### Task 4: `gcal.py` — `list` command outputs formatted event lines

**Group:** D (parallel with Tasks 9, 11; depends on Task 3)
**Behavior being verified:** `gcal.py list` prints `start  summary` lines per event; prints "No upcoming events." when calendar is empty; raises `RuntimeError` when `GMAIL_CLIENT_ID` is not set.
**Interface under test:** `main(["list", "--days", "1"])` CLI

**Files:**
- Create: `assistant/config/skills/google-calendar/scripts/gcal.py`
- Create: `assistant/config/skills/google-calendar/tests/test_gcal.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/google-calendar/tests/test_gcal.py
import sys
import os
import io
import unittest
from unittest.mock import patch

sys.path.insert(0, "scripts")


class TestGcalListCommand(unittest.TestCase):

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_list_prints_start_and_summary(self, mock_list, mock_refresh):
        mock_list.return_value = [{
            "id": "e1", "summary": "Team standup",
            "start": "2026-04-16T15:00:00Z", "end": "2026-04-16T15:30:00Z",
            "attendees": ["alice@x.com"], "description": "",
        }]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["list", "--days", "1"])
        output = captured.getvalue()
        self.assertIn("Team standup", output)
        self.assertIn("2026-04-16T15:00:00Z", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events", return_value=[])
    def test_list_prints_no_events_message_when_empty(self, mock_list, mock_refresh):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["list", "--days", "1"])
        self.assertIn("No upcoming events", captured.getvalue())

    @patch.dict(os.environ, {}, clear=True)
    def test_list_raises_when_client_id_missing(self):
        from gcal import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["list", "--days", "1"])
        self.assertIn("GMAIL_CLIENT_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal.py::TestGcalListCommand -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'gcal'`

- [ ] **Step 3: Implement**

```python
# assistant/config/skills/google-calendar/scripts/gcal.py
"""
Google Calendar CLI for Mahler.

Usage:
    python3 gcal.py list [--days N] [--hours-ahead N]
    python3 gcal.py create --title TITLE --start ISO8601 --end ISO8601 [--attendees email1,email2] [--description TEXT]
"""
import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
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

import gcal_client  # noqa: E402


def _get_credentials() -> tuple[str, str, str]:
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    if not client_id:
        raise RuntimeError("GMAIL_CLIENT_ID environment variable is not set")
    if not client_secret:
        raise RuntimeError("GMAIL_CLIENT_SECRET environment variable is not set")
    if not refresh_token:
        raise RuntimeError("GMAIL_REFRESH_TOKEN environment variable is not set")
    return client_id, client_secret, refresh_token


def cmd_list(args: argparse.Namespace) -> None:
    client_id, client_secret, refresh_token = _get_credentials()
    access_token = gcal_client.refresh_access_token(client_id, client_secret, refresh_token)
    now = datetime.now(timezone.utc)
    if args.hours_ahead is not None:
        time_max = (now + timedelta(hours=args.hours_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        time_max = (now + timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    events = gcal_client.list_events(access_token, time_min, time_max)
    if not events:
        print("No upcoming events.")
        return
    for evt in events:
        print(f"{evt['start']}  {evt['summary']}")
        if evt.get("attendees"):
            print(f"  Attendees: {', '.join(evt['attendees'])}")
        if evt.get("description"):
            print(f"  {evt['description'][:80]}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler Google Calendar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--days", type=int, default=7)
    p_list.add_argument("--hours-ahead", dest="hours_ahead", type=int, default=None)

    # create subcommand added in Task 5
    sub.add_parser("create")

    args = parser.parse_args(argv)
    if args.command == "list":
        cmd_list(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal.py::TestGcalListCommand -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/google-calendar/ && git commit -m "feat(google-calendar): add gcal.py list command"
```

---

### Task 5: `gcal.py` — `create` command outputs confirmation line

**Group:** E (depends on Task 4)
**Behavior being verified:** `gcal.py create` prints `Created: <id> — <title>` and passes attendees as a list to `create_event`.
**Interface under test:** `main(["create", "--title", ..., "--start", ..., "--end", ...])` CLI

**Files:**
- Modify: `assistant/config/skills/google-calendar/scripts/gcal.py`
- Modify: `assistant/config/skills/google-calendar/tests/test_gcal.py`

- [ ] **Step 1: Write the failing test**

Add to `test_gcal.py`:

```python
class TestGcalCreateCommand(unittest.TestCase):

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.create_event")
    def test_create_prints_confirmation_line(self, mock_create, mock_refresh):
        mock_create.return_value = {
            "id": "new-evt-abc", "summary": "Lunch with Alice",
            "start": "2026-04-17T12:00:00Z", "end": "2026-04-17T13:00:00Z",
            "attendees": [], "description": "",
        }
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["create", "--title", "Lunch with Alice",
                  "--start", "2026-04-17T12:00:00Z",
                  "--end", "2026-04-17T13:00:00Z"])
        output = captured.getvalue()
        self.assertIn("Created:", output)
        self.assertIn("Lunch with Alice", output)
        self.assertIn("new-evt-abc", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.create_event")
    def test_create_passes_attendee_list_to_create_event(self, mock_create, mock_refresh):
        mock_create.return_value = {
            "id": "evt-x", "summary": "Sync", "start": "2026-04-17T14:00:00Z",
            "end": "2026-04-17T14:30:00Z", "attendees": [], "description": "",
        }
        with patch("sys.stdout", io.StringIO()):
            from gcal import main
            main(["create", "--title", "Sync",
                  "--start", "2026-04-17T14:00:00Z", "--end", "2026-04-17T14:30:00Z",
                  "--attendees", "alice@x.com,bob@y.com"])
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["attendees"], ["alice@x.com", "bob@y.com"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal.py::TestGcalCreateCommand -v
```
Expected: FAIL — `RuntimeError: Unknown command: create`

- [ ] **Step 3: Implement**

Replace the `main` function in `gcal.py` and add `cmd_create`:

```python
def cmd_create(args: argparse.Namespace) -> None:
    client_id, client_secret, refresh_token = _get_credentials()
    access_token = gcal_client.refresh_access_token(client_id, client_secret, refresh_token)
    attendees = [e.strip() for e in args.attendees.split(",")] if args.attendees else None
    result = gcal_client.create_event(
        access_token=access_token,
        summary=args.title,
        start=args.start,
        end=args.end,
        attendees=attendees,
        description=args.description,
    )
    print(f"Created: {result['id']} — {result['summary']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler Google Calendar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--days", type=int, default=7)
    p_list.add_argument("--hours-ahead", dest="hours_ahead", type=int, default=None)

    p_create = sub.add_parser("create")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--start", required=True)
    p_create.add_argument("--end", required=True)
    p_create.add_argument("--attendees", default=None)
    p_create.add_argument("--description", default=None)

    args = parser.parse_args(argv)
    dispatch = {"list": cmd_list, "create": cmd_create}
    dispatch[args.command](args)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/test_gcal.py -v
```
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/google-calendar/ && git commit -m "feat(google-calendar): add gcal.py create command"
```

---

### Task 6: `meeting-prep/d1_client.py` — base D1 query + meeting_prep_log operations

**Group:** A (parallel with Task 1)
**Behavior being verified:** `D1Client.query` returns row dicts; `is_already_notified` returns correct bool; `insert_meeting_prep` writes correct SQL; `ensure_meeting_prep_table` creates the table; raises on D1 errors.
**Interface under test:** `D1Client(account_id, database_id, api_token)` public methods

**Files:**
- Create: `assistant/config/skills/meeting-prep/scripts/d1_client.py`
- Create: `assistant/config/skills/meeting-prep/tests/test_d1_client.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/meeting-prep/tests/test_d1_client.py
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "scripts")
from d1_client import D1Client, _OPENER


def _success_payload(rows):
    return {"result": [{"results": rows, "success": True}], "success": True, "errors": [], "messages": []}


def _make_response(payload, status=200):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_client():
    return D1Client("test-account", "test-db-456", "test-token")


class TestD1ClientQuery(unittest.TestCase):

    def test_query_returns_row_dicts_on_success(self):
        rows = [{"event_id": "evt1", "summary": "Standup"}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            result = _make_client().query("SELECT * FROM meeting_prep_log")
        self.assertEqual(result, rows)

    def test_query_raises_on_d1_error(self):
        payload = {"result": [], "success": False, "errors": [{"message": "table not found"}], "messages": []}
        with patch.object(_OPENER, "open", return_value=_make_response(payload)):
            with self.assertRaises(RuntimeError) as ctx:
                _make_client().query("SELECT * FROM nonexistent")
        self.assertIn("D1 query failed", str(ctx.exception))

    def test_query_raises_on_http_500(self):
        with patch.object(_OPENER, "open", return_value=_make_response({}, status=500)):
            with self.assertRaises(RuntimeError) as ctx:
                _make_client().query("SELECT 1")
        self.assertIn("D1 API error 500", str(ctx.exception))


class TestIsAlreadyNotified(unittest.TestCase):

    def test_returns_false_when_event_not_found(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            self.assertFalse(_make_client().is_already_notified("evt-new"))

    def test_returns_true_when_event_found(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([{"event_id": "evt-old"}]))):
            self.assertTrue(_make_client().is_already_notified("evt-old"))


class TestInsertMeetingPrep(unittest.TestCase):

    def test_insert_sends_correct_sql_and_params(self):
        captured = []
        def capture(req):
            captured.append(json.loads(req.data.decode("utf-8")))
            return _make_response(_success_payload([]))
        with patch.object(_OPENER, "open", side_effect=capture):
            _make_client().insert_meeting_prep("evt123", "Team standup", "2026-04-16T15:00:00Z")
        self.assertEqual(len(captured), 1)
        self.assertIn("INSERT OR IGNORE", captured[0]["sql"])
        self.assertIn("meeting_prep_log", captured[0]["sql"])
        self.assertEqual(captured[0]["params"][0], "evt123")
        self.assertEqual(captured[0]["params"][1], "Team standup")
        self.assertEqual(captured[0]["params"][2], "2026-04-16T15:00:00Z")


class TestEnsureMeetingPrepTable(unittest.TestCase):

    def test_creates_meeting_prep_log_table(self):
        captured = []
        def capture(req):
            captured.append(json.loads(req.data.decode("utf-8")))
            return _make_response(_success_payload([]))
        with patch.object(_OPENER, "open", side_effect=capture):
            _make_client().ensure_meeting_prep_table()
        self.assertEqual(len(captured), 1)
        self.assertIn("CREATE TABLE IF NOT EXISTS meeting_prep_log", captured[0]["sql"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_d1_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'd1_client'`

- [ ] **Step 3: Implement**

```python
# assistant/config/skills/meeting-prep/scripts/d1_client.py
import json
import re
import ssl
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

    def is_already_notified(self, event_id: str) -> bool:
        """Return True if event_id exists in meeting_prep_log."""
        rows = self.query(
            "SELECT event_id FROM meeting_prep_log WHERE event_id = ? LIMIT 1",
            [event_id],
        )
        return len(rows) > 0

    def insert_meeting_prep(self, event_id: str, summary: str, start_time: str) -> None:
        """Insert a meeting_prep_log row. Idempotent via INSERT OR IGNORE."""
        self.query(
            "INSERT OR IGNORE INTO meeting_prep_log (event_id, summary, start_time, notified_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            [event_id, summary, start_time],
        )

    def ensure_meeting_prep_table(self) -> None:
        """Create meeting_prep_log table if it does not exist. Safe to call on every run."""
        self.query(
            """CREATE TABLE IF NOT EXISTS meeting_prep_log (
    event_id    TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    start_time  TEXT NOT NULL,
    notified_at TEXT NOT NULL
)""",
            [],
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_d1_client.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/meeting-prep/ && git commit -m "feat(meeting-prep): add d1_client with meeting_prep_log operations"
```

---

### Task 7: `dedup.py` — `check` exits 0 for unseen, exits 1 for seen, raises on D1 error

**Group:** B (depends on Task 6)
**Behavior being verified:** `check --event-id` exits 0 when event not in D1; exits 1 when already notified; raises `RuntimeError` (does not silently exit 0) when D1 is unreachable.
**Interface under test:** `main(["check", "--event-id", ID])` CLI exit codes

**Files:**
- Create: `assistant/config/skills/meeting-prep/scripts/dedup.py`
- Create: `assistant/config/skills/meeting-prep/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/meeting-prep/tests/test_dedup.py
import sys
import os
import io
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "scripts")


class TestDedupCheck(unittest.TestCase):

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_check_does_not_exit_1_when_event_not_seen(self, MockClient):
        MockClient.return_value.is_already_notified.return_value = False
        # Should complete without raising SystemExit
        from dedup import main
        try:
            main(["check", "--event-id", "evt-new"])
        except SystemExit as e:
            self.fail(f"Expected clean exit, got SystemExit({e.code})")

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_check_exits_1_when_event_already_notified(self, MockClient):
        MockClient.return_value.is_already_notified.return_value = True
        from dedup import main
        with self.assertRaises(SystemExit) as ctx:
            main(["check", "--event-id", "evt-old"])
        self.assertEqual(ctx.exception.code, 1)

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_check_raises_runtime_error_on_d1_failure(self, MockClient):
        MockClient.return_value.is_already_notified.side_effect = RuntimeError("D1 unreachable")
        from dedup import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["check", "--event-id", "evt-x"])
        self.assertIn("D1 unreachable", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_check_raises_when_cf_account_id_missing(self):
        from dedup import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["check", "--event-id", "evt-x"])
        self.assertIn("CF_ACCOUNT_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_dedup.py::TestDedupCheck -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'dedup'`

- [ ] **Step 3: Implement**

```python
# assistant/config/skills/meeting-prep/scripts/dedup.py
"""
Meeting prep deduplication CLI for Mahler.

Usage:
    python3 dedup.py check --event-id EVENT_ID
        Exit 0: not yet notified — safe to proceed.
        Exit 1: already notified — stop.
        Non-zero + RuntimeError: D1 failure — do not proceed.

    python3 dedup.py log --event-id EVENT_ID --summary SUMMARY --start-time ISO8601
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

from d1_client import D1Client  # noqa: E402


def _get_client() -> D1Client:
    account_id = os.environ.get("CF_ACCOUNT_ID")
    database_id = os.environ.get("CF_D1_DATABASE_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id:
        raise RuntimeError("CF_ACCOUNT_ID environment variable is not set")
    if not database_id:
        raise RuntimeError("CF_D1_DATABASE_ID environment variable is not set")
    if not api_token:
        raise RuntimeError("CF_API_TOKEN environment variable is not set")
    return D1Client(account_id, database_id, api_token)


def cmd_check(args: argparse.Namespace) -> None:
    client = _get_client()
    client.ensure_meeting_prep_table()
    if client.is_already_notified(args.event_id):
        sys.exit(1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler meeting prep dedup")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--event-id", dest="event_id", required=True)

    # log subcommand added in Task 9
    sub.add_parser("log")

    args = parser.parse_args(argv)
    if args.command == "check":
        cmd_check(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_dedup.py::TestDedupCheck -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/meeting-prep/ && git commit -m "feat(meeting-prep): add dedup.py check command"
```

---

### Task 8: `dedup.py` — `log` writes meeting_prep_log row

**Group:** C (parallel with Tasks 3, 11; depends on Task 7)
**Behavior being verified:** `log --event-id --summary --start-time` calls `insert_meeting_prep` with correct args and prints confirmation.
**Interface under test:** `main(["log", "--event-id", ..., "--summary", ..., "--start-time", ...])` CLI

**Files:**
- Modify: `assistant/config/skills/meeting-prep/scripts/dedup.py`
- Modify: `assistant/config/skills/meeting-prep/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

Add to `test_dedup.py`:

```python
class TestDedupLog(unittest.TestCase):

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_log_calls_insert_with_correct_args(self, MockClient):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from dedup import main
            main(["log", "--event-id", "evt123",
                  "--summary", "Team standup",
                  "--start-time", "2026-04-16T15:00:00Z"])
        MockClient.return_value.insert_meeting_prep.assert_called_once_with(
            "evt123", "Team standup", "2026-04-16T15:00:00Z"
        )
        self.assertIn("evt123", captured.getvalue())
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_dedup.py::TestDedupLog -v
```
Expected: FAIL — `RuntimeError: Unknown command: log`

- [ ] **Step 3: Implement**

Replace `main` in `dedup.py` and add `cmd_log`:

```python
def cmd_log(args: argparse.Namespace) -> None:
    client = _get_client()
    client.insert_meeting_prep(args.event_id, args.summary, args.start_time)
    print(f"Logged: {args.event_id}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler meeting prep dedup")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--event-id", dest="event_id", required=True)

    p_log = sub.add_parser("log")
    p_log.add_argument("--event-id", dest="event_id", required=True)
    p_log.add_argument("--summary", required=True)
    p_log.add_argument("--start-time", dest="start_time", required=True)

    args = parser.parse_args(argv)
    dispatch = {"check": cmd_check, "log": cmd_log}
    dispatch[args.command](args)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_dedup.py -v
```
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/meeting-prep/ && git commit -m "feat(meeting-prep): add dedup.py log command"
```

---

### Task 9: `email_context.py` — outputs recent flagged emails for given attendees

**Group:** D (parallel with Tasks 4, 11; depends on Task 6)
**Behavior being verified:** `email-context --attendees` queries D1 email_triage_log for URGENT/NEEDS_ACTION from those addresses in the last 7 days; prints formatted lines; prints "No recent flagged emails" when none found.
**Interface under test:** `main(["email-context", "--attendees", "email1,email2"])` CLI

**Files:**
- Create: `assistant/config/skills/meeting-prep/scripts/email_context.py`
- Create: `assistant/config/skills/meeting-prep/tests/test_email_context.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/meeting-prep/tests/test_email_context.py
import sys
import os
import io
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "scripts")


class TestEmailContext(unittest.TestCase):

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("email_context.D1Client")
    def test_prints_flagged_emails_when_found(self, MockClient):
        MockClient.return_value.query.return_value = [
            {"classification": "URGENT", "from_addr": "alice@x.com",
             "subject": "Q2 Budget", "summary": "Need sign-off by Friday"},
        ]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from email_context import main
            main(["email-context", "--attendees", "alice@x.com"])
        output = captured.getvalue()
        self.assertIn("URGENT", output)
        self.assertIn("alice@x.com", output)
        self.assertIn("Q2 Budget", output)

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("email_context.D1Client")
    def test_prints_no_emails_message_when_none_found(self, MockClient):
        MockClient.return_value.query.return_value = []
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from email_context import main
            main(["email-context", "--attendees", "alice@x.com"])
        self.assertIn("No recent flagged emails", captured.getvalue())

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_when_cf_account_id_missing(self):
        from email_context import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["email-context", "--attendees", "alice@x.com"])
        self.assertIn("CF_ACCOUNT_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_email_context.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'email_context'`

- [ ] **Step 3: Implement**

```python
# assistant/config/skills/meeting-prep/scripts/email_context.py
"""
Fetch recent URGENT/NEEDS_ACTION emails from specific attendees.

Usage:
    python3 email_context.py email-context --attendees "email1@x.com,email2@y.com"
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

from d1_client import D1Client  # noqa: E402


def _get_client() -> D1Client:
    account_id = os.environ.get("CF_ACCOUNT_ID")
    database_id = os.environ.get("CF_D1_DATABASE_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id:
        raise RuntimeError("CF_ACCOUNT_ID environment variable is not set")
    if not database_id:
        raise RuntimeError("CF_D1_DATABASE_ID environment variable is not set")
    if not api_token:
        raise RuntimeError("CF_API_TOKEN environment variable is not set")
    return D1Client(account_id, database_id, api_token)


def cmd_email_context(args: argparse.Namespace) -> None:
    attendees = [e.strip() for e in args.attendees.split(",") if e.strip()]
    if not attendees:
        raise RuntimeError("--attendees must contain at least one email address")
    client = _get_client()
    placeholders = ",".join("?" for _ in attendees)
    rows = client.query(
        f"SELECT classification, from_addr, subject, summary FROM email_triage_log "
        f"WHERE from_addr IN ({placeholders}) "
        f"AND classification IN ('URGENT', 'NEEDS_ACTION') "
        f"AND processed_at > datetime('now', '-7 days') "
        f"ORDER BY processed_at DESC LIMIT 10",
        attendees,
    )
    if not rows:
        print("No recent flagged emails from these contacts.")
        return
    print("Recent emails (last 7 days, URGENT/NEEDS_ACTION):")
    for row in rows:
        classification = row.get("classification", "")
        from_addr = row.get("from_addr", "")
        subject = row.get("subject", "(no subject)")
        summary = row.get("summary", "")
        print(f"  [{classification}] {from_addr}: {subject} — {summary}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler meeting email context")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("email-context")
    p.add_argument("--attendees", required=True)
    args = parser.parse_args(argv)
    if args.command == "email-context":
        cmd_email_context(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_email_context.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/meeting-prep/ && git commit -m "feat(meeting-prep): add email_context.py for attendee email lookup"
```

---

### Task 10: `plugin.py` — returns None when no meeting, returns None on any exception

**Group:** E (parallel with Tasks 5, 12; depends on Task 4)
**Behavior being verified:** `upcoming_meeting_context` returns `None` when D1 has no upcoming meeting; returns `None` silently on any exception.
**Interface under test:** `upcoming_meeting_context(session_id, user_message, is_first_turn, _now=None, **kwargs) -> dict | None`

**Files:**
- Create: `assistant/config/plugins/calendar-aware/plugin.py`
- Create: `assistant/config/plugins/calendar-aware/tests/test_plugin.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/plugins/calendar-aware/tests/test_plugin.py
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPluginReturnNone(unittest.TestCase):

    @patch("plugin._query_upcoming_meeting", return_value=None)
    def test_returns_none_when_no_upcoming_meeting(self, _):
        from plugin import upcoming_meeting_context
        result = upcoming_meeting_context("sess1", "hello", True)
        self.assertIsNone(result)

    @patch("plugin._query_upcoming_meeting", side_effect=Exception("D1 connection refused"))
    def test_returns_none_silently_on_any_exception(self, _):
        from plugin import upcoming_meeting_context
        result = upcoming_meeting_context("sess1", "hello", True)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/plugins/calendar-aware && python3 -m pytest tests/test_plugin.py::TestPluginReturnNone -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'plugin'`

- [ ] **Step 3: Implement**

```python
# assistant/config/plugins/calendar-aware/plugin.py
"""
Calendar-aware pre_llm_call plugin for Mahler.
Injects a one-line upcoming meeting reminder into every chat turn.
Returns None silently on any failure — must never break a chat turn.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


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


def _query_upcoming_meeting() -> dict | None:
    """Query meeting_prep_log for the next meeting starting within 2 hours."""
    _load_hermes_env()
    meeting_prep_scripts = Path.home() / ".hermes" / "skills" / "meeting-prep" / "scripts"
    sys.path.insert(0, str(meeting_prep_scripts))
    from d1_client import D1Client
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    database_id = os.environ.get("CF_D1_DATABASE_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not database_id or not api_token:
        return None
    client = D1Client(account_id, database_id, api_token)
    return client.get_upcoming_meeting()


def upcoming_meeting_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    _now: datetime | None = None,
    **kwargs,
) -> dict | None:
    """Called before each LLM turn. Injects upcoming meeting context or returns None."""
    try:
        meeting = _query_upcoming_meeting()
        if not meeting:
            return None
        # Context string with minutes added in Task 12
        return None
    except Exception:
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", upcoming_meeting_context)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/plugins/calendar-aware && python3 -m pytest tests/test_plugin.py::TestPluginReturnNone -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/plugins/ && git commit -m "feat(calendar-aware): add plugin with silent-fail pre_llm_call hook"
```

---

### Task 11: `d1_client.py` — `get_upcoming_meeting` returns correct meeting or None

**Group:** D (parallel with Tasks 4, 9; depends on Task 6)
**Behavior being verified:** `get_upcoming_meeting` returns the first upcoming meeting row when one exists within 2 hours; returns None when D1 has no rows.
**Interface under test:** `D1Client.get_upcoming_meeting() -> dict | None`

**Files:**
- Modify: `assistant/config/skills/meeting-prep/scripts/d1_client.py`
- Modify: `assistant/config/skills/meeting-prep/tests/test_d1_client.py`

- [ ] **Step 1: Write the failing test**

Add to `test_d1_client.py`:

```python
class TestGetUpcomingMeeting(unittest.TestCase):

    def test_returns_none_when_no_rows(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            result = _make_client().get_upcoming_meeting()
        self.assertIsNone(result)

    def test_returns_first_upcoming_meeting_row(self):
        rows = [{"event_id": "evt-soon", "summary": "Budget review", "start_time": "2026-04-16T15:00:00Z"}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            result = _make_client().get_upcoming_meeting()
        self.assertIsNotNone(result)
        self.assertEqual(result["event_id"], "evt-soon")
        self.assertEqual(result["summary"], "Budget review")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_d1_client.py::TestGetUpcomingMeeting -v
```
Expected: FAIL — `AttributeError: 'D1Client' object has no attribute 'get_upcoming_meeting'`

- [ ] **Step 3: Implement**

Add to the `D1Client` class in `d1_client.py`:

```python
    def get_upcoming_meeting(self) -> Optional[dict]:
        """Return the next logged meeting starting within 2 hours, or None."""
        rows = self.query(
            "SELECT event_id, summary, start_time FROM meeting_prep_log "
            "WHERE start_time > datetime('now') AND start_time < datetime('now', '+2 hours') "
            "ORDER BY start_time ASC LIMIT 1",
            [],
        )
        return rows[0] if rows else None
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/test_d1_client.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/meeting-prep/scripts/d1_client.py assistant/config/skills/meeting-prep/tests/test_d1_client.py && git commit -m "feat(meeting-prep): add get_upcoming_meeting to d1_client"
```

---

### Task 12: `plugin.py` — returns context string when meeting is within 2 hours

**Group:** E (parallel with Tasks 5; depends on Task 10 and Task 11)
**Behavior being verified:** When a meeting exists within 2 hours, `upcoming_meeting_context` returns `{"context": "Upcoming meeting in Xmin: <summary>"}` with correct minutes.
**Interface under test:** `upcoming_meeting_context(session_id, user_message, is_first_turn, _now=datetime, **kwargs)`

**Files:**
- Modify: `assistant/config/plugins/calendar-aware/plugin.py`
- Modify: `assistant/config/plugins/calendar-aware/tests/test_plugin.py`

- [ ] **Step 1: Write the failing test**

Add to `test_plugin.py` (add `from datetime import datetime, timezone` at the top import block):

```python
from datetime import datetime, timezone


class TestPluginReturnContext(unittest.TestCase):

    @patch("plugin._query_upcoming_meeting")
    def test_returns_context_with_minutes_and_summary(self, mock_query):
        mock_query.return_value = {
            "event_id": "evt-soon",
            "summary": "Budget review",
            "start_time": "2026-04-16T15:00:00Z",
        }
        now = datetime(2026, 4, 16, 14, 0, 0, tzinfo=timezone.utc)
        from plugin import upcoming_meeting_context
        result = upcoming_meeting_context("sess1", "hello", True, _now=now)
        self.assertIsNotNone(result)
        self.assertIn("60min", result["context"])
        self.assertIn("Budget review", result["context"])

    @patch("plugin._query_upcoming_meeting")
    def test_returns_context_for_meeting_45_minutes_away(self, mock_query):
        mock_query.return_value = {
            "event_id": "evt-soon",
            "summary": "Standup",
            "start_time": "2026-04-16T14:45:00Z",
        }
        now = datetime(2026, 4, 16, 14, 0, 0, tzinfo=timezone.utc)
        from plugin import upcoming_meeting_context
        result = upcoming_meeting_context("sess1", "hello", True, _now=now)
        self.assertIsNotNone(result)
        self.assertIn("45min", result["context"])
        self.assertIn("Standup", result["context"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd assistant/config/plugins/calendar-aware && python3 -m pytest tests/test_plugin.py::TestPluginReturnContext -v
```
Expected: FAIL — `AssertionError: assertIsNotNone(None)` (Task 10 stub returns None for meeting branch)

- [ ] **Step 3: Implement**

Replace the stub `upcoming_meeting_context` body in `plugin.py` with the full implementation:

```python
def upcoming_meeting_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    _now: datetime | None = None,
    **kwargs,
) -> dict | None:
    """Called before each LLM turn. Injects upcoming meeting context or returns None."""
    try:
        meeting = _query_upcoming_meeting()
        if not meeting:
            return None
        now = _now or datetime.now(timezone.utc)
        start = datetime.fromisoformat(meeting["start_time"].replace("Z", "+00:00"))
        minutes_until = int((start - now).total_seconds() / 60)
        return {"context": f"Upcoming meeting in {minutes_until}min: {meeting['summary']}"}
    except Exception:
        return None
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd assistant/config/plugins/calendar-aware && python3 -m pytest tests/test_plugin.py -v
```
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/plugins/ && git commit -m "feat(calendar-aware): add context string with minutes and meeting summary"
```

---

### Task 13: `google-calendar/SKILL.md` + `meeting-prep/SKILL.md`

**Group:** F (parallel with Task 14; depends on Task 5)
**Behavior being verified:** Hermes has complete, unambiguous instructions for calendar CRUD and meeting prep orchestration.
**Interface under test:** N/A — instruction documents verified by review.

**Files:**
- Create: `assistant/config/skills/google-calendar/SKILL.md`
- Create: `assistant/config/skills/meeting-prep/SKILL.md`

- [ ] **Step 1: Create `google-calendar/SKILL.md`**

```markdown
---
name: google-calendar
description: List upcoming Google Calendar events and create new events. Use for any request involving the user's schedule, meetings, or calendar.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [calendar, google, scheduling, meetings, productivity]
    related_skills: [meeting-prep, notion-tasks]
---

## When to use

- When the user asks "what do I have today/tomorrow/this week?", "what's on my calendar?", or "am I free on Friday?"
- When the user asks to schedule, book, or create a meeting or event
- When the meeting-prep skill needs to fetch upcoming events

## Prerequisites

| Variable | Purpose |
|---|---|
| `GMAIL_CLIENT_ID` | OAuth2 client ID (same as Gmail integration) |
| `GMAIL_CLIENT_SECRET` | OAuth2 client secret |
| `GMAIL_REFRESH_TOKEN` | OAuth2 refresh token (must include `calendar.events` scope) |

All three must be set as Fly.io secrets. The script raises `RuntimeError` if any is missing.

## Date handling

All dates passed to the CLI must be in ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`. Before invoking any command, convert relative dates from the user's message to absolute ISO 8601 using today's date and the user's timezone (Pacific). Example: "Friday at 3pm" → `2026-04-17T22:00:00Z` (UTC).

## Operations

### List upcoming events

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py list --days N
```

`--days N` shows events from now to N days ahead (default: 7). Use `--hours-ahead N` to restrict to the next N hours.

Output format per event:
```
2026-04-16T15:00:00Z  Team standup
  Attendees: alice@x.com, bob@y.com
  Daily sync
```

### Create an event

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py create \
  --title "TITLE" \
  --start "YYYY-MM-DDTHH:MM:SSZ" \
  --end "YYYY-MM-DDTHH:MM:SSZ" \
  [--attendees "email1@x.com,email2@y.com"] \
  [--description "TEXT"]
```

Output on success: `Created: <event_id> — <title>`

## Output

Any failure raises `RuntimeError` and exits non-zero. Surface the error message to the user directly.
```

- [ ] **Step 2: Create `meeting-prep/SKILL.md`**

```markdown
---
name: meeting-prep
description: Check for upcoming meetings in the next hour and deliver an intelligent prep brief to Discord. Run every 15 minutes via cron. Uses google-calendar, notion-tasks, notion-wiki, and email context to synthesize a brief.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [calendar, meetings, briefing, productivity, prep]
    related_skills: [google-calendar, notion-tasks, notion-wiki, email-triage]
---

## When to use

- Invoked automatically by Hermes cron every 15 minutes
- When the user asks "prepare me for my next meeting" or "what do I need to know before my call?"

## Prerequisites

| Variable | Purpose |
|---|---|
| `GMAIL_CLIENT_ID` | OAuth2 client ID |
| `GMAIL_CLIENT_SECRET` | OAuth2 client secret |
| `GMAIL_REFRESH_TOKEN` | OAuth2 refresh token with calendar.events scope |
| `CF_ACCOUNT_ID` | Cloudflare account ID |
| `CF_D1_DATABASE_ID` | D1 database ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 read/write |
| `DISCORD_TRIAGE_WEBHOOK` | Discord webhook for brief delivery |
| `NOTION_API_TOKEN` | Notion token for task lookup |
| `NOTION_DATABASE_ID` | Notion tasks database ID |

## Procedure

### Step 1 — Fetch upcoming events

```bash
python3 ~/.hermes/skills/google-calendar/scripts/gcal.py list --hours-ahead 2
```

Parse the output. Filter for events whose start time (ISO 8601) is between 45 and 75 minutes from the current UTC time. Ignore all-day events (start time contains no `T` separator). If no events are in the 45–75 minute window, stop — nothing to do.

### Step 2 — Check deduplication

For the matching event, extract its event_id from the output line. Then:

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/dedup.py check --event-id EVENT_ID
```

- Exit 0: not yet briefed — continue to Step 3.
- Exit 1: already briefed — stop.
- Non-zero with RuntimeError: D1 failure — surface the error and stop.

### Step 3 — Gather context

Run these in parallel where possible:

**a) Recent emails from attendees** (skip if no attendees listed):

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/email_context.py email-context \
  --attendees "email1@x.com,email2@x.com"
```

**b) Open Notion tasks** (tasks due on or before the meeting date):

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  --status "Not started" --due-before YYYY-MM-DD
```

**c) Wiki lookup** (skip if meeting title is a name only, "1:1", "sync", "standup", or "catch up" with no description):

Extract 1–2 key topic words from the meeting title and description. For each:

```bash
python3 ~/.hermes/skills/notion-wiki/scripts/wiki.py search --query "TOPIC"
```

Read the top hit with `wiki.py read --id PAGE_ID`.

### Step 4 — Synthesize and post brief

Using all gathered context, compose a Discord embed with these sections:
- **Meeting:** title, start time, attendees
- **Recent emails:** formatted output from email_context.py (omit if none)
- **Open tasks:** list of relevant tasks (omit if none)
- **Wiki context:** 1–2 sentences from the wiki hit (omit if none)
- **What to know:** 3–5 bullet point synthesis (always present)

Post to `DISCORD_TRIAGE_WEBHOOK` using the urgent-alert format (embed with title + fields). Do NOT use the email-triage webhook pattern — call the Discord webhook directly with a JSON embed body.

### Step 5 — Log completion

```bash
python3 ~/.hermes/skills/meeting-prep/scripts/dedup.py log \
  --event-id EVENT_ID \
  --summary "MEETING_TITLE" \
  --start-time "ISO8601_START_TIME"
```

This step must succeed. If it raises RuntimeError, surface the error — do not silently skip the log write.

## Failure modes

- `gcal.py` raises RuntimeError → surface to Discord as an error message and stop
- `dedup.py check` exits 1 → stop silently (normal dedup path)
- `dedup.py check` raises RuntimeError → surface to Discord and stop
- Email context or wiki lookup fails → log the error inline in the brief but continue with what was gathered
- notion-tasks list fails → omit tasks section, continue
- `dedup.py log` raises RuntimeError → surface to Discord (brief was already sent; this is critical state)
```

- [ ] **Step 3: Commit**

```bash
git add assistant/config/skills/google-calendar/SKILL.md assistant/config/skills/meeting-prep/SKILL.md && git commit -m "docs(skills): add google-calendar and meeting-prep SKILL.md files"
```

---

### Task 14: Wiring — Dockerfile, entrypoint.sh, .env.example

**Group:** F (parallel with Task 13; depends on Task 5)
**Behavior being verified:** New skills and plugins are present in the Docker image; all required env vars are bridged to `~/.hermes/.env`; meeting-prep cron is registered at startup; NOTION_API_TOKEN/NOTION_DATABASE_ID are bridged (required by notion-tasks when invoked from cron).
**Interface under test:** Container startup — Hermes gateway starts with all skills discoverable.

**Files:**
- Modify: `assistant/Dockerfile`
- Modify: `assistant/entrypoint.sh`
- Modify: `assistant/.env.example`

- [ ] **Step 1: Update `Dockerfile`**

Add after the existing `COPY` lines for skills:

```dockerfile
COPY --chown=hermes:hermes config/skills/google-calendar /home/hermes/.hermes/skills/google-calendar
COPY --chown=hermes:hermes config/skills/meeting-prep /home/hermes/.hermes/skills/meeting-prep
COPY --chown=hermes:hermes config/plugins /home/hermes/.hermes/plugins
```

The full updated skills section of the Dockerfile becomes:

```dockerfile
# Copy custom skills
COPY --chown=hermes:hermes config/skills/email-triage /home/hermes/.hermes/skills/email-triage
COPY --chown=hermes:hermes config/skills/urgent-alert /home/hermes/.hermes/skills/urgent-alert
COPY --chown=hermes:hermes config/skills/morning-brief /home/hermes/.hermes/skills/morning-brief
COPY --chown=hermes:hermes config/skills/notion-tasks /home/hermes/.hermes/skills/notion-tasks
COPY --chown=hermes:hermes config/skills/notion-wiki /home/hermes/.hermes/skills/notion-wiki
COPY --chown=hermes:hermes config/skills/google-calendar /home/hermes/.hermes/skills/google-calendar
COPY --chown=hermes:hermes config/skills/meeting-prep /home/hermes/.hermes/skills/meeting-prep
COPY --chown=hermes:hermes config/plugins /home/hermes/.hermes/plugins
```

- [ ] **Step 2: Update `entrypoint.sh`**

**a)** Add `NOTION_API_TOKEN` and `NOTION_DATABASE_ID` to the `.env` bridge block (notion-tasks is called from the meeting-prep cron and needs these in `~/.hermes/.env`):

```bash
echo "NOTION_API_TOKEN=${NOTION_API_TOKEN:-}" >> "$HERMES_ENV"
echo "NOTION_DATABASE_ID=${NOTION_DATABASE_ID:-}" >> "$HERMES_ENV"
```

Add these two lines immediately after the closing `}` of the first `{ ... } > "$HERMES_ENV"` block.

**b)** Add the meeting-prep cron registration. Inside the `python3 -c "..."` heredoc, after the `morning-brief` block and before the `with open(jobs_file, 'w') ...` line, add:

```python
if 'meeting-prep' not in existing_skills:
    jobs.append(make_job(
        ['meeting-prep', 'google-calendar', 'notion-tasks', 'notion-wiki'],
        'Check if there is a meeting starting in 45 to 75 minutes. If so, check deduplication, gather context from recent emails, open tasks, and the wiki, synthesize a prep brief, post it to Discord, and log the event.',
        '*/15 * * * *',
    ))
    added.append('meeting-prep (every 15 min)')
```

- [ ] **Step 3: Update `.env.example`**

Add the two genuinely new entries (after the existing `NOTION_WIKI_CONCEPTS_DB_ID=` line):

```bash
# notion-tasks skill (Phase E4 — called from meeting-prep cron)
NOTION_API_TOKEN=
NOTION_DATABASE_ID=
```

Note: `GMAIL_REFRESH_TOKEN=` already exists in `.env.example` (line 9) — do NOT add it again.

- [ ] **Step 4: Verify Dockerfile builds (local check)**

```bash
cd assistant && docker build --build-arg HERMES_VERSION=v2026.4.13 -t mahler-test . 2>&1 | tail -20
```

Expected: build succeeds, no COPY errors.

- [ ] **Step 5: Commit**

```bash
git add assistant/Dockerfile assistant/entrypoint.sh assistant/.env.example && git commit -m "chore(assistant): wire google-calendar, meeting-prep, plugins into Docker + cron"
```

---

### Task 15: Run full test suite across all new skills and plugins

**Group:** G (depends on F)
**Behavior being verified:** All new unit tests pass from their respective skill directories.

**Files:** None modified — test run only.

- [ ] **Step 1: Run google-calendar tests**

```bash
cd assistant/config/skills/google-calendar && python3 -m pytest tests/ -v
```
Expected: PASS — all tests in `test_gcal_client.py` and `test_gcal.py`

- [ ] **Step 2: Run meeting-prep tests**

```bash
cd assistant/config/skills/meeting-prep && python3 -m pytest tests/ -v
```
Expected: PASS — all tests in `test_d1_client.py`, `test_dedup.py`, `test_email_context.py`

- [ ] **Step 3: Run plugin tests**

```bash
cd assistant/config/plugins/calendar-aware && python3 -m pytest tests/ -v
```
Expected: PASS — all tests in `test_plugin.py`

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "test: confirm full test suite passes for phase-e4"
```

---

## Challenge Review

### CEO Pass

**Premise:** Right problem, direct path. Calendar awareness is the natural extension of the existing email/tasks/wiki system. Without it Hermes cannot correlate schedule with communications. The plan follows the established `gmail_client → gcal_client → gcal.py → SKILL.md` pattern exactly. No cheaper alternative would deliver the same ambient-prep value.

**Scope:** Appropriately scoped to the spec. The three components (calendar skill, meeting-prep cron, ambient plugin) are tightly coupled enough that shipping them together is correct — partial delivery would leave the system in a half-working state where calendar is queryable but silent before meetings.

**12-Month Alignment:**
```
CURRENT STATE                     THIS PLAN                        12-MONTH IDEAL
Email + tasks + wiki, no     →    + Calendar list/create           Full Execution layer:
schedule awareness                + meeting prep brief             E4 (calendar) +
                                  + ambient plugin                 E5 (CRM) + E6 (approval)
```
Plan moves squarely toward the ideal. No tech debt created.

**Alternatives:** Spec documents the script-first vs Hermes-native cron decision with rationale. No gaps.

---

### Engineering Pass

**Architecture:** Data flow is clean. `gcal.py` → `gcal_client.py` → Google Calendar API mirrors `tasks.py` → `notion_client.py` exactly. `dedup.py` → `d1_client.py` → Cloudflare D1 mirrors the email-triage pattern. Plugin late-imports `d1_client` from the known container path to avoid circular dependency issues. All env var loading follows the established `_supplement_env_from_hermes()` pattern confirmed in `tasks.py` and `triage.py`.

One deployment concern: `entrypoint.sh` bridges `NOTION_API_TOKEN` and `NOTION_DATABASE_ID` via append (`>>`), which is correct — these lines go after the existing wiki-token block. Task 14's instruction is precise.

**Module Depth:** All modules are DEEP. `gcal_client.py` (3-function interface hides OAuth2 + Calendar API protocol), `d1_client.py` (5-method class hides D1 REST + SQL), `dedup.py` (2-command CLI with meaningful exit codes), `email_context.py` (1-command CLI hides cross-table query), `plugin.py` (1-function interface hides D1 lookup, time math, all exceptions). No shallow modules.

---

### Findings

**[BLOCKER] (confidence: 9/10) — `sys.path.insert(0, "..")` in `test_plugin.py` resolves to the wrong directory.**
~~FIXED: Task 10 Step 1 test now uses `sys.path.insert(0, str(Path(__file__).parent.parent))` which correctly resolves to `calendar-aware/` regardless of working directory.~~

---

**[BLOCKER] (confidence: 9/10) — Tasks 11 and 12 were horizontal slices.**
~~FIXED: `get_upcoming_meeting` removed from Task 6 implementation (Task 11 is now a genuine TDD cycle). `upcoming_meeting_context` in Task 10 now stubs the meeting branch with `return None`; Task 12 replaces the stub with the full context-string implementation (genuine failing test: `assertIsNotNone(None)`).~~

---

**[RISK] (confidence: 7/10) — Hermes v0.9.0 `pre_llm_call` plugin API is not verified against actual Hermes source.**

`plugin.py` uses `ctx.register_hook("pre_llm_call", upcoming_meeting_context)` in `register(ctx)`. The CLAUDE.md confirms the "pluggable context engine slot via `hermes plugins` (v0.9)" feature exists, but the exact method name (`register_hook`), hook name (`pre_llm_call`), and callback signature are not confirmed in any file in the repository. If the API differs, the plugin will fail at load or not inject context.

Fallback: Run `flyctl ssh console --user hermes -C "hermes logs errors"` after the first deploy to confirm the plugin loaded. The `except Exception: return None` in `upcoming_meeting_context` ensures chat turns are never broken even if the hook never fires.

---

**[RISK] (confidence: 8/10) — `.env.example` Task 14 duplicate `GMAIL_REFRESH_TOKEN=` entry.**
~~FIXED: Task 14 Step 3 now only adds `NOTION_API_TOKEN=` and `NOTION_DATABASE_ID=` with a note not to duplicate the existing `GMAIL_REFRESH_TOKEN=`.~~

---

**[RISK] (confidence: 6/10) — `dedup.py log` does not call `ensure_meeting_prep_table`.**

`cmd_log` calls `insert_meeting_prep` directly without `ensure_meeting_prep_table`. In the normal SKILL.md flow `check` always runs first and creates the table. Standalone `log` invocation (e.g., manual recovery) would fail with "no such table". Acceptable given the procedural constraint — the build agent should note this.

---

**[RISK] (confidence: 5/10) — Task 5 test asserts on internal call signature.**

`mock_create.call_args[1]` checks the `attendees` kwarg passed to `gcal_client.create_event`. Since `gcal_client` is mocked to avoid HTTP, this is the only observable point for attendee-parsing behavior. Pragmatically acceptable.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|-----------|---------|--------|
| `email_triage_log` has `from_addr` column | SAFE | Confirmed in `email-triage/scripts/d1_client.py` line 93 |
| Hermes v0.9.0 supports `ctx.register_hook("pre_llm_call", ...)` API | RISKY | Not verified in any file; CLAUDE.md only confirms the feature category exists |
| `tasks.py list --due-before` is a valid argument | SAFE | Documented in `tasks.py` docstring line 6 |
| `dedup.py check` exit codes respected by Hermes shell execution | VALIDATE | Hermes exit code handling for script-invoked CLIs not confirmed in docs |
| `CF_API_TOKEN` env var name matches what scripts expect | SAFE | Confirmed in `entrypoint.sh` and `triage.py` both use `CF_API_TOKEN` |
| `plugin.py` hardcoded path `~/.hermes/skills/meeting-prep/scripts` exists in container | SAFE | Dockerfile COPY for meeting-prep creates this path |
| `NOTION_API_TOKEN` and `NOTION_DATABASE_ID` are set as Fly.io secrets | VALIDATE | Present in local `.env`; must be set with `flyctl secrets set` before deploy |

---

### Summary

```
[BLOCKER] count: 0  (both resolved in this revision)
[RISK]    count: 3  (env.example duplicate resolved; Hermes plugin API and dedup.log fragility remain)
[QUESTION] count: 0
```

VERDICT: PROCEED_WITH_CAUTION — [Risk 1: verify Hermes pre_llm_call plugin API after first deploy via `hermes logs errors`; Risk 2: dedup.py log standalone invocation fragile if table not pre-created by check]
