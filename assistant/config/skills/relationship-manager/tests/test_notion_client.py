import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from notion_client import NotionClient


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
        client = NotionClient(api_token="ntok", database_id="ndb1")
        tasks = client.list_tasks_for_contact("Alice Chen")
        assert len(tasks) == 2
        assert tasks[0]["title"] == "[Alice Chen] Task 1"
        assert tasks[1]["title"] == "[Alice Chen] Task 2"


def test_list_tasks_handles_missing_optional_properties():
    page_no_optional = {
        "id": "page-xyz",
        "properties": {
            "Task name": {"title": [{"plain_text": "[Alice Chen] Minimal task"}]},
            "Status": {"status": {"name": "Todo"}},
        },
    }
    with patch("notion_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_notion_response(results=[page_no_optional])
        client = NotionClient(api_token="ntok", database_id="ndb1")
        tasks = client.list_tasks_for_contact("Alice Chen")
        assert len(tasks) == 1
        assert tasks[0]["due"] is None
        assert tasks[0]["priority"] is None
