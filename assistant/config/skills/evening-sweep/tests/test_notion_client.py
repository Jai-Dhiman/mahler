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
