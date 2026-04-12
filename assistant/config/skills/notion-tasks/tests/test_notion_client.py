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


if __name__ == "__main__":
    unittest.main()
