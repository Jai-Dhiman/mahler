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


if __name__ == "__main__":
    unittest.main()
