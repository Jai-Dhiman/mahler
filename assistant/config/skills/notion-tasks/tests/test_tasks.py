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
