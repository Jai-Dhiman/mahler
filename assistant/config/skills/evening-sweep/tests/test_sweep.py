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
            [],  # overdue
            [],  # open tasks (added in this task)
        ]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            sweep.main(_today=date(2026, 4, 19))

        output = mock_out.getvalue()
        self.assertIn("=== COMPLETED TODAY ===", output)
        self.assertIn("Write tests", output)
        mock_client.list_tasks.assert_any_call(status="Done", last_edited_after="2026-04-19")


class TestPastDueSection(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("sweep.NotionClient")
    def test_past_due_task_shows_days_overdue_and_done_tasks_excluded(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.side_effect = [
            [],  # completed
            [
                _make_task(title="File taxes", status="Not started", due="2026-04-16"),
                _make_task(title="Already done", status="Done", due="2026-04-15"),
            ],  # overdue
            [],  # open tasks (added in this task)
        ]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            sweep.main(_today=date(2026, 4, 19))

        output = mock_out.getvalue()
        self.assertIn("=== PAST DUE (not done) ===", output)
        self.assertIn("File taxes", output)
        self.assertIn("3 days overdue", output)
        self.assertNotIn("Already done", output)
        mock_client.list_tasks.assert_any_call(due_before="2026-04-18")


class TestOpenTasksSection(unittest.TestCase):

    @patch.dict(os.environ, {"NOTION_API_TOKEN": "tok", "NOTION_DATABASE_ID": "db-id"})
    @patch("sweep.NotionClient")
    def test_open_tasks_appear_with_title_due_and_priority(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.list_tasks.side_effect = [
            [],  # completed
            [],  # overdue
            [
                _make_task(title="Refactor auth", due="2026-04-21", priority="High"),
                _make_task(title="Write docs", due=None, priority="Low"),
            ],  # open tasks
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


if __name__ == "__main__":
    unittest.main()
