import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLogWin(unittest.TestCase):

    def test_log_win_inserts_win_row_to_d1(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_win
            log_win("mahler", "Shipped kaizen-reflection skill", "abc1234")
        mock_client.insert_project_log.assert_called_once_with(
            "mahler", "win", "Shipped kaizen-reflection skill", "abc1234"
        )

    def test_log_win_propagates_d1_exception(self):
        mock_client = MagicMock()
        mock_client.insert_project_log.side_effect = RuntimeError("D1 timeout")
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_win
            with self.assertRaises(RuntimeError):
                log_win("mahler", "Shipped something", "abc")


class TestLogBlockerNoKeyword(unittest.TestCase):

    def test_does_not_write_when_no_keywords_in_user_messages(self):
        transcript = {
            "cwd": "/Users/jdhiman/Documents/mahler",
            "messages": [
                {"role": "user", "content": "can you add a feature to the morning brief?"},
                {"role": "assistant", "content": "Sure, I will add it."},
            ],
        }
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/Users/jdhiman/Documents/mahler")
        mock_client.insert_project_log.assert_not_called()

    def test_does_not_write_when_messages_list_is_empty(self):
        transcript = {"cwd": "/tmp/project", "messages": []}
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")
        mock_client.insert_project_log.assert_not_called()

    def test_does_not_write_when_keyword_appears_only_in_assistant_message(self):
        transcript = {
            "cwd": "/tmp/project",
            "messages": [
                {"role": "user", "content": "what should I work on next?"},
                {"role": "assistant", "content": "the issue is that you have many options"},
            ],
        }
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")
        mock_client.insert_project_log.assert_not_called()
