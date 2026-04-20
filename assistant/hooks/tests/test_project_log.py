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


class TestLogBlockerKeywordMatch(unittest.TestCase):

    def test_writes_blocker_row_when_keyword_matched(self):
        transcript = {
            "cwd": "/Users/jdhiman/Documents/mahler",
            "messages": [
                {
                    "role": "user",
                    "content": "the main issue is that the model isn't giving accurate scores",
                },
                {"role": "assistant", "content": "Let me look into that."},
            ],
        }
        mock_client = MagicMock()

        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._call_openrouter",
                   return_value="The scoring model produces inaccurate outputs, blocking feature completion."), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc1234"):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/Users/jdhiman/Documents/mahler")

        mock_client.insert_project_log.assert_called_once_with(
            "mahler",
            "blocker",
            "The scoring model produces inaccurate outputs, blocking feature completion.",
            "abc1234",
        )

    def test_does_not_write_when_openrouter_returns_empty_string(self):
        transcript = {
            "cwd": "/tmp/project",
            "messages": [
                {"role": "user", "content": "i am stuck on this issue"},
            ],
        }
        mock_client = MagicMock()

        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._call_openrouter", return_value=""), \
             patch("project_log._derive_project_name", return_value="project"), \
             patch("project_log._derive_git_ref", return_value=""):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")

        mock_client.insert_project_log.assert_not_called()

    def test_uses_transcript_key_as_fallback_for_messages(self):
        transcript = {
            "cwd": "/tmp/project",
            "transcript": [
                {"role": "user", "content": "blocked on the D1 migration"},
            ],
        }
        mock_client = MagicMock()

        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._call_openrouter", return_value="D1 migration is blocking progress."), \
             patch("project_log._derive_project_name", return_value="project"), \
             patch("project_log._derive_git_ref", return_value=""):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")

        mock_client.insert_project_log.assert_called_once()
        args = mock_client.insert_project_log.call_args[0]
        self.assertEqual(args[1], "blocker")


class TestLogSessionHeartbeat(unittest.TestCase):

    def test_writes_session_row_with_project_ref_branch(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc1234"), \
             patch("project_log._derive_branch", return_value="main"):
            from project_log import log_session_heartbeat
            log_session_heartbeat("/Users/jdhiman/Documents/mahler")

        mock_client.ensure_tables.assert_called_once()
        mock_client.insert_session_heartbeat.assert_called_once_with("mahler", "abc1234", "main")

    def test_works_outside_git_repo(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._derive_project_name", return_value="tmp"), \
             patch("project_log._derive_git_ref", return_value=""), \
             patch("project_log._derive_branch", return_value=""):
            from project_log import log_session_heartbeat
            log_session_heartbeat("/tmp/project")

        mock_client.insert_session_heartbeat.assert_called_once_with("tmp", "", "")

    def test_propagates_d1_exception(self):
        mock_client = MagicMock()
        mock_client.insert_session_heartbeat.side_effect = RuntimeError("D1 error")
        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc"), \
             patch("project_log._derive_branch", return_value="main"):
            from project_log import log_session_heartbeat
            with self.assertRaises(RuntimeError):
                log_session_heartbeat("/Users/jdhiman/Documents/mahler")
