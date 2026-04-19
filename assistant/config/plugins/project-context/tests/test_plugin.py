import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProjectContextEmptyLog(unittest.TestCase):

    @patch("plugin._query_project_log", return_value=[])
    def test_returns_none_when_project_log_is_empty(self, _):
        from plugin import project_context
        result = project_context("sess1", "what's up", True)
        self.assertIsNone(result)

    @patch("plugin._query_project_log", return_value=[])
    def test_returns_none_on_non_first_turn_with_empty_log(self, _):
        from plugin import project_context
        result = project_context("sess1", "any updates?", False)
        self.assertIsNone(result)


class TestProjectContextFormatsRows(unittest.TestCase):

    @patch("plugin._query_project_log")
    def test_returns_context_dict_with_win_and_blocker_entries(self, mock_query):
        mock_query.return_value = [
            {
                "project": "mahler",
                "entry_type": "win",
                "summary": "Shipped kaizen-reflection skill",
                "created_at": "2026-04-19 10:00:00",
            },
            {
                "project": "traderjoe",
                "entry_type": "blocker",
                "summary": "Ghost trades still appearing after reconciliation fix",
                "created_at": "2026-04-18 09:00:00",
            },
        ]
        from plugin import project_context
        result = project_context("sess1", "what's the status?", True)
        self.assertIsNotNone(result)
        self.assertIn("context", result)
        self.assertIn("mahler", result["context"])
        self.assertIn("WIN", result["context"])
        self.assertIn("traderjoe", result["context"])
        self.assertIn("BLOCKER", result["context"])
        self.assertIn("kaizen-reflection", result["context"])

    @patch("plugin._query_project_log")
    def test_returns_context_on_non_first_turn(self, mock_query):
        mock_query.return_value = [
            {
                "project": "mahler",
                "entry_type": "win",
                "summary": "Shipped morning-brief improvements",
                "created_at": "2026-04-19 08:00:00",
            },
        ]
        from plugin import project_context
        result = project_context("sess1", "any updates?", False)
        self.assertIsNotNone(result)
        self.assertIn("mahler", result["context"])
