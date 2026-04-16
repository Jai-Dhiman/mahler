import sys
import unittest
from datetime import datetime, timezone
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


if __name__ == "__main__":
    unittest.main()
