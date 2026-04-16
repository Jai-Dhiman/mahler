import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
