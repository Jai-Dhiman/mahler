import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPriorityMapContextReturnsNone(unittest.TestCase):

    @patch("plugin._query_priority_map", return_value=None)
    def test_returns_none_when_priority_map_unavailable(self, _):
        from plugin import priority_map_context
        result = priority_map_context("sess1", "check my email", True)
        self.assertIsNone(result)

    @patch("plugin._query_priority_map", side_effect=Exception("D1 connection refused"))
    def test_returns_none_silently_on_any_exception(self, _):
        from plugin import priority_map_context
        result = priority_map_context("sess1", "hello", True)
        self.assertIsNone(result)


class TestPriorityMapContextReturnsContent(unittest.TestCase):

    @patch("plugin._query_priority_map")
    def test_returns_context_dict_with_priority_map_content(self, mock_query):
        mock_query.return_value = "## URGENT\nDrop everything.\n## NEEDS_ACTION\n..."
        from plugin import priority_map_context
        result = priority_map_context("sess1", "triage my email", True)
        self.assertIsNotNone(result)
        self.assertIn("URGENT", result["context"])
        self.assertIn("NEEDS_ACTION", result["context"])

    @patch("plugin._query_priority_map")
    def test_returns_context_on_non_first_turn(self, mock_query):
        mock_query.return_value = "## URGENT\nTest."
        from plugin import priority_map_context
        result = priority_map_context("sess1", "follow up", False)
        self.assertIsNotNone(result)
        self.assertIn("URGENT", result["context"])


if __name__ == "__main__":
    unittest.main()
