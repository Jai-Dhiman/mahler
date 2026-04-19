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
