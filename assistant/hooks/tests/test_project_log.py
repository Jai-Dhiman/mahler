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
