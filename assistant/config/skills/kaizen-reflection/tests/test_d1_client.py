import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from d1_client import D1Client, _OPENER


def _make_response(payload: dict, status: int = 200):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _success_payload(rows: list) -> dict:
    return {
        "result": [{"results": rows, "success": True}],
        "success": True,
        "errors": [],
        "messages": [],
    }


def _make_client() -> D1Client:
    return D1Client(
        account_id="test-account-123",
        database_id="test-db-456",
        api_token="test-token-abc",
    )


class TestGetTriagePatterns(unittest.TestCase):

    def test_returns_senders_meeting_min_count_threshold(self):
        rows = [
            {"from_addr": "news@acme.com", "classification": "NEEDS_ACTION", "occurrence_count": 5}
        ]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            patterns = client.get_triage_patterns(since_days=7, min_count=3)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["from_addr"], "news@acme.com")
        self.assertEqual(patterns[0]["classification"], "NEEDS_ACTION")
        self.assertEqual(patterns[0]["occurrence_count"], 5)

    def test_returns_empty_list_when_no_patterns_meet_threshold(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            patterns = client.get_triage_patterns(since_days=7, min_count=3)
        self.assertEqual(patterns, [])


class TestGetPriorityMapKaizen(unittest.TestCase):

    def test_returns_content_when_row_exists(self):
        rows = [{"content": "## URGENT\nDrop everything."}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            result = client.get_priority_map()
        self.assertEqual(result, "## URGENT\nDrop everything.")

    def test_raises_when_no_row_exists(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.get_priority_map()
        self.assertIn("priority_map table is empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
