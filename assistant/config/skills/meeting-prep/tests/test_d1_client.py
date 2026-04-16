import sys
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "scripts")
from d1_client import D1Client, _OPENER


def _success_payload(rows):
    return {"result": [{"results": rows, "success": True}], "success": True, "errors": [], "messages": []}


def _make_response(payload, status=200):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_client():
    return D1Client("test-account", "test-db-456", "test-token")


class TestD1ClientQuery(unittest.TestCase):

    def test_query_returns_row_dicts_on_success(self):
        rows = [{"event_id": "evt1", "summary": "Standup"}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            result = _make_client().query("SELECT * FROM meeting_prep_log")
        self.assertEqual(result, rows)

    def test_query_raises_on_d1_error(self):
        payload = {"result": [], "success": False, "errors": [{"message": "table not found"}], "messages": []}
        with patch.object(_OPENER, "open", return_value=_make_response(payload)):
            with self.assertRaises(RuntimeError) as ctx:
                _make_client().query("SELECT * FROM nonexistent")
        self.assertIn("D1 query failed", str(ctx.exception))

    def test_query_raises_on_http_500(self):
        with patch.object(_OPENER, "open", return_value=_make_response({}, status=500)):
            with self.assertRaises(RuntimeError) as ctx:
                _make_client().query("SELECT 1")
        self.assertIn("D1 API error 500", str(ctx.exception))


class TestIsAlreadyNotified(unittest.TestCase):

    def test_returns_false_when_event_not_found(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            self.assertFalse(_make_client().is_already_notified("evt-new"))

    def test_returns_true_when_event_found(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([{"event_id": "evt-old"}]))):
            self.assertTrue(_make_client().is_already_notified("evt-old"))


class TestInsertMeetingPrep(unittest.TestCase):

    def test_insert_sends_correct_sql_and_params(self):
        captured = []
        def capture(req):
            captured.append(json.loads(req.data.decode("utf-8")))
            return _make_response(_success_payload([]))
        with patch.object(_OPENER, "open", side_effect=capture):
            _make_client().insert_meeting_prep("evt123", "Team standup", "2026-04-16T15:00:00Z")
        self.assertEqual(len(captured), 1)
        self.assertIn("INSERT OR IGNORE", captured[0]["sql"])
        self.assertIn("meeting_prep_log", captured[0]["sql"])
        self.assertEqual(captured[0]["params"][0], "evt123")
        self.assertEqual(captured[0]["params"][1], "Team standup")
        self.assertEqual(captured[0]["params"][2], "2026-04-16T15:00:00Z")


class TestEnsureMeetingPrepTable(unittest.TestCase):

    def test_creates_meeting_prep_log_table(self):
        captured = []
        def capture(req):
            captured.append(json.loads(req.data.decode("utf-8")))
            return _make_response(_success_payload([]))
        with patch.object(_OPENER, "open", side_effect=capture):
            _make_client().ensure_meeting_prep_table()
        self.assertEqual(len(captured), 1)
        self.assertIn("CREATE TABLE IF NOT EXISTS meeting_prep_log", captured[0]["sql"])


if __name__ == "__main__":
    unittest.main()
