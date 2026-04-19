import sys
sys.path.insert(0, 'scripts')

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from d1_client import D1Client, _OPENER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(payload: dict, status: int = 200):
    """Return a mock response whose .read() returns JSON bytes and .status is set."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestD1ClientQuery(unittest.TestCase):

    def test_query_returns_row_dicts_on_success(self):
        rows = [{"message_id": "abc", "classification": "URGENT"}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            result = client.query("SELECT * FROM email_triage_log")
        self.assertEqual(result, rows)

    def test_query_raises_on_http_500(self):
        error_payload = {"success": False, "errors": [{"message": "Internal Server Error"}], "result": [], "messages": []}
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload, status=500)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.query("SELECT 1")
        self.assertIn("D1 API error 500", str(ctx.exception))

    def test_query_raises_when_success_false(self):
        payload = {
            "result": [],
            "success": False,
            "errors": [{"code": 7500, "message": "query failed"}],
            "messages": [],
        }
        with patch.object(_OPENER, "open", return_value=_make_response(payload)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.query("BAD SQL")
        self.assertIn("D1 query failed", str(ctx.exception))

    def test_query_returns_empty_list_when_result_is_empty(self):
        payload = {"result": [], "success": True, "errors": [], "messages": []}
        with patch.object(_OPENER, "open", return_value=_make_response(payload)):
            client = _make_client()
            result = client.query("SELECT * FROM email_triage_log WHERE 1=0")
        self.assertEqual(result, [])


class TestD1ClientIsAlreadyProcessed(unittest.TestCase):

    def test_returns_true_when_row_found(self):
        rows = [{"message_id": "msg-001"}]
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload(rows))):
            client = _make_client()
            self.assertTrue(client.is_already_processed("msg-001"))

    def test_returns_false_when_no_rows_returned(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))):
            client = _make_client()
            self.assertFalse(client.is_already_processed("msg-999"))


class TestD1ClientInsertTriageResult(unittest.TestCase):

    def test_insert_calls_query_with_insert_or_ignore(self):
        with patch.object(_OPENER, "open", return_value=_make_response(_success_payload([]))) as mock_open:
            client = _make_client()
            client.insert_triage_result({
                "message_id": "msg-001",
                "source": "gmail",
                "from_addr": "sender@example.com",
                "subject": "Hello",
                "received_at": "2026-04-07T10:00:00Z",
                "classification": "FYI",
                "summary": "A test email",
                "alerted": 0,
                "classification_error": 0,
                "processed_at": "2026-04-07T10:01:00Z",
            })

        mock_open.assert_called_once()
        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertIn("INSERT OR IGNORE", body["sql"])
        self.assertIn("email_triage_log", body["sql"])
        self.assertEqual(body["params"][0], "msg-001")


class TestD1ClientEnsureTables(unittest.TestCase):

    def test_ensure_tables_executes_create_table_statements(self):
        calls = []

        def capture_open(req):
            body = json.loads(req.data.decode("utf-8"))
            calls.append(body["sql"])
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.ensure_tables()

        self.assertEqual(len(calls), 5)
        self.assertIn("CREATE TABLE IF NOT EXISTS email_triage_log", calls[0])
        self.assertIn("CREATE TABLE IF NOT EXISTS triage_state", calls[1])
        self.assertIn("CREATE TABLE IF NOT EXISTS mahler_kv", calls[2])
        self.assertIn("CREATE TABLE IF NOT EXISTS priority_map", calls[3])


class TestD1ClientProjectLogTable(unittest.TestCase):

    def test_ensure_tables_creates_project_log_table(self):
        calls = []

        def capture_open(req):
            body = json.loads(req.data.decode("utf-8"))
            calls.append(body["sql"])
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.ensure_tables()

        self.assertEqual(len(calls), 5)
        self.assertIn("CREATE TABLE IF NOT EXISTS project_log", calls[4])
        self.assertIn("entry_type", calls[4])
        self.assertIn("summary", calls[4])


class TestD1ClientAuthHeader(unittest.TestCase):

    def test_authorization_header_is_set_correctly(self):
        captured_requests = []

        def capture_open(req):
            captured_requests.append(req)
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.query("SELECT 1")

        self.assertEqual(len(captured_requests), 1)
        auth = captured_requests[0].get_header("Authorization")
        self.assertEqual(auth, "Bearer test-token-abc")

    def test_authorization_header_present_on_insert(self):
        captured_requests = []

        def capture_open(req):
            captured_requests.append(req)
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.insert_triage_result({
                "message_id": "x",
                "source": "gmail",
                "classification": "NOISE",
                "processed_at": "2026-04-07T00:00:00Z",
            })

        auth = captured_requests[0].get_header("Authorization")
        self.assertEqual(auth, "Bearer test-token-abc")

    def test_authorization_header_present_on_ensure_tables(self):
        captured_requests = []

        def capture_open(req):
            captured_requests.append(req)
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=capture_open):
            client = _make_client()
            client.ensure_tables()

        self.assertEqual(len(captured_requests), 5)
        for req in captured_requests:
            self.assertEqual(req.get_header("Authorization"), "Bearer test-token-abc")


class TestD1ClientInit(unittest.TestCase):

    def test_invalid_account_id_raises(self):
        with self.assertRaises(ValueError):
            D1Client(account_id="bad/id", database_id="db-123", api_token="tok")

    def test_invalid_database_id_raises(self):
        with self.assertRaises(ValueError):
            D1Client(account_id="acct-123", database_id="bad id!", api_token="tok")


class TestGetPriorityMap(unittest.TestCase):

    def test_returns_content_when_row_exists(self):
        rows = [{"content": "## URGENT\nDrop everything.", "version": 1, "updated_at": "2026-04-18T00:00:00Z"}]
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


class TestSetPriorityMap(unittest.TestCase):

    def test_calls_d1_with_insert_sql_and_new_content(self):
        calls = []

        def fake_open(req):
            calls.append(req)
            payload = {"result": [{"results": [], "success": True}], "success": True, "errors": [], "messages": []}
            return _make_response(payload)

        with patch.object(_OPENER, "open", side_effect=fake_open):
            client = _make_client()
            client.set_priority_map("## URGENT\nUpdated content.")

        self.assertEqual(len(calls), 1)
        body = json.loads(calls[0].data.decode("utf-8"))
        self.assertIn("priority_map", body["sql"])
        self.assertIn("## URGENT\nUpdated content.", body["params"])


if __name__ == "__main__":
    unittest.main()
