import json
import sys
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, "scripts")

from outlook_client import refresh_access_token, fetch_unread_emails


def _make_response(body: dict | list, status: int = 200) -> MagicMock:
    raw = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_http_error(code: int, body: bytes, url: str = "https://example.com") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url=url, code=code, msg="Error", hdrs=MagicMock(), fp=BytesIO(body))


def _make_graph_message(
    msg_id: str = "graph-id-1",
    internet_id: str = "abc@mail.example.com",
    subject: str = "Hello world",
    from_name: str = "Alice",
    from_email: str = "alice@example.com",
    received: str = "2026-04-08T10:00:00Z",
    body_content: str = "This is the email body.",
    body_type: str = "text",
    is_junk: bool = False,
) -> dict:
    return {
        "id": msg_id,
        "internetMessageId": f"<{internet_id}>",
        "subject": subject,
        "from": {"emailAddress": {"name": from_name, "address": from_email}},
        "receivedDateTime": received,
        "body": {"contentType": body_type, "content": body_content},
        "internetMessageHeaders": [
            {"name": "From", "value": f"{from_name} <{from_email}>"},
        ],
    }


class TestRefreshAccessToken(unittest.TestCase):
    @patch("outlook_client._OPENER")
    def test_success_returns_token(self, mock_opener):
        mock_opener.open.return_value = _make_response({"access_token": "tok123"})
        access_token, new_refresh = refresh_access_token("client_id", "client_secret", "refresh_tok")
        self.assertEqual(access_token, "tok123")
        self.assertEqual(new_refresh, "")

    @patch("outlook_client._OPENER")
    def test_raises_on_401(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(401, b'{"error": "invalid_client"}')
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("bad_id", "bad_secret", "bad_token")
        self.assertIn("401", str(ctx.exception))

    @patch("outlook_client._OPENER")
    def test_raises_if_access_token_missing(self, mock_opener):
        mock_opener.open.return_value = _make_response({"token_type": "Bearer"})
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("client_id", "client_secret", "refresh_tok")
        self.assertIn("access_token", str(ctx.exception))


class TestFetchUnreadEmails(unittest.TestCase):
    def _run_fetch(self, inbox_messages: list, junk_messages: list) -> list:
        token_resp = _make_response({"access_token": "test_token"})
        inbox_resp = _make_response({"value": inbox_messages})
        junk_resp = _make_response({"value": junk_messages})
        mark_read_resp = _make_response({}, status=200)

        with patch("outlook_client._OPENER") as mock_opener:
            mock_opener.open.side_effect = [
                token_resp,
                inbox_resp,
                *[mark_read_resp for _ in inbox_messages],
                junk_resp,
                *[mark_read_resp for _ in junk_messages],
            ]
            results, _ = fetch_unread_emails("client_id", "client_secret", "refresh_tok")
            return results

    def test_inbox_fetch_returns_correct_shape(self):
        msg = _make_graph_message(
            subject="Hello INBOX",
            from_name="Alice",
            from_email="alice@example.com",
            internet_id="inbox1@mail.example.com",
        )
        results = self._run_fetch([msg], [])

        self.assertEqual(len(results), 1)
        m = results[0]
        self.assertEqual(m["source"], "outlook")
        self.assertEqual(m["message_id"], "inbox1@mail.example.com")
        self.assertEqual(m["subject"], "Hello INBOX")
        self.assertIn("alice@example.com", m["from_addr"])
        self.assertFalse(m["is_junk_rescue"])
        self.assertIn("body_preview", m)
        self.assertIn("received_at", m)
        self.assertIsInstance(m["headers"], dict)

    def test_junk_fetch_returns_junk_rescue_true(self):
        msg = _make_graph_message(
            subject="Junk Subject",
            internet_id="junkmail@mail.example.com",
        )
        results = self._run_fetch([], [msg])

        junk_msgs = [m for m in results if m["is_junk_rescue"]]
        self.assertEqual(len(junk_msgs), 1)
        self.assertTrue(junk_msgs[0]["is_junk_rescue"])
        self.assertEqual(junk_msgs[0]["subject"], "Junk Subject")

    def test_both_folders_fetched_combined_results(self):
        inbox_msg = _make_graph_message(subject="Inbox Email", internet_id="inbox@mail.example.com")
        junk_msg = _make_graph_message(subject="Junk Email", internet_id="junk@mail.example.com")
        results = self._run_fetch([inbox_msg], [junk_msg])

        self.assertEqual(len(results), 2)
        inbox_msgs = [m for m in results if not m["is_junk_rescue"]]
        junk_msgs = [m for m in results if m["is_junk_rescue"]]
        self.assertEqual(len(inbox_msgs), 1)
        self.assertEqual(len(junk_msgs), 1)

    def test_empty_folders_returns_empty_list(self):
        results = self._run_fetch([], [])
        self.assertEqual(results, [])

    def test_raises_on_auth_failure(self):
        with patch("outlook_client._OPENER") as mock_opener:
            mock_opener.open.side_effect = _make_http_error(401, b'{"error": "invalid_grant"}')
            with self.assertRaises(RuntimeError) as ctx:
                fetch_unread_emails("bad_id", "bad_secret", "bad_token")
        self.assertIn("401", str(ctx.exception))

    def test_html_body_stripped(self):
        msg = _make_graph_message(
            body_content="<html><body><p>Hello <b>world</b></p></body></html>",
            body_type="html",
        )
        results = self._run_fetch([msg], [])
        self.assertNotIn("<", results[0]["body_preview"])
        self.assertIn("Hello", results[0]["body_preview"])

    def test_graph_api_error_returns_empty_for_that_folder(self):
        token_resp = _make_response({"access_token": "test_token"})
        inbox_error = _make_http_error(403, b'{"error": {"code": "ErrorAccessDenied"}}')
        junk_error = _make_http_error(403, b'{"error": {"code": "ErrorAccessDenied"}}')

        with patch("outlook_client._OPENER") as mock_opener:
            mock_opener.open.side_effect = [token_resp, inbox_error, junk_error]
            results, _ = fetch_unread_emails("client_id", "client_secret", "refresh_tok")

        self.assertEqual(results, [])

    def test_received_at_formatted_as_iso8601_utc(self):
        msg = _make_graph_message(received="2026-04-08T15:30:00Z")
        results = self._run_fetch([msg], [])
        self.assertEqual(results[0]["received_at"], "2026-04-08T15:30:00Z")

    def test_missing_internet_message_id_falls_back_to_graph_id(self):
        msg = _make_graph_message(msg_id="graph-abc-123", internet_id="")
        msg["internetMessageId"] = ""
        results = self._run_fetch([msg], [])
        self.assertEqual(results[0]["message_id"], "outlook:graph-abc-123")


if __name__ == "__main__":
    unittest.main()
