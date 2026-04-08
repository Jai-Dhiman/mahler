import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
import json
import base64
import urllib.error

sys.path.insert(0, "scripts")

from gmail_client import refresh_access_token, fetch_unread_emails


def _make_response(body: dict | list, status: int = 200) -> MagicMock:
    raw = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8").rstrip("=")


def _make_message(msg_id: str = "abc123") -> dict:
    return {
        "id": msg_id,
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": "Hello world"},
                {"name": "Date", "value": "Mon, 07 Apr 2026 10:00:00 +0000"},
            ],
            "body": {"data": _b64("This is the email body content.")},
        },
    }


def _make_http_error(code: int, body: bytes, url: str = "https://example.com") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url=url, code=code, msg="Error", hdrs=MagicMock(), fp=BytesIO(body))


class TestRefreshAccessToken(unittest.TestCase):

    @patch("gmail_client._OPENER")
    def test_success_returns_token(self, mock_opener):
        mock_opener.open.return_value = _make_response({"access_token": "tok123"})
        result = refresh_access_token("client_id", "client_secret", "refresh_tok")
        self.assertEqual(result, "tok123")

    @patch("gmail_client._OPENER")
    def test_raises_on_401(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(401, b'{"error": "invalid_client"}')
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("bad_id", "bad_secret", "bad_token")
        self.assertIn("401", str(ctx.exception))

    @patch("gmail_client._OPENER")
    def test_raises_if_access_token_missing(self, mock_opener):
        mock_opener.open.return_value = _make_response({"token_type": "Bearer"})
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("client_id", "client_secret", "refresh_tok")
        self.assertIn("access_token", str(ctx.exception))


class TestFetchUnreadEmails(unittest.TestCase):

    @patch("gmail_client._OPENER")
    def test_returns_empty_list_when_no_messages_field(self, mock_opener):
        mock_opener.open.return_value = _make_response({})
        result = fetch_unread_emails("access_tok")
        self.assertEqual(result, [])

    @patch("gmail_client._OPENER")
    def test_correct_number_of_api_calls(self, mock_opener):
        list_resp = _make_response({"messages": [{"id": "id1"}, {"id": "id2"}]})
        msg1_resp = _make_response(_make_message("id1"))
        msg2_resp = _make_response(_make_message("id2"))
        mock_opener.open.side_effect = [list_resp, msg1_resp, msg2_resp]

        result = fetch_unread_emails("access_tok")

        self.assertEqual(mock_opener.open.call_count, 3)
        self.assertEqual(len(result), 2)

    @patch("gmail_client._OPENER")
    def test_returns_correctly_shaped_email_message(self, mock_opener):
        list_resp = _make_response({"messages": [{"id": "abc123"}]})
        msg_resp = _make_response(_make_message("abc123"))
        mock_opener.open.side_effect = [list_resp, msg_resp]

        results = fetch_unread_emails("access_tok")

        self.assertEqual(len(results), 1)
        msg = results[0]
        self.assertEqual(msg["message_id"], "abc123")
        self.assertEqual(msg["source"], "gmail")
        self.assertEqual(msg["from_addr"], "sender@example.com")
        self.assertEqual(msg["subject"], "Hello world")
        self.assertEqual(msg["received_at"], "2026-04-07T10:00:00Z")
        self.assertIn("This is the email body content.", msg["body_preview"])
        self.assertFalse(msg["is_junk_rescue"])
        self.assertIn("from", msg["headers"])

    @patch("gmail_client._OPENER")
    def test_raises_on_non_200_from_list_endpoint(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(
            403,
            b'{"error": "insufficientPermissions"}',
            url="https://gmail.googleapis.com/gmail/v1/users/me/messages",
        )
        with self.assertRaises(RuntimeError) as ctx:
            fetch_unread_emails("bad_token")
        self.assertIn("403", str(ctx.exception))

    @patch("gmail_client._OPENER")
    def test_rfc2047_encoded_subject_decoded(self, mock_opener):
        encoded_subject = "=?utf-8?b?SmFpJ3MgSW52b2ljZQ==?="
        msg = _make_message("enc1")
        for h in msg["payload"]["headers"]:
            if h["name"] == "Subject":
                h["value"] = encoded_subject

        list_resp = _make_response({"messages": [{"id": "enc1"}]})
        msg_resp = _make_response(msg)
        mock_opener.open.side_effect = [list_resp, msg_resp]

        results = fetch_unread_emails("access_tok")
        self.assertEqual(results[0]["subject"], "Jai's Invoice")


if __name__ == "__main__":
    unittest.main()
