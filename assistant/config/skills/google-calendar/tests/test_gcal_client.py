import json
import pathlib
import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from gcal_client import refresh_access_token, list_events, create_event


def _make_response(body, status=200):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_http_error(code, body_bytes):
    return urllib.error.HTTPError(
        url="https://oauth2.googleapis.com/token",
        code=code, msg="Error", hdrs=MagicMock(), fp=BytesIO(body_bytes),
    )


class TestRefreshAccessToken(unittest.TestCase):

    @patch("gcal_client._OPENER")
    def test_success_returns_access_token(self, mock_opener):
        mock_opener.open.return_value = _make_response({"access_token": "goog_tok_123"})
        result = refresh_access_token("client_id", "client_secret", "refresh_tok")
        self.assertEqual(result, "goog_tok_123")

    @patch("gcal_client._OPENER")
    def test_raises_on_http_401(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(401, b'{"error":"invalid_client"}')
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("bad", "bad", "bad")
        self.assertIn("401", str(ctx.exception))

    @patch("gcal_client._OPENER")
    def test_raises_when_access_token_missing_from_response(self, mock_opener):
        mock_opener.open.return_value = _make_response({"token_type": "Bearer"})
        with self.assertRaises(RuntimeError) as ctx:
            refresh_access_token("cid", "csec", "rtok")
        self.assertIn("access_token", str(ctx.exception))


class TestListEvents(unittest.TestCase):

    @patch("gcal_client._OPENER")
    def test_returns_empty_list_when_no_items(self, mock_opener):
        mock_opener.open.return_value = _make_response({"kind": "calendar#events", "items": []})
        result = list_events("tok", "2026-04-16T00:00:00Z", "2026-04-16T23:59:00Z")
        self.assertEqual(result, [])

    @patch("gcal_client._OPENER")
    def test_returns_normalized_event_with_all_fields(self, mock_opener):
        item = {
            "id": "evt123",
            "summary": "Team standup",
            "start": {"dateTime": "2026-04-16T15:00:00Z"},
            "end": {"dateTime": "2026-04-16T15:30:00Z"},
            "attendees": [{"email": "a@x.com"}, {"email": "b@x.com"}],
            "description": "Daily sync",
        }
        mock_opener.open.return_value = _make_response({"items": [item]})
        result = list_events("tok", "2026-04-16T00:00:00Z", "2026-04-16T23:59:00Z")
        self.assertEqual(len(result), 1)
        evt = result[0]
        self.assertEqual(evt["id"], "evt123")
        self.assertEqual(evt["summary"], "Team standup")
        self.assertEqual(evt["start"], "2026-04-16T15:00:00Z")
        self.assertEqual(evt["end"], "2026-04-16T15:30:00Z")
        self.assertEqual(evt["attendees"], ["a@x.com", "b@x.com"])
        self.assertEqual(evt["description"], "Daily sync")

    @patch("gcal_client._OPENER")
    def test_all_day_event_uses_date_field_and_empty_attendees(self, mock_opener):
        item = {
            "id": "evt456",
            "summary": "Birthday",
            "start": {"date": "2026-04-20"},
            "end": {"date": "2026-04-21"},
        }
        mock_opener.open.return_value = _make_response({"items": [item]})
        result = list_events("tok", "2026-04-16T00:00:00Z", "2026-04-21T00:00:00Z")
        self.assertEqual(result[0]["start"], "2026-04-20")
        self.assertEqual(result[0]["attendees"], [])

    @patch("gcal_client._OPENER")
    def test_raises_on_403(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(403, b'{"error":"forbidden"}')
        with self.assertRaises(RuntimeError) as ctx:
            list_events("tok", "2026-04-16T00:00:00Z", "2026-04-16T23:59:00Z")
        self.assertIn("403", str(ctx.exception))


class TestCreateEvent(unittest.TestCase):

    @patch("gcal_client._OPENER")
    def test_returns_normalized_event_with_id_and_summary(self, mock_opener):
        mock_opener.open.return_value = _make_response({
            "id": "new-evt-abc",
            "summary": "Lunch with Alice",
            "start": {"dateTime": "2026-04-17T12:00:00Z"},
            "end": {"dateTime": "2026-04-17T13:00:00Z"},
        })
        result = create_event(
            access_token="tok",
            summary="Lunch with Alice",
            start="2026-04-17T12:00:00Z",
            end="2026-04-17T13:00:00Z",
        )
        self.assertEqual(result["id"], "new-evt-abc")
        self.assertEqual(result["summary"], "Lunch with Alice")
        self.assertEqual(result["start"], "2026-04-17T12:00:00Z")

    @patch("gcal_client._OPENER")
    def test_raises_on_403(self, mock_opener):
        mock_opener.open.side_effect = _make_http_error(403, b'{"error":"forbidden"}')
        with self.assertRaises(RuntimeError) as ctx:
            create_event("tok", "Meeting", "2026-04-17T12:00:00Z", "2026-04-17T13:00:00Z")
        self.assertIn("403", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
