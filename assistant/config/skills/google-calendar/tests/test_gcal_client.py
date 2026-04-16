import json
import pathlib
import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from gcal_client import refresh_access_token


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


if __name__ == "__main__":
    unittest.main()
