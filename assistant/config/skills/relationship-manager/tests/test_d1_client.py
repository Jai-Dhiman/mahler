import json
import pytest
from unittest.mock import patch, MagicMock


def _make_d1_response(rows=None, status=200):
    body = {"success": True, "errors": [], "result": [{"results": rows or []}]}
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_upsert_contact_sends_correct_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        from d1_client import D1Client
        client = D1Client("acct1", "db1", "tok1")
        client.upsert_contact("Alice Chen", "alice@example.com", "professional", "Works at Sequoia")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "INSERT INTO contacts" in body["sql"]
        assert "ON CONFLICT" in body["sql"]
        assert body["params"] == ["Alice Chen", "alice@example.com", "professional", "Works at Sequoia"]
