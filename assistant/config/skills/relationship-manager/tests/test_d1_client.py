import json
import pytest
from unittest.mock import patch, MagicMock
from d1_client import D1Client


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
        client = D1Client("acct1", "db1", "tok1")
        client.upsert_contact("Alice Chen", "alice@example.com", "professional", "Works at Sequoia")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "INSERT INTO contacts" in body["sql"]
        assert "ON CONFLICT" in body["sql"]
        assert body["params"] == ["Alice Chen", "alice@example.com", "professional", "Works at Sequoia"]


def test_get_contact_returns_row():
    row = {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
           "type": "professional", "last_contact": None, "context": "Works at Sequoia",
           "created_at": "2026-04-19"}
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=[row])
        client = D1Client("acct1", "db1", "tok1")
        result = client.get_contact("Alice Chen")
        assert result["name"] == "Alice Chen"
        assert result["email"] == "alice@example.com"
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "WHERE lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["Alice Chen"]


def test_get_contact_raises_when_not_found():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=[])
        client = D1Client("acct1", "db1", "tok1")
        with pytest.raises(RuntimeError, match="Contact not found"):
            client.get_contact("Nobody")
