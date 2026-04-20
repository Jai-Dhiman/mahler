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


def test_list_contacts_returns_all():
    rows = [
        {"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
         "last_contact": None, "context": "", "created_at": "2026-04-19"},
        {"id": 2, "name": "Bob Smith", "email": "b@x.com", "type": "personal",
         "last_contact": None, "context": "", "created_at": "2026-04-19"},
    ]
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=rows)
        client = D1Client("acct1", "db1", "tok1")
        result = client.list_contacts()
        assert len(result) == 2
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "WHERE type" not in body["sql"]


def test_list_contacts_filters_by_type():
    rows = [{"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
             "last_contact": None, "context": "", "created_at": "2026-04-19"}]
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response(rows=rows)
        client = D1Client("acct1", "db1", "tok1")
        result = client.list_contacts(type="professional")
        assert len(result) == 1
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "WHERE type = ?" in body["sql"]
        assert body["params"] == ["professional"]


def test_touch_last_contact_sends_update_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        client = D1Client("acct1", "db1", "tok1")
        client.touch_last_contact("Alice Chen", "2026-04-19")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "UPDATE contacts SET last_contact = ?" in body["sql"]
        assert "lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["2026-04-19", "Alice Chen"]


def test_update_contact_sends_correct_field_sql():
    with patch("d1_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_d1_response()
        client = D1Client("acct1", "db1", "tok1")
        client.update_contact("Alice Chen", "context", "Partner at Sequoia now")
        req = mock_opener.open.call_args[0][0]
        body = json.loads(req.data)
        assert "UPDATE contacts SET context = ?" in body["sql"]
        assert "lower(name) = lower(?)" in body["sql"]
        assert body["params"] == ["Partner at Sequoia now", "Alice Chen"]


def test_update_contact_rejects_unknown_field():
    client = D1Client("acct1", "db1", "tok1")
    with pytest.raises(ValueError, match="Cannot update field"):
        client.update_contact("Alice Chen", "password", "hack")
