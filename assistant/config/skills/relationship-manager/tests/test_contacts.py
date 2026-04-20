import io
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

_ENV = {
    "CF_ACCOUNT_ID": "acct1",
    "CF_D1_DATABASE_ID": "db1",
    "CF_API_TOKEN": "tok1",
    "NOTION_API_TOKEN": "ntok",
    "NOTION_DATABASE_ID": "ndb1",
    "GMAIL_CLIENT_ID": "cid",
    "GMAIL_CLIENT_SECRET": "csec",
    "GMAIL_REFRESH_TOKEN": "rtok",
}


def test_add_command_creates_contact_and_prints_confirmation():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main([
                    "add",
                    "--name", "Alice Chen",
                    "--email", "alice@example.com",
                    "--type", "professional",
                    "--context", "Works at Sequoia",
                ])
            mock_d1.ensure_table.assert_called_once()
            mock_d1.upsert_contact.assert_called_once_with(
                "Alice Chen", "alice@example.com", "professional", "Works at Sequoia"
            )
            assert "Added: Alice Chen (professional)" in out.getvalue()


def test_add_command_context_defaults_to_empty_string():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main([
                    "add",
                    "--name", "Bob Smith",
                    "--email", "bob@example.com",
                    "--type", "personal",
                ])
            mock_d1.upsert_contact.assert_called_once_with(
                "Bob Smith", "bob@example.com", "personal", ""
            )


def test_summarize_shows_contact_card_and_open_tasks():
    contact_row = {
        "id": 1, "name": "Alice Chen", "email": "alice@example.com",
        "type": "professional", "last_contact": "2026-04-15",
        "context": "Works at Sequoia", "created_at": "2026-04-01",
    }
    tasks = [{"id": "p1", "title": "[Alice Chen] Send IC memo", "status": "In progress", "due": "2026-04-25", "priority": "High"}]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.NotionClient") as MockNotion:
            mock_d1 = MagicMock()
            mock_d1.get_contact.return_value = contact_row
            MockD1.return_value = mock_d1
            mock_notion = MagicMock()
            mock_notion.list_tasks_for_contact.return_value = tasks
            MockNotion.return_value = mock_notion
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["summarize", "--name", "Alice Chen"])
            result = out.getvalue()
            assert "Alice Chen (professional)" in result
            assert "alice@example.com" in result
            assert "2026-04-15" in result
            assert "Works at Sequoia" in result
            assert "[Alice Chen] Send IC memo" in result
            mock_notion.list_tasks_for_contact.assert_called_once_with("Alice Chen")


def test_summarize_shows_none_when_no_tasks():
    contact_row = {
        "id": 1, "name": "Bob Smith", "email": "bob@example.com",
        "type": "personal", "last_contact": None,
        "context": "", "created_at": "2026-04-01",
    }
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.NotionClient") as MockNotion:
            mock_d1 = MagicMock()
            mock_d1.get_contact.return_value = contact_row
            MockD1.return_value = mock_d1
            mock_notion = MagicMock()
            mock_notion.list_tasks_for_contact.return_value = []
            MockNotion.return_value = mock_notion
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["summarize", "--name", "Bob Smith"])
            result = out.getvalue()
            assert "never" in result
            assert "none" in result


def test_list_command_prints_all_contacts():
    rows = [
        {"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
         "last_contact": "2026-04-15", "context": "", "created_at": "2026-04-01"},
        {"id": 2, "name": "Bob Smith", "email": "b@x.com", "type": "personal",
         "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = rows
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["list"])
            result = out.getvalue()
            assert "Alice Chen (professional)" in result
            assert "2026-04-15" in result
            assert "Bob Smith (personal)" in result
            assert "never" in result
            mock_d1.list_contacts.assert_called_once_with(type=None)


def test_list_command_filters_by_type():
    rows = [{"id": 1, "name": "Alice Chen", "email": "a@x.com", "type": "professional",
             "last_contact": None, "context": "", "created_at": "2026-04-01"}]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = rows
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["list", "--type", "professional"])
            mock_d1.list_contacts.assert_called_once_with(type="professional")


def test_talked_to_updates_last_contact_to_today():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                with patch("contacts.date") as mock_date:
                    mock_date.today.return_value.isoformat.return_value = "2026-04-19"
                    import contacts
                    contacts.main(["talked-to", "--name", "Alice Chen"])
            mock_d1.touch_last_contact.assert_called_once_with("Alice Chen", "2026-04-19")
            assert "Noted: talked to Alice Chen on 2026-04-19" in out.getvalue()
