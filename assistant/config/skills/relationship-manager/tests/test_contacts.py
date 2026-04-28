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


def test_update_command_patches_field():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["update", "--name", "Alice Chen", "--field", "context", "--value", "Partner at Sequoia now"])
            mock_d1.update_contact.assert_called_once_with("Alice Chen", "context", "Partner at Sequoia now")
            assert "Updated: Alice Chen" in out.getvalue()


def test_delete_command_removes_contact():
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1:
            mock_d1 = MagicMock()
            MockD1.return_value = mock_d1
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["delete", "--name", "Alice Chen"])
            mock_d1.delete_contact.assert_called_once_with("Alice Chen")
            assert "Deleted: Alice Chen" in out.getvalue()


def test_sync_calendar_updates_matching_contacts():
    events = [
        {
            "summary": "Sequoia Partner Meeting",
            "start": "2026-04-18T15:00:00Z",
            "end": "2026-04-18T16:00:00Z",
            "attendees": ["alice@example.com", "other@external.com"],
        }
    ]
    contact_rows = [
        {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
         "type": "professional", "last_contact": None, "context": "", "created_at": "2026-04-01"},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com",
         "type": "personal", "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = contact_rows
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["sync-calendar", "--days", "1"])
            mock_d1.touch_last_contact.assert_called_once_with("Alice Chen", "2026-04-18")
            result = out.getvalue()
            assert "Alice Chen" in result
            assert "1 contact" in result


def test_sync_calendar_auto_add_creates_new_contacts():
    events = [
        {
            "summary": "Intro call",
            "start": "2026-04-18T15:00:00Z",
            "end": "2026-04-18T16:00:00Z",
            "attendees": ["alice@example.com", "newperson@startup.com"],
        }
    ]
    contact_rows = [
        {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
         "type": "professional", "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = contact_rows
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["sync-calendar", "--days", "1", "--auto-add"])
            mock_d1.upsert_contact.assert_called_once_with(
                "Newperson", "newperson@startup.com", "professional", "Auto-added from calendar"
            )
            assert "added 1 new" in out.getvalue()
            assert "Newperson" in out.getvalue()


def test_sync_calendar_auto_add_skips_owner_email():
    events = [
        {
            "summary": "Solo prep",
            "start": "2026-04-18T15:00:00Z",
            "end": "2026-04-18T16:00:00Z",
            "attendees": ["me@owner.com", "newperson@startup.com"],
        }
    ]
    with patch.dict(os.environ, {**_ENV, "MAHLER_OWNER_EMAIL": "me@owner.com"}):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = []
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["sync-calendar", "--days", "1", "--auto-add"])
            upsert_calls = mock_d1.upsert_contact.call_args_list
            upserted_emails = [c.args[1] for c in upsert_calls]
            assert "me@owner.com" not in upserted_emails
            assert "newperson@startup.com" in upserted_emails


def test_sync_calendar_without_auto_add_does_not_create_contacts():
    events = [
        {
            "summary": "Meeting",
            "start": "2026-04-18T15:00:00Z",
            "end": "2026-04-18T16:00:00Z",
            "attendees": ["newperson@startup.com"],
        }
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = []
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            import contacts
            contacts.main(["sync-calendar", "--days", "1"])
            mock_d1.upsert_contact.assert_not_called()


def test_sync_calendar_prints_no_matches_when_none():
    events = [
        {
            "summary": "Internal sync",
            "start": "2026-04-18T10:00:00Z",
            "end": "2026-04-18T11:00:00Z",
            "attendees": ["stranger@nowhere.com"],
        }
    ]
    contact_rows = [
        {"id": 1, "name": "Alice Chen", "email": "alice@example.com",
         "type": "professional", "last_contact": None, "context": "", "created_at": "2026-04-01"},
    ]
    with patch.dict(os.environ, _ENV):
        with patch("contacts.D1Client") as MockD1, patch("contacts.gcal_client") as mock_gcal:
            mock_d1 = MagicMock()
            mock_d1.list_contacts.return_value = contact_rows
            MockD1.return_value = mock_d1
            mock_gcal.refresh_access_token.return_value = "access_tok"
            mock_gcal.list_events.return_value = events
            out = io.StringIO()
            with patch("sys.stdout", out):
                import contacts
                contacts.main(["sync-calendar", "--days", "1"])
            mock_d1.touch_last_contact.assert_not_called()
            assert "0 contacts" in out.getvalue()
