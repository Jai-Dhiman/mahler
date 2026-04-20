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
