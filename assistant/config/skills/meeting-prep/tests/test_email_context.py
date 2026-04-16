import pathlib
import sys
import os
import io
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))


class TestEmailContext(unittest.TestCase):

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("email_context.D1Client")
    def test_prints_flagged_emails_when_found(self, MockClient):
        MockClient.return_value.query.return_value = [
            {"classification": "URGENT", "from_addr": "alice@x.com",
             "subject": "Q2 Budget", "summary": "Need sign-off by Friday"},
        ]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from email_context import main
            main(["email-context", "--attendees", "alice@x.com"])
        output = captured.getvalue()
        self.assertIn("URGENT", output)
        self.assertIn("alice@x.com", output)
        self.assertIn("Q2 Budget", output)

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("email_context.D1Client")
    def test_prints_no_emails_message_when_none_found(self, MockClient):
        MockClient.return_value.query.return_value = []
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from email_context import main
            main(["email-context", "--attendees", "alice@x.com"])
        self.assertIn("No recent flagged emails", captured.getvalue())

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_when_cf_account_id_missing(self):
        from email_context import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["email-context", "--attendees", "alice@x.com"])
        self.assertIn("CF_ACCOUNT_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
