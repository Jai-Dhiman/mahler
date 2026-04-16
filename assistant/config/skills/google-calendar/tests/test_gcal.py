import pathlib
import sys
import os
import io
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))


class TestGcalListCommand(unittest.TestCase):

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_list_prints_start_and_summary(self, mock_list, mock_refresh):
        mock_list.return_value = [{
            "id": "e1", "summary": "Team standup",
            "start": "2026-04-16T15:00:00Z", "end": "2026-04-16T15:30:00Z",
            "attendees": ["alice@x.com"], "description": "",
        }]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["list", "--days", "1"])
        output = captured.getvalue()
        self.assertIn("Team standup", output)
        self.assertIn("2026-04-16T15:00:00Z", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events", return_value=[])
    def test_list_prints_no_events_message_when_empty(self, mock_list, mock_refresh):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["list", "--days", "1"])
        self.assertIn("No upcoming events", captured.getvalue())

    @patch.dict(os.environ, {}, clear=True)
    def test_list_raises_when_client_id_missing(self):
        from gcal import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["list", "--days", "1"])
        self.assertIn("GMAIL_CLIENT_ID", str(ctx.exception))


class TestGcalCreateCommand(unittest.TestCase):

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.create_event")
    def test_create_prints_confirmation_line(self, mock_create, mock_refresh):
        mock_create.return_value = {
            "id": "new-evt-abc", "summary": "Lunch with Alice",
            "start": "2026-04-17T12:00:00Z", "end": "2026-04-17T13:00:00Z",
            "attendees": [], "description": "",
        }
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["create", "--title", "Lunch with Alice",
                  "--start", "2026-04-17T12:00:00Z",
                  "--end", "2026-04-17T13:00:00Z"])
        output = captured.getvalue()
        self.assertIn("Created:", output)
        self.assertIn("Lunch with Alice", output)
        self.assertIn("new-evt-abc", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.create_event")
    def test_create_passes_attendee_list_to_create_event(self, mock_create, mock_refresh):
        mock_create.return_value = {
            "id": "evt-x", "summary": "Sync", "start": "2026-04-17T14:00:00Z",
            "end": "2026-04-17T14:30:00Z", "attendees": [], "description": "",
        }
        with patch("sys.stdout", io.StringIO()):
            from gcal import main
            main(["create", "--title", "Sync",
                  "--start", "2026-04-17T14:00:00Z", "--end", "2026-04-17T14:30:00Z",
                  "--attendees", "alice@x.com,bob@y.com"])
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["attendees"], ["alice@x.com", "bob@y.com"])


if __name__ == "__main__":
    unittest.main()
