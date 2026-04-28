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
    @patch("gcal.gcal_client.list_events")
    def test_list_prints_event_id_so_meeting_prep_can_extract_it(self, mock_list, mock_refresh):
        mock_list.return_value = [{
            "id": "evt-abc123", "summary": "Product review",
            "start": "2026-04-20T19:00:00Z", "end": "2026-04-20T20:00:00Z",
            "attendees": [], "description": "",
        }]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["list", "--hours-ahead", "2"])
        self.assertIn("evt-abc123", captured.getvalue())

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


class TestGcalUpcomingCommand(unittest.TestCase):

    def _make_event(self, event_id, summary, minutes_from_now, offset="+00:00"):
        from datetime import datetime, timezone, timedelta
        start = datetime.now(timezone.utc) + timedelta(minutes=minutes_from_now)
        if offset == "+00:00":
            start_str = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        else:
            # simulate PDT offset (-07:00)
            from datetime import timezone as tz
            pdt = tz(timedelta(hours=-7))
            start_pdt = start.astimezone(pdt)
            start_str = start_pdt.strftime("%Y-%m-%dT%H:%M:%S-07:00")
        return {
            "id": event_id, "summary": summary,
            "start": start_str, "end": start_str,
            "attendees": [], "description": "",
        }

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_upcoming_returns_event_in_window(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("evt-in", "In-window meeting", 55)]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        output = captured.getvalue()
        self.assertIn("evt-in", output)
        self.assertIn("In-window meeting", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_upcoming_excludes_event_too_close(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("evt-close", "Too soon", 20)]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        self.assertIn("No meetings in window", captured.getvalue())

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_upcoming_handles_pdt_offset_correctly(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("evt-pdt", "PDT meeting", 55, offset="-07:00")]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        output = captured.getvalue()
        self.assertIn("evt-pdt", output)
        self.assertIn("PDT meeting", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_upcoming_outputs_start_time_in_utc(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("evt-utc", "UTC check", 60, offset="-07:00")]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        output = captured.getvalue()
        # Start time in output must end in Z (UTC), not a PDT offset
        import re
        self.assertRegex(output, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class TestGcalUpcomingDescription(unittest.TestCase):

    def _make_event(self, description):
        from datetime import datetime, timezone, timedelta
        start = datetime.now(timezone.utc) + timedelta(minutes=55)
        return {
            "id": "evt-desc", "summary": "Intro Call",
            "start": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "end": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "attendees": [], "description": description,
        }

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_description_newlines_replaced_with_pipe(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("Location: New York\nSize: 16 people\nVertical: Consumer")]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        output = captured.getvalue()
        self.assertIn("Description:", output)
        self.assertIn("New York", output)
        self.assertIn("16 people", output)
        self.assertNotIn("\n  Location", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_html_stripped_from_description(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event(
            "<p>Meeting with Riley:</p><p><b>Vibecode</b>: AI-powered app builder &amp; more</p>"
        )]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        output = captured.getvalue()
        self.assertIn("Vibecode", output)
        self.assertIn("AI-powered app builder & more", output)
        self.assertNotIn("<p>", output)
        self.assertNotIn("<b>", output)
        self.assertNotIn("&amp;", output)

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_description_truncated_at_800_chars(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("x" * 1000)]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        desc_line = [l for l in captured.getvalue().splitlines() if "Description:" in l][0]
        desc_value = desc_line.split("Description:", 1)[1].strip()
        self.assertLessEqual(len(desc_value), 800)


class TestGcalUpcomingMultipleEvents(unittest.TestCase):

    def _make_event(self, event_id, summary, minutes_from_now):
        from datetime import datetime, timezone, timedelta
        start = datetime.now(timezone.utc) + timedelta(minutes=minutes_from_now)
        return {
            "id": event_id, "summary": summary,
            "start": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "end": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "attendees": [], "description": f"Description of {summary}",
        }

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_only_first_matching_event_returned(self, mock_list, mock_refresh):
        mock_list.return_value = [
            self._make_event("evt-first", "First Meeting", 50),
            self._make_event("evt-second", "Second Meeting", 65),
        ]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75"])
        output = captured.getvalue()
        self.assertIn("evt-first", output)
        self.assertNotIn("evt-second", output)
        self.assertIn("Description of First Meeting", output)
        self.assertNotIn("Description of Second Meeting", output)


class TestGcalUpcomingSkipKeywords(unittest.TestCase):

    def _make_event(self, event_id, summary, minutes_from_now, description=""):
        from datetime import datetime, timezone, timedelta
        start = datetime.now(timezone.utc) + timedelta(minutes=minutes_from_now)
        return {
            "id": event_id, "summary": summary,
            "start": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "end": start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "attendees": [], "description": description,
        }

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_skip_keywords_blocks_rehearsal(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("evt-r", "Orchestra Rehearsal", 55)]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75",
                  "--skip-keywords", "orchestra,rehearsal,bohemian,jinks,encampment"])
        self.assertIn("No meetings in window", captured.getvalue())

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_skip_keywords_allows_real_meeting(self, mock_list, mock_refresh):
        mock_list.return_value = [self._make_event("evt-m", "Product Review", 55)]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75",
                  "--skip-keywords", "orchestra,rehearsal,bohemian,jinks,encampment"])
        self.assertIn("evt-m", captured.getvalue())
        self.assertIn("Product Review", captured.getvalue())

    @patch.dict(os.environ, {
        "GMAIL_CLIENT_ID": "test_cid",
        "GMAIL_CLIENT_SECRET": "test_csec",
        "GMAIL_REFRESH_TOKEN": "test_rtok",
    })
    @patch("gcal.gcal_client.refresh_access_token", return_value="access_tok")
    @patch("gcal.gcal_client.list_events")
    def test_skip_keywords_matches_description_too(self, mock_list, mock_refresh):
        mock_list.return_value = [
            self._make_event("evt-d", "Club Event", 55, description="Bohemian encampment planning")
        ]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from gcal import main
            main(["upcoming", "--min-minutes", "45", "--max-minutes", "75",
                  "--skip-keywords", "orchestra,rehearsal,bohemian,jinks,encampment"])
        self.assertIn("No meetings in window", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
