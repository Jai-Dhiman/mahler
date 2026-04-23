import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestEmptyQueue(unittest.TestCase):
    def test_empty_queue_prints_no_work_and_returns_zero(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = []
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = orchestrate.main(
                argv=[],
                d1_client=d1,
                runner=MagicMock(),
                llm_caller=MagicMock(),
                discord_poster=MagicMock(),
            )
        self.assertEqual(rc, 0)
        self.assertIn("NO_WORK", captured.getvalue())


class TestTrivialMeeting(unittest.TestCase):
    def test_trivial_meeting_posts_summary_and_marks_done(self):
        import orchestrate
        row = {
            "recording_id": 42,
            "title": "Test call",
            "attendees": "[]",
            "summary": "Fathom intro test.",
        }
        d1 = MagicMock()
        runner = MagicMock()
        runner.return_value = MagicMock(returncode=0, stdout="", stderr="")
        llm = MagicMock(return_value="no action items")
        poster = MagicMock()
        result = orchestrate.process_meeting(
            row, runner=runner, llm_caller=llm, discord_poster=poster, d1_client=d1
        )
        expected = (
            "Post-meeting: Test call\n"
            "Action items created:\n"
            "  None\n"
            "CRM updated: No CRM matches"
        )
        self.assertEqual(result, expected)
        poster.assert_called_once_with(expected)
        d1.mark_done.assert_called_once_with(42)


if __name__ == "__main__":
    unittest.main()
