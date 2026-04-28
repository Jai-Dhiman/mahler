import pathlib
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from orchestrate import (
    _parse_gcal_output,
    _should_skip_wiki,
    _extract_wiki_keywords,
    check_dedup,
    fetch_email_context,
    fetch_open_tasks,
    synthesize_brief,
    run_prep,
)

import os


class TestParseGcalOutput(unittest.TestCase):

    def test_parses_event_with_attendees_and_description(self):
        output = (
            "evt_abc123  2026-04-28T17:30:00Z  Intro Call with Curium\n"
            "  Attendees: rakhee@curium.ai, jai@example.com\n"
            "  Description: Discuss GEO initiative and partnership | Location: NYC\n"
        )
        event = _parse_gcal_output(output)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_id, "evt_abc123")
        self.assertEqual(event.start, "2026-04-28T17:30:00Z")
        self.assertEqual(event.title, "Intro Call with Curium")
        self.assertIn("rakhee@curium.ai", event.attendees)
        self.assertIn("jai@example.com", event.attendees)
        self.assertIn("GEO", event.description)
        self.assertIn("NYC", event.description)

    def test_parses_rich_description_with_job_details(self):
        output = (
            "evt_xyz  2026-04-29T16:30:00Z  interview between Danial and Jai\n"
            "  Attendees: danial@222.place\n"
            "  Description: Location: New York | Size: 16 people | Vertical: Consumer | "
            "Website: https://222.place | Title: Technical Support Lead | Salary: $65K - $100K\n"
        )
        event = _parse_gcal_output(output)
        self.assertIsNotNone(event)
        self.assertIn("Technical Support Lead", event.description)
        self.assertIn("222.place", event.description)
        self.assertIn("New York", event.description)

    def test_returns_none_when_no_meetings(self):
        self.assertIsNone(_parse_gcal_output("No meetings in window.\n"))

    def test_parses_event_without_attendees(self):
        output = "evt_xyz  2026-04-28T18:00:00Z  Team standup\n"
        event = _parse_gcal_output(output)
        self.assertIsNotNone(event)
        self.assertEqual(event.attendees, [])
        self.assertEqual(event.description, "")


class TestSkipWiki(unittest.TestCase):

    def test_skips_social_title_without_description(self):
        self.assertTrue(_should_skip_wiki("sync", ""))
        self.assertTrue(_should_skip_wiki("1:1", ""))
        self.assertTrue(_should_skip_wiki("standup", ""))

    def test_does_not_skip_social_title_with_description(self):
        self.assertFalse(_should_skip_wiki("sync", "Discuss Q2 roadmap"))

    def test_skips_name_only_meeting(self):
        self.assertTrue(_should_skip_wiki("meeting with Riley", ""))
        self.assertTrue(_should_skip_wiki("call with Jane Doe", ""))

    def test_does_not_skip_substantive_title(self):
        self.assertFalse(_should_skip_wiki("Intro Call with Curium", ""))
        self.assertFalse(_should_skip_wiki("Product roadmap review", ""))


class TestExtractWikiKeywords(unittest.TestCase):

    def test_extracts_two_content_words(self):
        kws = _extract_wiki_keywords("Intro Call with Curium", "GEO initiative")
        self.assertIn("Curium", kws)
        self.assertLessEqual(len(kws), 2)

    def test_skips_stop_words(self):
        kws = _extract_wiki_keywords("Meeting with the Team", "")
        self.assertNotIn("Meeting", kws)
        self.assertNotIn("with", kws)
        self.assertNotIn("the", kws)


class TestCheckDedup(unittest.TestCase):

    def _make_runner(self, returncode):
        result = MagicMock()
        result.returncode = returncode
        result.stderr = ""
        return lambda cmd, **kw: result

    def test_returns_true_when_not_briefed(self):
        self.assertTrue(check_dedup("evt1", self._make_runner(0)))

    def test_returns_false_when_already_briefed(self):
        self.assertFalse(check_dedup("evt1", self._make_runner(1)))

    def test_raises_on_unexpected_exit_code(self):
        with self.assertRaises(RuntimeError):
            check_dedup("evt1", self._make_runner(2))


class TestFetchEmailContext(unittest.TestCase):

    def _make_runner(self, stdout, returncode=0):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = ""
        return lambda cmd, **kw: result

    def test_returns_none_when_no_flagged_emails(self):
        runner = self._make_runner("No recent flagged emails from these contacts.\n")
        self.assertIsNone(fetch_email_context(["ext@x.com"], runner))

    def test_returns_email_text(self):
        runner = self._make_runner("Recent emails (last 7 days):\n  [URGENT] ext@x.com: Invoice — please review")
        result = fetch_email_context(["ext@x.com"], runner)
        self.assertIn("Invoice", result)

    def test_returns_none_when_no_external_attendees(self):
        os.environ["MAHLER_OWNER_EMAIL"] = "me@example.com"
        result = fetch_email_context(["me@example.com"], lambda *a, **kw: None)
        del os.environ["MAHLER_OWNER_EMAIL"]
        self.assertIsNone(result)


class TestFetchOpenTasks(unittest.TestCase):

    def test_parses_task_titles(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "[abc123] Send Q2 memo\n  (status=Not started, priority=High)\n[def456] Review contract\n"
        result.stderr = ""
        runner = lambda cmd, **kw: result
        tasks = fetch_open_tasks("2026-04-28", runner)
        self.assertIn("Send Q2 memo", tasks)
        self.assertIn("Review contract", tasks)

    def test_returns_empty_on_failure(self):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "error"
        tasks = fetch_open_tasks("2026-04-28", lambda cmd, **kw: result)
        self.assertEqual(tasks, [])


class TestSynthesizeBrief(unittest.TestCase):

    def test_bullets_normalized(self):
        llm = lambda prompt: "- First point\n- Second point\n- Third point"
        result = synthesize_brief(
            title="Intro Call",
            start="2026-04-28T17:30:00Z",
            attendees=["x@x.com"],
            description="",
            emails=None,
            tasks=[],
            wiki=None,
            crm=None,
            llm_caller=llm,
        )
        for line in result.splitlines():
            self.assertTrue(line.startswith("•"), f"Expected bullet, got: {line!r}")

    def test_already_bulleted_lines_pass_through(self):
        llm = lambda prompt: "• First\n• Second\n• Third"
        result = synthesize_brief(
            title="Test", start="2026-04-28T17:00:00Z",
            attendees=[], description="", emails=None, tasks=[], wiki=None, crm=None,
            llm_caller=llm,
        )
        self.assertIn("• First", result)

    def test_max_five_bullets(self):
        llm = lambda prompt: "\n".join(f"• Point {i}" for i in range(10))
        result = synthesize_brief(
            title="Test", start="2026-04-28T17:00:00Z",
            attendees=[], description="", emails=None, tasks=[], wiki=None, crm=None,
            llm_caller=llm,
        )
        self.assertLessEqual(len(result.splitlines()), 5)


class TestRunPrep(unittest.TestCase):

    def _cmd_matches(self, cmd, key):
        return any(key in str(c) for c in cmd)

    def test_returns_no_work_when_no_meetings(self):
        no_meeting_result = MagicMock(returncode=0, stdout="No meetings in window.\n", stderr="")

        def runner(cmd, **kw):
            if self._cmd_matches(cmd, "gcal.py"):
                return no_meeting_result
            raise AssertionError(f"Unexpected cmd: {cmd}")

        result = run_prep(runner=runner, llm_caller=lambda p: "• bullet")
        self.assertEqual(result, "NO_WORK")

    def test_returns_no_work_when_already_briefed(self):
        gcal_result = MagicMock(returncode=0, stdout="evt1  2026-04-28T17:30:00Z  Test Meeting\n", stderr="")
        dedup_result = MagicMock(returncode=1, stdout="", stderr="")

        def runner(cmd, **kw):
            if self._cmd_matches(cmd, "gcal.py"):
                return gcal_result
            if self._cmd_matches(cmd, "dedup.py"):
                return dedup_result
            raise AssertionError(f"Unexpected cmd: {cmd}")

        result = run_prep(runner=runner, llm_caller=lambda p: "• bullet")
        self.assertEqual(result, "NO_WORK")

    def test_full_flow_returns_brief_sent(self):
        responses = {
            "gcal.py": MagicMock(returncode=0, stdout="evt42  2026-04-28T17:30:00Z  Product Review\n  Attendees: ext@x.com\n", stderr=""),
            "dedup.py": MagicMock(returncode=0, stdout="Logged: evt42", stderr=""),
            "email_context.py": MagicMock(returncode=0, stdout="No recent flagged emails from these contacts.", stderr=""),
            "tasks.py": MagicMock(returncode=0, stdout="", stderr=""),
            "wiki.py": MagicMock(returncode=0, stdout="No results.", stderr=""),
            "contacts.py": MagicMock(returncode=1, stdout="", stderr="not found"),
            "post_brief.py": MagicMock(returncode=0, stdout="Brief sent.", stderr=""),
        }

        def runner(cmd, **kw):
            for key, val in responses.items():
                if any(key in str(c) for c in cmd):
                    return val
            raise AssertionError(f"Unexpected cmd: {cmd}")

        result = run_prep(runner=runner, llm_caller=lambda p: "• First bullet\n• Second bullet\n• Third bullet")
        self.assertEqual(result, "Brief sent: Product Review")


if __name__ == "__main__":
    unittest.main()
