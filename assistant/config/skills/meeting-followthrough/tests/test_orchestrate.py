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


class TestGenerateActionItems(unittest.TestCase):
    def test_no_action_items_llm_response_returns_empty_list(self):
        import orchestrate
        llm = MagicMock(return_value="no action items")
        result = orchestrate.generate_action_items(
            summary="a meeting happened",
            attendees=[],
            crm_context={},
            open_tasks=[],
            llm_caller=llm,
        )
        self.assertEqual(result, [])

    def test_single_task_line_with_attendee_and_priority(self):
        import orchestrate
        llm = MagicMock(return_value="TASK: [Alice] Send Q2 memo | PRIORITY: High")
        result = orchestrate.generate_action_items(
            summary="meeting",
            attendees=[],
            crm_context={},
            open_tasks=[],
            llm_caller=llm,
        )
        self.assertEqual(
            result,
            [{"title": "[Alice] Send Q2 memo", "priority": "High", "attendee": "Alice"}],
        )

    def test_open_tasks_flow_into_prompt_for_dedup(self):
        import orchestrate
        captured = {}
        def llm(prompt):
            captured["prompt"] = prompt
            return "no action items"
        orchestrate.generate_action_items(
            summary="meeting",
            attendees=[],
            crm_context={},
            open_tasks=["Follow up with Alice", "Review Q2 deck"],
            llm_caller=llm,
        )
        self.assertIn("Follow up with Alice", captured["prompt"])
        self.assertIn("Review Q2 deck", captured["prompt"])


class TestCrmFanout(unittest.TestCase):
    def test_crm_context_for_external_attendee_flows_to_llm(self):
        import orchestrate
        row = {
            "recording_id": 51,
            "title": "Sync",
            "attendees": '[{"name": "Alice", "email": "alice@ext.com", "is_external": true}]',
            "summary": "we talked",
        }
        def runner(argv, **_):
            if "summarize" in argv and "Alice" in argv:
                return MagicMock(returncode=0, stdout="Alice is a senior PM", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        captured_prompt = {}
        def llm(prompt):
            captured_prompt["p"] = prompt
            return "no action items"
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=MagicMock(),
            d1_client=MagicMock(),
        )
        self.assertIn("Alice is a senior PM", captured_prompt["p"])


if __name__ == "__main__":
    unittest.main()
