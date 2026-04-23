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

    def test_owner_attendee_not_fetched_from_crm(self):
        import orchestrate
        row = {
            "recording_id": 52,
            "title": "Solo",
            "attendees": '[{"name": "Jai Dhiman", "email": "jai@mahler.local", "is_external": false}]',
            "summary": "solo prep",
        }
        calls: list[list[str]] = []
        def runner(argv, **_):
            calls.append(argv)
            return MagicMock(returncode=0, stdout="", stderr="")
        captured = {}
        def llm(prompt):
            captured["p"] = prompt
            return "no action items"
        with patch.dict("os.environ", {"MAHLER_OWNER_EMAIL": "jai@mahler.local"}):
            orchestrate.process_meeting(
                row,
                runner=runner,
                llm_caller=llm,
                discord_poster=MagicMock(),
                d1_client=MagicMock(),
            )
        summarize_calls = [c for c in calls if "summarize" in c]
        self.assertEqual(summarize_calls, [])
        self.assertNotIn("jai@mahler.local", captured["p"])


class TestOpenTasksFlow(unittest.TestCase):
    def test_open_tasks_from_tasks_list_appear_in_llm_prompt(self):
        import orchestrate
        row = {
            "recording_id": 60,
            "title": "Planning",
            "attendees": "[]",
            "summary": "planning chat",
        }
        def runner(argv, **_):
            if "list" in argv and "--status" in argv:
                # Matches real tasks.py list output: "[uuid] Title\n  (meta)"
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "[abc-111] Follow up with Alice\n"
                        "  (status=Not started, priority=Medium)\n"
                        "[abc-222] Review Q2 deck\n"
                        "  (status=Not started)\n"
                    ),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")
        captured = {}
        def llm(prompt):
            captured["p"] = prompt
            return "no action items"
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=MagicMock(),
            d1_client=MagicMock(),
        )
        self.assertIn("Follow up with Alice", captured["p"])
        self.assertIn("Review Q2 deck", captured["p"])


class TestTasksCreate(unittest.TestCase):
    def test_each_action_item_becomes_tasks_create_call(self):
        import orchestrate
        row = {
            "recording_id": 70,
            "title": "1:1",
            "attendees": "[]",
            "summary": "1:1 summary",
        }
        calls: list[list[str]] = []
        def runner(argv, **_):
            calls.append(argv)
            return MagicMock(returncode=0, stdout="", stderr="")
        llm = MagicMock(return_value=(
            "TASK: [Alice] Send Q2 memo | PRIORITY: High\n"
            "TASK: Follow up on Series A | PRIORITY: Medium"
        ))
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=MagicMock(),
            d1_client=MagicMock(),
        )
        create_calls = [c for c in calls if "create" in c]
        self.assertEqual(len(create_calls), 2)
        titles = {c[c.index("--title") + 1] for c in create_calls}
        self.assertEqual(titles, {"[Alice] Send Q2 memo", "Follow up on Series A"})
        priorities = {c[c.index("--priority") + 1] for c in create_calls}
        self.assertEqual(priorities, {"High", "Medium"})


    def test_partial_create_failure_posts_partial_and_marks_done(self):
        import orchestrate
        row = {
            "recording_id": 71,
            "title": "1:1",
            "attendees": "[]",
            "summary": "1:1 summary",
        }
        create_count = 0
        def runner(argv, **_):
            nonlocal create_count
            if "create" in argv:
                create_count += 1
                if create_count == 1:
                    return MagicMock(returncode=0, stdout="Created: x", stderr="")
                return MagicMock(returncode=1, stdout="", stderr="Notion 500")
            return MagicMock(returncode=0, stdout="", stderr="")
        llm = MagicMock(return_value=(
            "TASK: First task | PRIORITY: High\n"
            "TASK: Second task | PRIORITY: Medium"
        ))
        posted: list[str] = []
        d1 = MagicMock()
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=llm,
            discord_poster=lambda c: posted.append(c),
            d1_client=d1,
        )
        self.assertEqual(len(posted), 1)
        self.assertIn("First task", posted[0])
        self.assertIn("WARNING", posted[0])
        d1.mark_done.assert_called_once_with(71)


class TestCrmTalkedTo(unittest.TestCase):
    def test_talked_to_only_for_attendees_found_in_crm(self):
        import orchestrate
        row = {
            "recording_id": 80,
            "title": "Quarterly",
            "attendees": (
                '[{"name": "Alice", "email": "alice@ext.com", "is_external": true},'
                ' {"name": "Bob", "email": "bob@ext.com", "is_external": true}]'
            ),
            "summary": "quarterly review",
        }
        calls: list[list[str]] = []
        def runner(argv, **_):
            calls.append(argv)
            if "summarize" in argv and "Alice" in argv:
                return MagicMock(returncode=0, stdout="Alice context", stderr="")
            if "summarize" in argv and "Bob" in argv:
                return MagicMock(returncode=1, stdout="", stderr="not in CRM")
            return MagicMock(returncode=0, stdout="", stderr="")
        captured_post = {}
        def poster(content):
            captured_post["c"] = content
        orchestrate.process_meeting(
            row,
            runner=runner,
            llm_caller=MagicMock(return_value="no action items"),
            discord_poster=poster,
            d1_client=MagicMock(),
        )
        talked_to = [c for c in calls if "talked-to" in c]
        self.assertEqual(len(talked_to), 1)
        self.assertIn("Alice", talked_to[0])
        self.assertIn("CRM updated: Alice", captured_post["c"])


class TestMainMultipleMeetings(unittest.TestCase):
    def test_each_pending_row_is_processed_independently(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = [
            {"recording_id": 101, "title": "Sync A", "attendees": "[]", "summary": "a"},
            {"recording_id": 102, "title": "Sync B", "attendees": "[]", "summary": "b"},
        ]
        posted: list[str] = []
        def poster(c):
            posted.append(c)
        orchestrate.main(
            argv=[],
            d1_client=d1,
            runner=lambda argv, **_: MagicMock(returncode=0, stdout="", stderr=""),
            llm_caller=MagicMock(return_value="no action items"),
            discord_poster=poster,
        )
        marked = [call.args[0] for call in d1.mark_done.call_args_list]
        self.assertEqual(marked, [101, 102])
        self.assertEqual(len(posted), 2)
        self.assertIn("Sync A", posted[0])
        self.assertIn("Sync B", posted[1])


class TestMainErrorHandling(unittest.TestCase):
    def test_llm_exception_surfaces_to_discord_and_returns_nonzero(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = [
            {"recording_id": 201, "title": "Flaky", "attendees": "[]", "summary": "x"},
        ]
        def bad_llm(_prompt):
            raise RuntimeError("OpenRouter 500")
        posted: list[str] = []
        def poster(c):
            posted.append(c)
        rc = orchestrate.main(
            argv=[],
            d1_client=d1,
            runner=lambda argv, **_: MagicMock(returncode=0, stdout="", stderr=""),
            llm_caller=bad_llm,
            discord_poster=poster,
        )
        self.assertEqual(rc, 1)
        self.assertEqual(d1.mark_done.call_count, 0)
        self.assertTrue(any("Flaky" in p and "OpenRouter 500" in p for p in posted))


if __name__ == "__main__":
    unittest.main()
