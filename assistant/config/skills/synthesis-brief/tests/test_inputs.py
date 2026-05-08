import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

import inputs


class TestLoadAllEmpty(unittest.TestCase):
    def test_returns_empty_bundle_and_creates_tables(self):
        d1 = MagicMock()
        d1.query.return_value = []
        honcho = MagicMock()
        honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        self.assertEqual(bundle.recent_items, [])
        self.assertEqual(bundle.context_items, [])
        self.assertEqual(bundle.past_briefs, [])
        self.assertEqual(bundle.identifiers, set())

        executed_sqls = [call.args[0] for call in d1.query.call_args_list]
        joined = "\n".join(executed_sqls)
        self.assertIn("CREATE TABLE IF NOT EXISTS local_capture", joined)
        self.assertIn("CREATE TABLE IF NOT EXISTS synthesis_brief", joined)


class TestLoadAllProjectWins(unittest.TestCase):
    def test_includes_project_log_wins_in_context_items(self):
        win_rows = [
            {"id": 42, "project": "crescendAI", "summary": "Shipped V6 atoms",
             "created_at": "2026-05-04 12:00:00"},
            {"id": 43, "project": "mahler", "summary": "Added wiki search",
             "created_at": "2026-05-05 09:00:00"},
        ]

        d1 = MagicMock()
        def query_side_effect(sql, params=None):
            if "FROM project_log" in sql:
                return win_rows
            return []
        d1.query.side_effect = query_side_effect

        honcho = MagicMock()
        honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        wins = [it for it in bundle.context_items if it.source == "project_log"]
        self.assertEqual(len(wins), 2)
        self.assertEqual(wins[0].id, "project_log:42")
        self.assertIn("Shipped V6 atoms", wins[0].content)


class TestLoadAllHoncho(unittest.TestCase):
    def test_includes_honcho_conclusions_in_context_items(self):
        d1 = MagicMock()
        d1.query.return_value = []

        c1 = MagicMock(); c1.content = "Jai is focused on traderjoe"; c1.created_at = "2026-05-01T00:00:00Z"
        c2 = MagicMock(); c2.content = "Jai ships on Sundays"; c2.created_at = "2026-05-03T00:00:00Z"

        honcho = MagicMock()
        honcho.list_conclusions.return_value = [c1, c2]

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        honcho_items = [it for it in bundle.context_items if it.source == "honcho"]
        self.assertEqual(len(honcho_items), 2)
        self.assertEqual(honcho_items[0].content, "Jai is focused on traderjoe")
        honcho.list_conclusions.assert_called_once_with(since_days=14)


class TestLoadAllLocalCapturePartition(unittest.TestCase):
    def test_recent_rows_go_to_recent_older_to_context(self):
        recent_rows = [
            {"id": 1, "source": "memory", "project": "MEMORY.md",
             "content": "Recent memory", "captured_at": "2026-05-07 04:00:00"},
        ]
        context_rows = [
            {"id": 2, "source": "git", "project": "crescendAI",
             "content": "[crescendAI] abc1 Old commit", "captured_at": "2026-05-02 10:00:00"},
        ]

        def query_side_effect(sql, params=None):
            if "FROM local_capture" in sql and "captured_at >= datetime" in sql:
                if params and params[0] == "-1" and "captured_at < datetime" not in sql:
                    return recent_rows
                if params and params[0] == "-14" and "captured_at < datetime" in sql:
                    return context_rows
            return []

        d1 = MagicMock()
        d1.query.side_effect = query_side_effect
        honcho = MagicMock(); honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        recent_local = [it for it in bundle.recent_items if it.source in ("memory", "git")]
        context_local = [it for it in bundle.context_items if it.source in ("memory", "git")]
        self.assertEqual(len(recent_local), 1)
        self.assertEqual(recent_local[0].source, "memory")
        self.assertEqual(recent_local[0].id, "memory:1")
        self.assertEqual(len(context_local), 1)
        self.assertEqual(context_local[0].source, "git")
        self.assertEqual(context_local[0].id, "git:2")


import json as _json


class TestLoadAllPastBriefs(unittest.TestCase):
    def test_returns_past_briefs_with_parsed_connections(self):
        past_rows = [
            {"posted_at": "2026-05-06 13:00:00",
             "connections_json": _json.dumps([{"summary": "A", "citations": []}]),
             "pattern": "P1", "question": "Q1"},
            {"posted_at": "2026-05-05 13:00:00",
             "connections_json": _json.dumps([{"summary": "B", "citations": []}]),
             "pattern": "P2", "question": "Q2"},
        ]

        def query_side_effect(sql, params=None):
            if "FROM synthesis_brief" in sql:
                return past_rows
            return []

        d1 = MagicMock(); d1.query.side_effect = query_side_effect
        honcho = MagicMock(); honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        self.assertEqual(len(bundle.past_briefs), 2)
        self.assertEqual(bundle.past_briefs[0]["pattern"], "P1")
        self.assertEqual(bundle.past_briefs[0]["connections"][0]["summary"], "A")


if __name__ == "__main__":
    unittest.main()
