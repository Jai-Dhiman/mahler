import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import validator
from inputs import InputBundle, Item


class TestThinContext(unittest.TestCase):
    def test_returns_false_thin_context_when_recent_under_3_and_context_under_5(self):
        bundle = InputBundle(
            recent_items=[Item("memory", "memory:1", "x", "2026-05-07")],
            context_items=[Item("project_log", "project_log:1", "y", "2026-05-06")],
            past_briefs=[],
            identifiers={"memory:1", "project_log:1"},
        )
        ok, reason = validator.validate({}, bundle)
        self.assertFalse(ok)
        self.assertEqual(reason, "thin_context")


class TestCitations(unittest.TestCase):
    def _bundle(self):
        return InputBundle(
            recent_items=[Item("memory", f"memory:{i}", "x", "2026-05-07") for i in range(4)],
            context_items=[Item("project_log", f"project_log:{i}", "y", "2026-05-04") for i in range(6)],
            past_briefs=[],
            identifiers={f"memory:{i}" for i in range(4)} | {f"project_log:{i}" for i in range(6)},
        )

    def test_returns_false_insufficient_citations_when_under_2_of_3_qualify(self):
        bundle = self._bundle()
        brief = {
            "connections": [
                {"summary": "A", "citations": [{"source": "memory", "id": "memory:1"}]},  # only 1
                {"summary": "B", "citations": []},                                          # 0
                {"summary": "C", "citations": [
                    {"source": "memory", "id": "memory:2"},
                    {"source": "project_log", "id": "project_log:0"},
                ]},                                                                         # 2 ok
            ],
            "pattern": "p",
            "question": "q",
        }
        ok, reason = validator.validate(brief, bundle)
        self.assertFalse(ok)
        self.assertEqual(reason, "insufficient_citations")


if __name__ == "__main__":
    unittest.main()
