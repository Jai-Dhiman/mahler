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


if __name__ == "__main__":
    unittest.main()
