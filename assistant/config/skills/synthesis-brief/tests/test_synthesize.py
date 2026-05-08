import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

import synthesize
from inputs import InputBundle, Item


def _full_bundle():
    return InputBundle(
        recent_items=[Item("memory", f"memory:{i}", "x", "2026-05-07") for i in range(4)],
        context_items=[Item("project_log", f"project_log:{i}", "y", "2026-05-04") for i in range(6)],
        past_briefs=[],
        identifiers={f"memory:{i}" for i in range(4)} | {f"project_log:{i}" for i in range(6)},
    )


_GOOD_BRIEF = {
    "connections": [
        {"summary": "A", "citations": [
            {"source": "memory", "id": "memory:0"},
            {"source": "memory", "id": "memory:1"},
        ]},
        {"summary": "B", "citations": [
            {"source": "memory", "id": "memory:2"},
            {"source": "project_log", "id": "project_log:0"},
        ]},
        {"summary": "C", "citations": [
            {"source": "memory", "id": "memory:3"},
            {"source": "project_log", "id": "project_log:1"},
        ]},
    ],
    "pattern": "Pattern X",
    "question": "Question Y",
}


class TestSynthesizeDryRun(unittest.TestCase):
    def test_dry_run_prints_brief_and_does_not_write(self):
        d1 = MagicMock()
        d1.query.return_value = []
        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
        }
        captured = io.StringIO()
        with patch.dict("os.environ", env, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", return_value=json.dumps(_GOOD_BRIEF)), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run", "--dry-run"])

        out = captured.getvalue()
        self.assertIn("Pattern X", out)
        self.assertIn("Question Y", out)

        # No INSERT INTO synthesis_brief and no mahler_kv write
        write_calls = [
            c for c in d1.query.call_args_list
            if "INSERT INTO synthesis_brief" in c.args[0]
            or "mahler_kv" in c.args[0] and "INSERT" in c.args[0]
        ]
        self.assertEqual(write_calls, [])


if __name__ == "__main__":
    unittest.main()
