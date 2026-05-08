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


class TestSynthesizeThinContext(unittest.TestCase):
    def test_skips_llm_when_bundle_thin(self):
        thin_bundle = InputBundle(
            recent_items=[Item("memory", "memory:0", "x", "2026-05-07")],
            context_items=[],
            past_briefs=[],
            identifiers={"memory:0"},
        )
        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
        }
        captured = io.StringIO()
        llm_mock = MagicMock()
        with patch.dict("os.environ", env, clear=True), \
             patch("synthesize._build_d1", return_value=MagicMock()), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=thin_bundle), \
             patch("synthesize._call_llm", llm_mock), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run"])

        self.assertIn("thin_context", captured.getvalue())
        llm_mock.assert_not_called()


class TestSynthesizePersistence(unittest.TestCase):
    def test_writes_synthesis_brief_row_and_mahler_kv_on_success(self):
        d1 = MagicMock()
        d1.query.return_value = []
        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
        }
        with patch.dict("os.environ", env, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", return_value=json.dumps(_GOOD_BRIEF)):
            synthesize.main_with_args(["--run"])

        sb_inserts = [c for c in d1.query.call_args_list if "INSERT INTO synthesis_brief" in c.args[0]]
        kv_writes = [c for c in d1.query.call_args_list if "mahler_kv" in c.args[0] and "INSERT" in c.args[0]]
        self.assertEqual(len(sb_inserts), 1)
        self.assertEqual(len(kv_writes), 1)

        kv_params = kv_writes[0].args[1]
        self.assertEqual(kv_params[0], "synthesis_brief:latest")
        payload = json.loads(kv_params[1])
        self.assertEqual(payload["pattern"], "Pattern X")
        self.assertIn("posted_at", payload)


_ENV = {
    "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
    "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
}


class TestSynthesizeLLMError(unittest.TestCase):
    def test_skips_gracefully_when_llm_raises(self):
        d1 = MagicMock()
        d1.query.return_value = []
        captured = io.StringIO()
        with patch.dict("os.environ", _ENV, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", side_effect=RuntimeError("HTTP 500")), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run"])

        self.assertIn("skipped", captured.getvalue())
        self.assertIn("llm_error", captured.getvalue())
        inserts = [c for c in d1.query.call_args_list if "INSERT INTO synthesis_brief" in c.args[0]]
        self.assertEqual(inserts, [])


class TestSynthesizeMalformedJSON(unittest.TestCase):
    def test_skips_when_llm_returns_invalid_json(self):
        d1 = MagicMock()
        d1.query.return_value = []
        captured = io.StringIO()
        with patch.dict("os.environ", _ENV, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", return_value="not valid json"), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run"])

        self.assertIn("skipped: malformed", captured.getvalue())
        inserts = [c for c in d1.query.call_args_list if "INSERT INTO synthesis_brief" in c.args[0]]
        self.assertEqual(inserts, [])


class TestSynthesizeValidatorRejects(unittest.TestCase):
    def test_skips_when_validator_rejects_insufficient_citations(self):
        bad_brief = {
            "connections": [
                {"summary": "A", "citations": [{"source": "memory", "id": "memory:0"}]},
                {"summary": "B", "citations": []},
                {"summary": "C", "citations": []},
            ],
            "pattern": "P",
            "question": "Q",
        }
        d1 = MagicMock()
        d1.query.return_value = []
        captured = io.StringIO()
        with patch.dict("os.environ", _ENV, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", return_value=json.dumps(bad_brief)), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run"])

        self.assertIn("skipped: insufficient_citations", captured.getvalue())
        inserts = [c for c in d1.query.call_args_list if "INSERT INTO synthesis_brief" in c.args[0]]
        self.assertEqual(inserts, [])


if __name__ == "__main__":
    unittest.main()
