# config/skills/project-synthesis/tests/test_synthesize.py
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct",
    "CF_D1_DATABASE_ID": "db",
    "CF_API_TOKEN": "cftoken",
    "OPENROUTER_API_KEY": "orkey",
    "HONCHO_API_KEY": "hkey",
}

_SAMPLE_ROWS = [
    {
        "project": "mahler",
        "entry_type": "win",
        "summary": "Shipped morning brief news extension",
        "git_ref": "abc123",
        "created_at": "2026-04-18",
    },
    {
        "project": "traderjoe",
        "entry_type": "blocker",
        "summary": "Spread calculation off by factor of 2",
        "git_ref": "def456",
        "created_at": "2026-04-19",
    },
]


class TestSynthesizeRun(unittest.TestCase):

    def test_run_concludes_llm_output_once_when_entries_exist(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.return_value = _SAMPLE_ROWS
        synthesis = "Jai split attention between mahler and traderjoe this week, shipping a news feature while hitting a spread calculation blocker."

        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
            patch("synthesize._call_llm", return_value=synthesis),
            patch("synthesize.honcho_client") as mock_honcho,
            patch("sys.stdout", io.StringIO()),
        ):
            import synthesize
            synthesize.main(["--run"])

        mock_honcho.conclude.assert_called_once_with(synthesis, session_id="project-synthesis")

    def test_run_skips_conclude_and_prints_no_activity_when_d1_empty(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.return_value = []

        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
            patch("synthesize.honcho_client") as mock_honcho,
            patch("sys.stdout", captured),
        ):
            import synthesize
            synthesize.main(["--run"])

        self.assertIn("No project activity", captured.getvalue())
        mock_honcho.conclude.assert_not_called()

    def test_run_raises_when_d1_query_fails(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.side_effect = RuntimeError("D1 query failed: 500")

        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
        ):
            import synthesize
            with self.assertRaises(RuntimeError) as ctx:
                synthesize.main(["--run"])

        self.assertIn("D1 query failed", str(ctx.exception))

    def test_run_raises_when_llm_fails_and_does_not_conclude(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.return_value = _SAMPLE_ROWS

        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
            patch("synthesize._call_llm", side_effect=RuntimeError("OpenRouter error: HTTP 429")),
            patch("synthesize.honcho_client") as mock_honcho,
        ):
            import synthesize
            with self.assertRaises(RuntimeError) as ctx:
                synthesize.main(["--run"])

        self.assertIn("OpenRouter error", str(ctx.exception))
        mock_honcho.conclude.assert_not_called()


if __name__ == "__main__":
    unittest.main()
