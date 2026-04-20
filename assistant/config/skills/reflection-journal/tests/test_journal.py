import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct123",
    "CF_D1_DATABASE_ID": "db123",
    "CF_API_TOKEN": "cftoken",
    "OPENROUTER_API_KEY": "orkey",
    "HONCHO_API_KEY": "hkey",
}


def _patch_env(extra: dict | None = None):
    env = dict(_BASE_ENV)
    if extra:
        env.update(extra)
    return patch.dict("os.environ", env, clear=True)


class TestJournalPrompt(unittest.TestCase):

    def test_prompt_prints_all_three_reflection_questions(self):
        captured = io.StringIO()
        with _patch_env(), patch("sys.stdout", captured):
            import journal
            journal.main(["--prompt"])

        output = captured.getvalue()
        self.assertIn("How did last week go overall", output)
        self.assertIn("drained", output)
        self.assertIn("avoiding", output)

    def test_prompt_does_not_require_env_vars(self):
        captured = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), patch("sys.stdout", captured):
            import journal
            journal.main(["--prompt"])

        self.assertIn("How did last week go overall", captured.getvalue())


class TestJournalRecord(unittest.TestCase):

    def test_record_stores_raw_text_in_d1_and_concludes_facts_to_honcho(self):
        mock_d1 = MagicMock()
        llm_response = (
            "FACT: Jai is energized by shipping features\n"
            "FACT: meetings consistently drain his energy"
        )

        captured = io.StringIO()
        with (
            _patch_env(),
            patch("journal.D1Client", return_value=mock_d1),
            patch("journal._call_llm", return_value=llm_response),
            patch("journal.honcho_client") as mock_honcho,
            patch("sys.stdout", captured),
        ):
            import journal
            journal.main(["--record", "Good week overall. Meetings drained me."])

        mock_d1.insert_reflection.assert_called_once()
        args = mock_d1.insert_reflection.call_args[0]
        self.assertRegex(args[0], r"^\d{4}-W\d{2}$")
        self.assertEqual(args[1], "Good week overall. Meetings drained me.")
        self.assertEqual(mock_honcho.conclude.call_count, 2)
        self.assertIn("Reflection recorded.", captured.getvalue())

    def test_record_raises_when_d1_fails_before_honcho_called(self):
        mock_d1 = MagicMock()
        mock_d1.insert_reflection.side_effect = RuntimeError("D1 write failed")

        with (
            _patch_env(),
            patch("journal.D1Client", return_value=mock_d1),
            patch("journal.honcho_client") as mock_honcho,
        ):
            import journal
            with self.assertRaises(RuntimeError) as ctx:
                journal.main(["--record", "Some reflection text"])

        self.assertIn("D1 write failed", str(ctx.exception))
        mock_honcho.conclude.assert_not_called()

    def test_record_raises_when_llm_fails_after_d1_write_committed(self):
        mock_d1 = MagicMock()

        with (
            _patch_env(),
            patch("journal.D1Client", return_value=mock_d1),
            patch("journal._call_llm", side_effect=RuntimeError("OpenRouter error: HTTP 429")),
            patch("journal.honcho_client") as mock_honcho,
        ):
            import journal
            with self.assertRaises(RuntimeError) as ctx:
                journal.main(["--record", "Some reflection text"])

        self.assertIn("OpenRouter error", str(ctx.exception))
        mock_d1.insert_reflection.assert_called_once()
        mock_honcho.conclude.assert_not_called()

    def test_record_zero_facts_when_llm_returns_no_patterns(self):
        mock_d1 = MagicMock()

        captured = io.StringIO()
        with (
            _patch_env(),
            patch("journal.D1Client", return_value=mock_d1),
            patch("journal._call_llm", return_value="NO_PATTERNS"),
            patch("journal.honcho_client") as mock_honcho,
            patch("sys.stdout", captured),
        ):
            import journal
            journal.main(["--record", "Pretty uneventful week."])

        mock_d1.insert_reflection.assert_called_once()
        mock_honcho.conclude.assert_not_called()
        self.assertIn("Reflection recorded.", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
