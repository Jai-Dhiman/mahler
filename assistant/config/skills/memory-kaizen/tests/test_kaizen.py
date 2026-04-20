# config/skills/memory-kaizen/tests/test_kaizen.py
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "OPENROUTER_API_KEY": "orkey",
    "HONCHO_API_KEY": "hkey",
}


def _make_conclusion(content: str):
    c = MagicMock()
    c.content = content
    return c


_SEVEN_CONCLUSIONS = [_make_conclusion(f"Jai fact {i}") for i in range(7)]


class TestKaizenRun(unittest.TestCase):

    def test_run_writes_each_pattern_as_separate_conclusion(self):
        llm_response = (
            "PATTERN: Jai consistently ships on Sundays.\n"
            "PATTERN: Jai finds auth-related issues recurring across projects."
        )
        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
            patch("kaizen._call_llm", return_value=llm_response),
            patch("sys.stdout", captured),
        ):
            mock_honcho.list_conclusions.return_value = _SEVEN_CONCLUSIONS
            import kaizen
            kaizen.main(["--run"])

        self.assertEqual(mock_honcho.conclude.call_count, 2)
        mock_honcho.conclude.assert_any_call(
            "Jai consistently ships on Sundays.", session_id="memory-kaizen"
        )
        mock_honcho.conclude.assert_any_call(
            "Jai finds auth-related issues recurring across projects.", session_id="memory-kaizen"
        )
        self.assertIn("2 patterns", captured.getvalue())

    def test_run_skips_conclude_and_reports_insufficient_when_fewer_than_5(self):
        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
            patch("sys.stdout", captured),
        ):
            mock_honcho.list_conclusions.return_value = [_make_conclusion("only one")]
            import kaizen
            kaizen.main(["--run"])

        self.assertIn("Insufficient data", captured.getvalue())
        mock_honcho.conclude.assert_not_called()

    def test_run_skips_conclude_when_llm_returns_no_patterns(self):
        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
            patch("kaizen._call_llm", return_value="NO_PATTERNS"),
            patch("sys.stdout", captured),
        ):
            mock_honcho.list_conclusions.return_value = _SEVEN_CONCLUSIONS
            import kaizen
            kaizen.main(["--run"])

        mock_honcho.conclude.assert_not_called()
        self.assertIn("no multi-entry patterns", captured.getvalue())

    def test_run_raises_when_list_conclusions_fails(self):
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
        ):
            mock_honcho.list_conclusions.side_effect = RuntimeError("Honcho list_conclusions failed: 503")
            import kaizen
            with self.assertRaises(RuntimeError) as ctx:
                kaizen.main(["--run"])

        self.assertIn("Honcho list_conclusions failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
