import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
