import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct123",
    "CF_D1_DATABASE_ID": "db123",
    "CF_API_TOKEN": "cftoken",
    "OPENROUTER_API_KEY": "orkey",
}


def _patch_env(extra: dict | None = None):
    env = dict(_BASE_ENV)
    if extra:
        env.update(extra)
    return patch.dict("os.environ", env, clear=True)


class TestReflectRunWithProposals(unittest.TestCase):

    def test_outputs_json_proposals_for_detected_patterns(self):
        patterns = [
            {"from_addr": "news@acme.com", "classification": "NEEDS_ACTION", "occurrence_count": 5}
        ]
        llm_response = json.dumps([
            {
                "sender": "news@acme.com",
                "current_tier": "NEEDS_ACTION",
                "proposed_tier": "FYI",
                "evidence": "5 occurrences in 7 days with no follow-up",
            }
        ])
        mock_d1 = MagicMock()
        mock_d1.get_triage_patterns.return_value = patterns
        mock_d1.get_priority_map.return_value = "## URGENT\nDrop everything."

        captured = io.StringIO()
        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm", return_value=llm_response),
            patch("sys.stdout", captured),
        ):
            import reflect
            reflect.main(["--run"])

        proposals = json.loads(captured.getvalue())
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["sender"], "news@acme.com")
        self.assertEqual(proposals[0]["proposed_tier"], "FYI")
        self.assertIn("evidence", proposals[0])


class TestReflectRunNoProposals(unittest.TestCase):

    def test_prints_no_proposals_message_when_no_patterns(self):
        mock_d1 = MagicMock()
        mock_d1.get_triage_patterns.return_value = []

        captured = io.StringIO()
        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("sys.stdout", captured),
        ):
            import reflect
            reflect.main(["--run"])

        self.assertIn("No proposals this week", captured.getvalue())

    def test_does_not_call_llm_when_no_patterns(self):
        mock_d1 = MagicMock()
        mock_d1.get_triage_patterns.return_value = []

        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm") as mock_llm,
            patch("sys.stdout", io.StringIO()),
        ):
            import reflect
            reflect.main(["--run"])

        mock_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
