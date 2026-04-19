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


class TestReflectApply(unittest.TestCase):

    def _make_proposal(self, **overrides) -> str:
        base = {
            "sender": "news@acme.com",
            "current_tier": "NEEDS_ACTION",
            "proposed_tier": "FYI",
            "evidence": "5 occurrences in 7 days with no follow-up",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_writes_updated_map_to_d1(self):
        current_map = "## NEEDS_ACTION\n\n**Examples:**\n- Direct questions"
        updated_map = "## FYI\n\n**Examples:**\n- news@acme.com"

        mock_d1 = MagicMock()
        mock_d1.get_priority_map.return_value = current_map

        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm", return_value=updated_map),
        ):
            import reflect
            reflect.main(["--apply", self._make_proposal()])

        mock_d1.set_priority_map.assert_called_once_with(updated_map)

    def test_raises_on_invalid_proposal_json(self):
        with (
            _patch_env(),
            patch("reflect.D1Client"),
        ):
            import reflect
            with self.assertRaises(RuntimeError) as ctx:
                reflect.main(["--apply", "not-valid-json"])
        self.assertIn("Invalid proposal JSON", str(ctx.exception))

    def test_raises_on_proposal_missing_required_keys(self):
        incomplete = json.dumps({"sender": "news@acme.com"})
        with (
            _patch_env(),
            patch("reflect.D1Client"),
        ):
            import reflect
            with self.assertRaises(RuntimeError) as ctx:
                reflect.main(["--apply", incomplete])
        self.assertIn("missing required keys", str(ctx.exception))

    def test_raises_and_does_not_write_when_llm_call_fails(self):
        mock_d1 = MagicMock()
        mock_d1.get_priority_map.return_value = "## NEEDS_ACTION\n- Direct questions"

        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm", side_effect=RuntimeError("OpenRouter auth failure")),
        ):
            import reflect
            with self.assertRaises(RuntimeError) as ctx:
                reflect.main(["--apply", self._make_proposal()])

        self.assertIn("OpenRouter", str(ctx.exception))
        mock_d1.set_priority_map.assert_not_called()

    def test_raises_and_does_not_write_when_llm_returns_implausible_map(self):
        mock_d1 = MagicMock()
        mock_d1.get_priority_map.return_value = "## NEEDS_ACTION\n- Direct questions"

        with (
            _patch_env(),
            patch("reflect.D1Client", return_value=mock_d1),
            patch("reflect._call_llm", return_value="Sorry, I cannot help with that."),
        ):
            import reflect
            with self.assertRaises(RuntimeError) as ctx:
                reflect.main(["--apply", self._make_proposal()])

        self.assertIn("implausible priority map", str(ctx.exception))
        mock_d1.set_priority_map.assert_not_called()


if __name__ == "__main__":
    unittest.main()
