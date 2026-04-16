import pathlib
import sys
import os
import io
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))


class TestDedupCheck(unittest.TestCase):

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_check_does_not_exit_1_when_event_not_seen(self, MockClient):
        MockClient.return_value.is_already_notified.return_value = False
        # Should complete without raising SystemExit
        from dedup import main
        try:
            main(["check", "--event-id", "evt-new"])
        except SystemExit as e:
            self.fail(f"Expected clean exit, got SystemExit({e.code})")

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_check_exits_1_when_event_already_notified(self, MockClient):
        MockClient.return_value.is_already_notified.return_value = True
        from dedup import main
        with self.assertRaises(SystemExit) as ctx:
            main(["check", "--event-id", "evt-old"])
        self.assertEqual(ctx.exception.code, 1)

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_check_raises_runtime_error_on_d1_failure(self, MockClient):
        MockClient.return_value.is_already_notified.side_effect = RuntimeError("D1 unreachable")
        from dedup import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["check", "--event-id", "evt-x"])
        self.assertIn("D1 unreachable", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_check_raises_when_cf_account_id_missing(self):
        from dedup import main
        with self.assertRaises(RuntimeError) as ctx:
            main(["check", "--event-id", "evt-x"])
        self.assertIn("CF_ACCOUNT_ID", str(ctx.exception))


class TestDedupLog(unittest.TestCase):

    @patch.dict(os.environ, {
        "CF_ACCOUNT_ID": "acct-abc",
        "CF_D1_DATABASE_ID": "db-123",
        "CF_API_TOKEN": "tok-xyz",
    })
    @patch("dedup.D1Client")
    def test_log_calls_insert_with_correct_args(self, MockClient):
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            from dedup import main
            main(["log", "--event-id", "evt123",
                  "--summary", "Team standup",
                  "--start-time", "2026-04-16T15:00:00Z"])
        MockClient.return_value.insert_meeting_prep.assert_called_once_with(
            "evt123", "Team standup", "2026-04-16T15:00:00Z"
        )
        self.assertIn("evt123", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
