import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_msg(author_id: str, content: str, minutes_ago: int, now: datetime) -> dict:
    ts = now - timedelta(minutes=minutes_ago)
    return {
        "author": {"id": author_id},
        "content": content,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
    }


NOW = datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)
BOT_ID = "bot-123"
USER_ID = "user-456"


class TestSkipsNonFirstTurn(unittest.TestCase):
    def test_returns_none_when_not_first_turn(self):
        from plugin import conversation_history_context
        result = conversation_history_context("sess", "hello", is_first_turn=False, _now=NOW)
        self.assertIsNone(result)


class TestMissingEnv(unittest.TestCase):
    @patch("plugin._get_bot_id", return_value=BOT_ID)
    @patch("plugin._fetch_session_messages", return_value=[])
    @patch("plugin._load_hermes_env")
    def test_returns_none_when_token_missing(self, _load, _fetch, _bot):
        with patch.dict("os.environ", {"DISCORD_HOME_CHANNEL": "123"}, clear=False):
            import os
            os.environ.pop("DISCORD_BOT_TOKEN", None)
            from plugin import conversation_history_context
            result = conversation_history_context("sess", "hi", is_first_turn=True, _now=NOW)
            self.assertIsNone(result)

    @patch("plugin._get_bot_id", return_value=BOT_ID)
    @patch("plugin._fetch_session_messages", return_value=[])
    @patch("plugin._load_hermes_env")
    def test_returns_none_when_channel_missing(self, _load, _fetch, _bot):
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "tok"}, clear=False):
            import os
            os.environ.pop("DISCORD_HOME_CHANNEL", None)
            from plugin import conversation_history_context
            result = conversation_history_context("sess", "hi", is_first_turn=True, _now=NOW)
            self.assertIsNone(result)


class TestSilentFailure(unittest.TestCase):
    @patch("plugin._load_hermes_env")
    @patch("plugin._get_bot_id", side_effect=Exception("network error"))
    def test_returns_none_on_any_exception(self, _bot, _load):
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "tok", "DISCORD_HOME_CHANNEL": "123"}):
            from plugin import conversation_history_context
            result = conversation_history_context("sess", "hi", is_first_turn=True, _now=NOW)
            self.assertIsNone(result)


class TestNoRecentHistory(unittest.TestCase):
    @patch("plugin._load_hermes_env")
    @patch("plugin._get_bot_id", return_value=BOT_ID)
    @patch("plugin._fetch_session_messages", return_value=[])
    def test_returns_none_when_no_messages_in_window(self, _fetch, _bot, _load):
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "tok", "DISCORD_HOME_CHANNEL": "123"}):
            from plugin import conversation_history_context
            result = conversation_history_context("sess", "hi", is_first_turn=True, _now=NOW)
            self.assertIsNone(result)


class TestContextInjection(unittest.TestCase):
    def _run(self, messages, user_message="hi"):
        with patch("plugin._load_hermes_env"):
            with patch("plugin._get_bot_id", return_value=BOT_ID):
                with patch("plugin._fetch_session_messages", return_value=messages):
                    with patch.dict("os.environ", {
                        "DISCORD_BOT_TOKEN": "tok",
                        "DISCORD_HOME_CHANNEL": "123",
                    }):
                        from plugin import conversation_history_context
                        return conversation_history_context(
                            "sess", user_message, is_first_turn=True, _now=NOW
                        )

    def test_returns_context_with_prior_messages(self):
        msgs = [
            {"label": "You", "content": "what's on my calendar?", "ts": NOW - timedelta(minutes=30)},
            {"label": "Mahler", "content": "You have a standup at 3pm.", "ts": NOW - timedelta(minutes=29)},
        ]
        result = self._run(msgs, user_message="can you reschedule it?")
        assert result is not None
        self.assertIn("what's on my calendar?", result["context"])
        self.assertIn("standup at 3pm", result["context"])
        self.assertIn("You:", result["context"])
        self.assertIn("Mahler:", result["context"])

    def test_strips_current_user_message_from_tail(self):
        msgs = [
            {"label": "You", "content": "what's on my calendar?", "ts": NOW - timedelta(minutes=30)},
            {"label": "Mahler", "content": "Standup at 3pm.", "ts": NOW - timedelta(minutes=29)},
            {"label": "You", "content": "reschedule it", "ts": NOW - timedelta(minutes=1)},
        ]
        result = self._run(msgs, user_message="reschedule it")
        assert result is not None
        lines = result["context"].splitlines()
        user_lines = [l for l in lines if l.startswith("[") and "You: reschedule it" in l]
        self.assertEqual(len(user_lines), 0)

    def test_only_current_message_returns_none(self):
        msgs = [
            {"label": "You", "content": "hello mahler", "ts": NOW - timedelta(minutes=1)},
        ]
        result = self._run(msgs, user_message="hello mahler")
        self.assertIsNone(result)


class TestFetchSessionMessages(unittest.TestCase):
    def test_filters_messages_outside_window(self):
        raw = [
            _make_msg(USER_ID, "recent msg", 10, NOW),
            _make_msg(BOT_ID, "old msg", 60, NOW),  # outside 45-min window
        ]
        with patch("plugin._discord_get", return_value=raw):
            from plugin import _fetch_session_messages
            cutoff = NOW - timedelta(minutes=45)
            result = _fetch_session_messages("123", "tok", BOT_ID, cutoff)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], "recent msg")

    def test_labels_bot_messages_as_mahler(self):
        raw = [_make_msg(BOT_ID, "I found 3 tasks.", 5, NOW)]
        with patch("plugin._discord_get", return_value=raw):
            from plugin import _fetch_session_messages
            cutoff = NOW - timedelta(minutes=45)
            result = _fetch_session_messages("123", "tok", BOT_ID, cutoff)
            self.assertEqual(result[0]["label"], "Mahler")

    def test_labels_non_bot_messages_as_you(self):
        raw = [_make_msg(USER_ID, "show tasks", 5, NOW)]
        with patch("plugin._discord_get", return_value=raw):
            from plugin import _fetch_session_messages
            cutoff = NOW - timedelta(minutes=45)
            result = _fetch_session_messages("123", "tok", BOT_ID, cutoff)
            self.assertEqual(result[0]["label"], "You")

    def test_skips_empty_content(self):
        raw = [
            _make_msg(USER_ID, "", 5, NOW),
            _make_msg(USER_ID, "   ", 4, NOW),
            _make_msg(BOT_ID, "valid", 3, NOW),
        ]
        with patch("plugin._discord_get", return_value=raw):
            from plugin import _fetch_session_messages
            cutoff = NOW - timedelta(minutes=45)
            result = _fetch_session_messages("123", "tok", BOT_ID, cutoff)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], "valid")

    def test_raises_on_invalid_channel_id(self):
        from plugin import _fetch_session_messages
        with self.assertRaises(ValueError):
            _fetch_session_messages("not-a-snowflake", "tok", BOT_ID, NOW)


if __name__ == "__main__":
    unittest.main()
