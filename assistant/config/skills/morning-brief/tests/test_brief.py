import json
import sys
import unittest
import unittest.mock
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Make brief.py importable from sibling scripts/ dir
_TESTS_DIR = Path(__file__).parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# Patch the email-triage import before importing brief
import types
_fake_d1 = types.ModuleType("d1_client")


class _FakeD1Client:
    def __init__(self, *args, **kwargs):
        pass

    def query(self, sql, params=None):
        return []


_fake_d1.D1Client = _FakeD1Client
sys.modules["d1_client"] = _fake_d1

from brief import (
    compute_cutoff,
    build_embed,
    post_brief,
    load_env,
)


class TestComputeCutoff(unittest.TestCase):
    def test_returns_iso_string(self):
        result = compute_cutoff(12)
        # Should parse without error
        dt = datetime.fromisoformat(result.replace("Z", "+00:00"))
        self.assertIsInstance(dt, datetime)

    def test_approximately_n_hours_ago(self):
        now = datetime.now(timezone.utc)
        result = compute_cutoff(12)
        dt = datetime.fromisoformat(result.replace("Z", "+00:00"))
        expected = now - timedelta(hours=12)
        diff = abs((dt - expected).total_seconds())
        self.assertLess(diff, 5)  # within 5 seconds

    def test_since_hours_respected(self):
        result_12 = compute_cutoff(12)
        result_24 = compute_cutoff(24)
        dt_12 = datetime.fromisoformat(result_12.replace("Z", "+00:00"))
        dt_24 = datetime.fromisoformat(result_24.replace("Z", "+00:00"))
        self.assertGreater(dt_12, dt_24)


class TestBuildEmbed(unittest.TestCase):
    def _make_row(self, classification, from_addr="sender@example.com", subject="Test Subject", summary="A test summary"):
        return {
            "classification": classification,
            "from_addr": from_addr,
            "subject": subject,
            "summary": summary,
        }

    def test_morning_title_and_color(self):
        rows = [self._make_row("NEEDS_ACTION")]
        payload = build_embed(rows, "morning", 12)
        embed = payload["embeds"][0]
        self.assertIn("Morning Brief", embed["title"])
        self.assertEqual(embed["color"], 3447003)

    def test_evening_title_and_color(self):
        rows = [self._make_row("FYI")]
        payload = build_embed(rows, "evening", 12)
        embed = payload["embeds"][0]
        self.assertIn("Evening Brief", embed["title"])
        self.assertEqual(embed["color"], 10181046)

    def test_title_contains_date(self):
        rows = []
        payload = build_embed(rows, "morning", 12)
        embed = payload["embeds"][0]
        # Should have something like "Apr 7" in title
        self.assertRegex(embed["title"], r"\w+ \d+")

    def test_needs_action_field_present_with_correct_count(self):
        rows = [
            self._make_row("NEEDS_ACTION", "a@x.com", "Subject A", "Summary A"),
            self._make_row("NEEDS_ACTION", "b@x.com", "Subject B", "Summary B"),
        ]
        payload = build_embed(rows, "morning", 12)
        fields = payload["embeds"][0]["fields"]
        na_fields = [f for f in fields if "Needs Action" in f["name"]]
        self.assertEqual(len(na_fields), 1)
        self.assertIn("2", na_fields[0]["name"])

    def test_fyi_field_present(self):
        rows = [self._make_row("FYI", subject="FYI Subject")]
        payload = build_embed(rows, "morning", 12)
        fields = payload["embeds"][0]["fields"]
        fyi_fields = [f for f in fields if "FYI" in f["name"]]
        self.assertEqual(len(fyi_fields), 1)
        self.assertIn("FYI Subject", fyi_fields[0]["value"])

    def test_noise_field_shows_count(self):
        rows = [
            self._make_row("NOISE"),
            self._make_row("NOISE"),
            self._make_row("NOISE"),
        ]
        payload = build_embed(rows, "morning", 12)
        fields = payload["embeds"][0]["fields"]
        noise_fields = [f for f in fields if "Noise" in f["name"]]
        self.assertEqual(len(noise_fields), 1)
        self.assertIn("3", noise_fields[0]["value"])

    def test_empty_rows_description_and_no_na_fyi_fields(self):
        payload = build_embed([], "morning", 12)
        embed = payload["embeds"][0]
        self.assertEqual(embed.get("description"), "Nothing needs your attention.")
        fields = embed.get("fields", [])
        na_fields = [f for f in fields if "Needs Action" in f["name"]]
        fyi_fields = [f for f in fields if "FYI" in f["name"]]
        self.assertEqual(len(na_fields), 0)
        self.assertEqual(len(fyi_fields), 0)

    def test_empty_rows_noise_still_present(self):
        payload = build_embed([], "morning", 12)
        fields = payload["embeds"][0].get("fields", [])
        noise_fields = [f for f in fields if "Noise" in f["name"]]
        self.assertEqual(len(noise_fields), 1)
        self.assertIn("0", noise_fields[0]["value"])

    def test_no_needs_action_no_fyi_but_noise_shows_description(self):
        rows = [self._make_row("NOISE")]
        payload = build_embed(rows, "morning", 12)
        embed = payload["embeds"][0]
        self.assertEqual(embed.get("description"), "Nothing needs your attention.")

    def test_urgent_rows_excluded(self):
        rows = [
            self._make_row("URGENT", "urgent@x.com", "Urgent Subject", "Urgent summary"),
            self._make_row("NEEDS_ACTION", "na@x.com", "NA Subject", "NA summary"),
        ]
        payload = build_embed(rows, "morning", 12)
        fields = payload["embeds"][0]["fields"]
        na_fields = [f for f in fields if "Needs Action" in f["name"]]
        # Only 1 NEEDS_ACTION, not 2
        self.assertIn("1", na_fields[0]["name"])
        # URGENT subject must not appear in any field
        all_values = " ".join(f["value"] for f in fields)
        self.assertNotIn("Urgent Subject", all_values)

    def test_needs_action_truncation(self):
        # Create rows that will exceed 1024 chars
        rows = [
            self._make_row("NEEDS_ACTION", f"sender{i}@example.com", f"Subject {i}", "A" * 100)
            for i in range(20)
        ]
        payload = build_embed(rows, "morning", 12)
        fields = payload["embeds"][0]["fields"]
        na_fields = [f for f in fields if "Needs Action" in f["name"]]
        value = na_fields[0]["value"]
        self.assertLessEqual(len(value), 1024)
        self.assertIn("... +", value)

    def test_footer_contains_since_hours(self):
        payload = build_embed([], "morning", 6)
        footer = payload["embeds"][0]["footer"]["text"]
        self.assertIn("6h", footer)
        self.assertIn("Mahler", footer)

    def test_needs_action_format(self):
        rows = [self._make_row("NEEDS_ACTION", "alice@example.com", "My Subject", "My summary")]
        payload = build_embed(rows, "morning", 12)
        fields = payload["embeds"][0]["fields"]
        na_fields = [f for f in fields if "Needs Action" in f["name"]]
        value = na_fields[0]["value"]
        self.assertIn("alice@example.com", value)
        self.assertIn("My Subject", value)
        self.assertIn("My summary", value)

    def test_multi_source_item_shows_source_count_suffix(self):
        news_items = [
            {
                "title": "Big market crash affects global economies",
                "url": "https://example.com/crash",
                "category": "Macro/Markets",
                "source_count": 3,
            }
        ]
        payload = build_embed([], "morning", 12, news_items=news_items)
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertIn("· 3 sources", news_fields[0]["value"])

    def test_no_news_field_when_news_items_is_none(self):
        payload = build_embed([], "morning", 12, news_items=None)
        fields = payload["embeds"][0].get("fields", [])
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 0)

    def test_news_items_appends_worth_reading_field(self):
        news_items = [
            {
                "title": "OpenAI releases GPT-5 model",
                "url": "https://example.com/gpt5",
                "category": "AI/Tech",
                "source_count": 1,
            }
        ]
        payload = build_embed([], "morning", 12, news_items=news_items)
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 1)
        self.assertIn(
            "[OpenAI releases GPT-5 model](https://example.com/gpt5)",
            news_fields[0]["value"],
        )


class TestPostBrief(unittest.TestCase):
    def test_raises_on_non_https_url(self):
        from brief import post_brief
        with self.assertRaises(RuntimeError):
            post_brief("http://example.com/webhook", {"embeds": []})

    def test_raises_on_non_https_no_scheme(self):
        from brief import post_brief
        with self.assertRaises(RuntimeError):
            post_brief("ftp://example.com/webhook", {"embeds": []})

    def test_raises_on_non_200_response(self):
        from brief import post_brief
        import urllib.error

        mock_response = unittest.mock.MagicMock()
        mock_response.status = 500
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = unittest.mock.MagicMock(return_value=False)

        http_error = urllib.error.HTTPError(
            url="https://discord.com/api/webhooks/test",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=unittest.mock.MagicMock(read=lambda: b"error"),
        )

        with unittest.mock.patch("urllib.request.OpenerDirector.open", side_effect=http_error):
            with self.assertRaises(RuntimeError):
                post_brief("https://discord.com/api/webhooks/test", {"embeds": []})

    def test_succeeds_on_204(self):
        from brief import post_brief

        mock_response = unittest.mock.MagicMock()
        mock_response.status = 204
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = unittest.mock.MagicMock(return_value=False)

        with unittest.mock.patch("urllib.request.OpenerDirector.open", return_value=mock_response):
            # Should not raise
            post_brief("https://discord.com/api/webhooks/test", {"embeds": []})


class TestLoadEnv(unittest.TestCase):
    def test_raises_on_missing_var(self):
        from brief import load_env
        # Patch environ with all vars missing
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                load_env(dry_run=False)

    def test_raises_on_missing_webhook_when_not_dry_run(self):
        from brief import load_env
        env = {
            "CF_ACCOUNT_ID": "acc123",
            "CF_D1_DATABASE_ID": "db-abc123",
            "CF_API_TOKEN": "tok_abc",
        }
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            with self.assertRaises(RuntimeError):
                load_env(dry_run=False)

    def test_no_webhook_required_in_dry_run(self):
        from brief import load_env
        env = {
            "CF_ACCOUNT_ID": "acc123",
            "CF_D1_DATABASE_ID": "db-abc123",
            "CF_API_TOKEN": "tok_abc",
        }
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            result = load_env(dry_run=True)
            self.assertEqual(result["CF_ACCOUNT_ID"], "acc123")
            self.assertNotIn("DISCORD_TRIAGE_WEBHOOK", result)

    def test_returns_all_vars_when_present(self):
        from brief import load_env
        env = {
            "CF_ACCOUNT_ID": "acc123",
            "CF_D1_DATABASE_ID": "db-abc123",
            "CF_API_TOKEN": "tok_abc",
            "DISCORD_TRIAGE_WEBHOOK": "https://discord.com/api/webhooks/test",
        }
        with unittest.mock.patch.dict("os.environ", env, clear=True):
            result = load_env(dry_run=False)
            self.assertEqual(result["CF_ACCOUNT_ID"], "acc123")
            self.assertEqual(result["DISCORD_TRIAGE_WEBHOOK"], "https://discord.com/api/webhooks/test")


class TestMainNewsWiring(unittest.TestCase):
    def _run_dry_run(self, env, patches):
        """Run main() with --dry-run and return the parsed JSON payload."""
        import json
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["brief.py", "--period", "morning", "--dry-run"]
        sys.stdout = StringIO()
        try:
            ctx = unittest.mock.patch.dict("os.environ", env, clear=True)
            ctx.__enter__()
            active = [p.__enter__() for p in patches]
            from brief import main
            main()
            output = sys.stdout.getvalue()
            for p in reversed(patches):
                p.__exit__(None, None, None)
            ctx.__exit__(None, None, None)
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
        return json.loads(output)

    def test_dry_run_output_includes_news_field(self):
        fixture_items = [
            {
                "title": "AI breakthrough in language model research",
                "url": "https://example.com/ai",
                "category": "AI/Tech",
                "source_count": 1,
            }
        ]
        env = {
            "CF_ACCOUNT_ID": "acc123",
            "CF_D1_DATABASE_ID": "db-abc",
            "CF_API_TOKEN": "tok_abc",
        }
        patches = [
            unittest.mock.patch("brief._load_news_sources", return_value={"AI/Tech": []}),
            unittest.mock.patch("brief.fetch_top_news", return_value=fixture_items),
            unittest.mock.patch("d1_client.D1Client.query", return_value=[]),
        ]
        payload = self._run_dry_run(env, patches)
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 1)
        self.assertIn("AI breakthrough", news_fields[0]["value"])

    def test_news_fetch_failure_does_not_prevent_email_section(self):
        env = {
            "CF_ACCOUNT_ID": "acc123",
            "CF_D1_DATABASE_ID": "db-abc",
            "CF_API_TOKEN": "tok_abc",
        }
        patches = [
            unittest.mock.patch("brief._load_news_sources", side_effect=RuntimeError("news_sources.json not found")),
            unittest.mock.patch("d1_client.D1Client.query", return_value=[]),
        ]
        payload = self._run_dry_run(env, patches)
        # Email embed must still be present even though news fetch failed
        self.assertIn("embeds", payload)
        self.assertIn("Morning Brief", payload["embeds"][0]["title"])
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 0)


if __name__ == "__main__":
    unittest.main()
