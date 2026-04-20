"""
Integration tests for triage.py.

Run from: config/skills/email-triage/
    python -m pytest tests/test_triage_integration.py -v
or:
    python -m unittest tests/test_triage_integration.py
"""

import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, "scripts")

from email_types import EmailMessage
from triage import classify_batch, main, send_urgent_alert, _run_attribution_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_email(
    message_id: str = "msg-001",
    source: str = "gmail",
    from_addr: str = "boss@example.com",
    subject: str = "Hello",
    body_preview: str = "Hey there",
    is_junk_rescue: bool = False,
    headers: dict | None = None,
) -> EmailMessage:
    return EmailMessage(
        message_id=message_id,
        source=source,
        from_addr=from_addr,
        subject=subject,
        received_at="2026-04-07T10:00:00Z",
        body_preview=body_preview,
        is_junk_rescue=is_junk_rescue,
        headers=headers or {},
    )


_BASE_ENV = {
    "GMAIL_CLIENT_ID": "gcid",
    "GMAIL_CLIENT_SECRET": "gcs",
    "GMAIL_REFRESH_TOKEN": "grt",
    "OUTLOOK_CLIENT_ID": "outlook_client_id",
    "OUTLOOK_CLIENT_SECRET": "outlook_client_secret",
    "OUTLOOK_REFRESH_TOKEN": "outlook_refresh_token",
    "CF_ACCOUNT_ID": "acct123",
    "CF_D1_DATABASE_ID": "db123",
    "CF_API_TOKEN": "cftoken",
    "OPENROUTER_API_KEY": "orkey",
    "DISCORD_TRIAGE_WEBHOOK": "https://discord.com/api/webhooks/x",
}


def _patch_env(extra: dict | None = None):
    env = dict(_BASE_ENV)
    if extra:
        env.update(extra)
    return patch.dict("os.environ", env, clear=True)


def _make_d1(already_processed_ids: set | None = None):
    d1 = MagicMock()
    d1.is_already_processed.side_effect = lambda mid: mid in (already_processed_ids or set())
    return d1


def _llm_result(message_id: str, classification: str = "FYI", summary: str = "test") -> dict:
    return {"message_id": message_id, "classification": classification, "summary": summary}


# ---------------------------------------------------------------------------
# Test 1: Deduplication — emails already in D1 are skipped
# ---------------------------------------------------------------------------

class TestDeduplication(unittest.TestCase):
    def test_already_processed_emails_not_written_to_d1(self):
        email_existing = _make_email("existing-001")
        email_new = _make_email("new-001")

        d1 = _make_d1(already_processed_ids={"existing-001"})
        d1.get_priority_map.return_value = "priority map"

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[email_existing, email_new]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=[_llm_result("new-001", "FYI")]),
        ):
            main([])

        # D1 insert called once — only for the new email
        self.assertEqual(d1.insert_triage_result.call_count, 1)
        inserted = d1.insert_triage_result.call_args[0][0]
        self.assertEqual(inserted["message_id"], "new-001")


# ---------------------------------------------------------------------------
# Test 2: Deterministic NOISE skips LLM
# ---------------------------------------------------------------------------

class TestPrefilterSkipsLLM(unittest.TestCase):
    def test_noise_email_does_not_call_classify_batch(self):
        # Email with list-unsubscribe header is deterministically NOISE
        noise_email = _make_email(
            "noise-001",
            headers={"list-unsubscribe": "<https://example.com/unsub>"},
        )

        d1 = _make_d1()
        d1.get_priority_map.return_value = "priority map"

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[noise_email]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch") as mock_classify,
        ):
            main([])

        mock_classify.assert_not_called()
        # Still stored as NOISE
        self.assertEqual(d1.insert_triage_result.call_count, 1)
        inserted = d1.insert_triage_result.call_args[0][0]
        self.assertEqual(inserted["classification"], "NOISE")


# ---------------------------------------------------------------------------
# Test 3: URGENT email triggers send_urgent_alert subprocess
# ---------------------------------------------------------------------------

class TestUrgentAlert(unittest.TestCase):
    def test_urgent_email_triggers_alert(self):
        urgent_email = _make_email("urgent-001", from_addr="ceo@corp.com", subject="CRITICAL")

        d1 = _make_d1()
        d1.get_priority_map.return_value = "priority map"

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[urgent_email]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=[_llm_result("urgent-001", "URGENT", "Needs attention now")]),
            patch("triage.send_urgent_alert") as mock_alert,
        ):
            main([])

        mock_alert.assert_called_once()
        called_email, called_summary = mock_alert.call_args[0]
        self.assertEqual(called_email["message_id"], "urgent-001")
        self.assertEqual(called_summary, "Needs attention now")


# ---------------------------------------------------------------------------
# Test 4: D1 write happens before alert subprocess
# ---------------------------------------------------------------------------

class TestD1BeforeAlert(unittest.TestCase):
    def test_d1_insert_before_alert_subprocess(self):
        urgent_email = _make_email("urgent-002", subject="Server is down")

        d1 = _make_d1()
        d1.get_priority_map.return_value = "priority map"
        call_order = []

        d1.insert_triage_result.side_effect = lambda *a, **kw: call_order.append("d1_insert")

        def fake_alert(email, summary):
            call_order.append("alert")

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[urgent_email]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=[_llm_result("urgent-002", "URGENT", "Down now")]),
            patch("triage.send_urgent_alert", side_effect=fake_alert),
        ):
            main([])

        self.assertIn("d1_insert", call_order)
        self.assertIn("alert", call_order)
        d1_idx = call_order.index("d1_insert")
        alert_idx = call_order.index("alert")
        self.assertLess(d1_idx, alert_idx, "D1 insert must happen before alert")


# ---------------------------------------------------------------------------
# Test 5: --dry-run skips D1 write and alert subprocess
# ---------------------------------------------------------------------------

class TestDryRun(unittest.TestCase):
    def test_dry_run_no_d1_write_no_alert(self):
        urgent_email = _make_email("dry-001", subject="Fire!")

        d1 = _make_d1()

        # dry-run doesn't require DISCORD_TRIAGE_WEBHOOK
        env = {k: v for k, v in _BASE_ENV.items() if k != "DISCORD_TRIAGE_WEBHOOK"}

        d1.get_priority_map.return_value = "priority map"

        with (
            patch.dict("os.environ", env, clear=True),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[urgent_email]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=[_llm_result("dry-001", "URGENT", "Dry fire")]),
            patch("triage.send_urgent_alert") as mock_alert,
        ):
            main(["--dry-run"])

        d1.insert_triage_result.assert_not_called()
        mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Missing required env var raises RuntimeError before API calls
# ---------------------------------------------------------------------------

class TestMissingEnvVar(unittest.TestCase):
    def test_missing_openrouter_key_raises_before_api(self):
        env = {k: v for k, v in _BASE_ENV.items() if k != "OPENROUTER_API_KEY"}

        with (
            patch.dict("os.environ", env, clear=True),
            patch("triage.D1Client") as mock_d1_cls,
            patch("triage.gmail_client.refresh_access_token") as mock_refresh,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                main([])

        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        mock_refresh.assert_not_called()

    def test_missing_discord_webhook_raises_when_not_dry_run(self):
        env = {k: v for k, v in _BASE_ENV.items() if k != "DISCORD_TRIAGE_WEBHOOK"}

        with (
            patch.dict("os.environ", env, clear=True),
            patch("triage.gmail_client.refresh_access_token") as mock_refresh,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                main([])

        self.assertIn("DISCORD_TRIAGE_WEBHOOK", str(ctx.exception))
        mock_refresh.assert_not_called()

    def test_missing_discord_webhook_ok_in_dry_run(self):
        env = {k: v for k, v in _BASE_ENV.items() if k != "DISCORD_TRIAGE_WEBHOOK"}
        d1 = _make_d1()

        d1.get_priority_map.return_value = "priority map"

        with (
            patch.dict("os.environ", env, clear=True),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
        ):
            # Should not raise
            main(["--dry-run"])


# ---------------------------------------------------------------------------
# Test 7: LLM error sets NEEDS_ACTION, continues processing
# ---------------------------------------------------------------------------

class TestLLMClassificationError(unittest.TestCase):
    def test_llm_error_result_continues_and_uses_needs_action(self):
        email1 = _make_email("err-001", subject="Normal")
        email2 = _make_email("err-002", subject="Also normal")

        d1 = _make_d1()

        # Return error results (simulating what classify_batch does on parse failure)
        error_results = [
            {"message_id": "err-001", "classification": "NEEDS_ACTION", "summary": "", "classification_error": True},
            {"message_id": "err-002", "classification": "NEEDS_ACTION", "summary": "", "classification_error": True},
        ]

        d1.get_priority_map.return_value = "priority map"

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[email1, email2]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=error_results),
            patch("triage.send_urgent_alert") as mock_alert,
        ):
            # Should not raise
            main([])

        # Both emails stored as NEEDS_ACTION with error flag
        self.assertEqual(d1.insert_triage_result.call_count, 2)
        for c in d1.insert_triage_result.call_args_list:
            record = c[0][0]
            self.assertEqual(record["classification"], "NEEDS_ACTION")
            self.assertEqual(record["classification_error"], 1)

        mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: Gmail failure doesn't prevent Outlook results from being stored
# ---------------------------------------------------------------------------

class TestGmailFailurePartialSuccess(unittest.TestCase):
    def test_outlook_stored_even_if_gmail_fails(self):
        outlook_email = _make_email("out-001", source="outlook", subject="Outlook OK")

        d1 = _make_d1()

        d1.get_priority_map.return_value = "priority map"

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", side_effect=RuntimeError("Gmail down")),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([outlook_email], "")),
            patch("triage.classify_batch", return_value=[_llm_result("out-001", "FYI", "From Outlook")]),
        ):
            # Should not raise; Gmail error is caught and logged
            main([])

        # Outlook email was stored
        self.assertEqual(d1.insert_triage_result.call_count, 1)
        inserted = d1.insert_triage_result.call_args[0][0]
        self.assertEqual(inserted["message_id"], "out-001")
        self.assertEqual(inserted["source"], "outlook")


# ---------------------------------------------------------------------------
# Unit tests for classify_batch error handling
# ---------------------------------------------------------------------------

class TestClassifyBatchErrorHandling(unittest.TestCase):
    def test_http_error_non_auth_returns_needs_action(self):
        import urllib.error
        emails = [_make_email("b-001"), _make_email("b-002")]

        http_error = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        # HTTPError.read() needs to return bytes
        http_error.read = lambda: b"server error"

        with patch("triage._OPENER") as mock_opener:
            mock_opener.open.side_effect = http_error
            results = classify_batch(emails, "priority map", "apikey", "model")

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["classification"], "NEEDS_ACTION")
            self.assertTrue(r["classification_error"])

    def test_auth_failure_raises_immediately(self):
        import urllib.error
        emails = [_make_email("b-003")]

        http_error = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )
        http_error.read = lambda: b"unauthorized"

        with patch("triage._OPENER") as mock_opener:
            mock_opener.open.side_effect = http_error
            with self.assertRaises(RuntimeError) as ctx:
                classify_batch(emails, "priority map", "badkey", "model")

        self.assertIn("auth failure", str(ctx.exception).lower())

    def test_json_parse_failure_returns_needs_action(self):
        import json
        emails = [_make_email("b-004")]

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "not valid json["}}]
        }).encode()

        with patch("triage._OPENER") as mock_opener:
            mock_opener.open.return_value = mock_resp
            results = classify_batch(emails, "priority map", "apikey", "model")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["classification"], "NEEDS_ACTION")
        self.assertTrue(results[0]["classification_error"])


# ---------------------------------------------------------------------------
# Test 9: Priority map read from D1
# ---------------------------------------------------------------------------

class TestPriorityMapFromD1(unittest.TestCase):

    def test_triage_uses_priority_map_from_d1(self):
        email = _make_email("msg-pm-001")
        d1 = _make_d1()
        d1.get_priority_map.return_value = "## URGENT\nTest priority map."

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[email]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
            patch("triage.classify_batch", return_value=[_llm_result("msg-pm-001", "FYI")]) as mock_classify,
        ):
            main(["--dry-run"])

        args, _ = mock_classify.call_args
        self.assertEqual(args[1], "## URGENT\nTest priority map.")

    def test_triage_raises_when_d1_priority_map_unavailable(self):
        d1 = _make_d1()
        d1.get_priority_map.side_effect = RuntimeError("priority_map table is empty")

        with (
            _patch_env(),
            patch("triage.D1Client", return_value=d1),
            patch("triage.gmail_client.refresh_access_token", return_value="tok"),
            patch("triage.gmail_client.fetch_unread_emails", return_value=[]),
            patch("triage.outlook_client.fetch_unread_emails", return_value=([], "")),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                main(["--dry-run"])
        self.assertIn("priority_map table is empty", str(ctx.exception))


_UNATTRIBUTED_ROW = {
    "message_id": "msg-1",
    "conversation_id": "conv-abc",
    "from_addr": "alice@example.com",
    "subject": "Budget review",
    "classification": "URGENT",
}


class TestAttributionPass(unittest.TestCase):

    def test_calls_honcho_before_mark_replied(self):
        mock_d1 = MagicMock()
        mock_d1.get_unattributed_recent.return_value = [_UNATTRIBUTED_ROW]
        call_order = []

        def fake_conclude(*a, **kw):
            call_order.append("honcho")

        def fake_mark_replied(*a, **kw):
            call_order.append("d1")

        mock_d1.mark_replied.side_effect = fake_mark_replied

        with (
            patch.dict("os.environ", {"HONCHO_API_KEY": "test-key"}),
            patch("triage.outlook_client.refresh_access_token", return_value=("acc-tok", "")),
            patch("triage.outlook_client.fetch_sent_replies", return_value={"conv-abc": "2026-04-19T10:00:00Z"}),
            patch("triage.honcho_client.conclude", side_effect=fake_conclude),
            patch("triage._kv_get", return_value=None),
        ):
            _run_attribution_pass(_BASE_ENV, mock_d1, dry_run=False)

        self.assertEqual(call_order, ["honcho", "d1"])

    def test_does_not_call_mark_replied_if_honcho_raises(self):
        second_row = {
            "message_id": "msg-2",
            "conversation_id": "conv-xyz",
            "from_addr": "bob@example.com",
            "subject": "Q4 plan",
            "classification": "NEEDS_ACTION",
        }
        mock_d1 = MagicMock()
        mock_d1.get_unattributed_recent.return_value = [_UNATTRIBUTED_ROW, second_row]

        with (
            patch.dict("os.environ", {"HONCHO_API_KEY": "test-key"}),
            patch("triage.outlook_client.refresh_access_token", return_value=("acc-tok", "")),
            patch(
                "triage.outlook_client.fetch_sent_replies",
                return_value={
                    "conv-abc": "2026-04-19T10:00:00Z",
                    "conv-xyz": "2026-04-19T11:00:00Z",
                },
            ),
            patch("triage.honcho_client.conclude", side_effect=RuntimeError("Honcho down")),
            patch("triage._kv_get", return_value=None),
        ):
            _run_attribution_pass(_BASE_ENV, mock_d1, dry_run=False)

        mock_d1.mark_replied.assert_not_called()

    def test_exits_cleanly_when_outlook_refresh_raises(self):
        mock_d1 = MagicMock()
        mock_d1.get_unattributed_recent.return_value = [_UNATTRIBUTED_ROW]

        with (
            patch.dict("os.environ", {"HONCHO_API_KEY": "test-key"}),
            patch("triage.outlook_client.refresh_access_token", side_effect=RuntimeError("Outlook down")),
            patch("triage._kv_get", return_value=None),
        ):
            # Must not raise
            _run_attribution_pass(_BASE_ENV, mock_d1, dry_run=False)

        mock_d1.mark_replied.assert_not_called()


if __name__ == "__main__":
    unittest.main()
