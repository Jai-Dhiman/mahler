import sys
import unittest
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, patch, call
import imaplib

sys.path.insert(0, "scripts")

from outlook_client import fetch_unread_emails


def _make_raw_email(
    subject: str = "Test Subject",
    from_addr: str = "Sender <sender@example.com>",
    body: str = "Hello world",
    message_id: str = "<abc123@mail.example.com>",
    date: str = "Mon, 01 Jan 2024 12:00:00 +0000",
    content_type: str = "plain",
) -> bytes:
    msg = MIMEText(body, content_type)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["Message-ID"] = message_id
    msg["Date"] = date
    return msg.as_bytes()


def _make_fetch_response(raw_bytes: bytes) -> list:
    return [(b"1 (RFC822 {" + str(len(raw_bytes)).encode() + b"}", raw_bytes), b")"]


def _make_imap_mock(
    inbox_uids: bytes = b"1",
    junk_uids: bytes = b"",
    inbox_raw: bytes = None,
    junk_raw: bytes = None,
    junk_select_status: str = "OK",
) -> MagicMock:
    mock_imap = MagicMock()
    mock_imap.login.return_value = ("OK", [b"Logged in"])

    inbox_default_raw = _make_raw_email()
    junk_default_raw = _make_raw_email(
        subject="Junk Mail",
        message_id="<junk1@mail.example.com>",
    )

    actual_inbox_raw = inbox_raw if inbox_raw is not None else inbox_default_raw
    actual_junk_raw = junk_raw if junk_raw is not None else junk_default_raw

    def select_side_effect(folder, readonly=True):
        if folder == "INBOX":
            return ("OK", [b"1"])
        elif folder == "Junk":
            return (junk_select_status, [b"1"])
        return ("NO", [b"[NONEXISTENT]"])

    mock_imap.select.side_effect = select_side_effect

    def search_side_effect(charset, criteria):
        folder_name = mock_imap.select.call_args[0][0]
        if folder_name == "INBOX":
            return ("OK", [inbox_uids])
        elif folder_name == "Junk":
            return ("OK", [junk_uids])
        return ("OK", [b""])

    mock_imap.search.side_effect = search_side_effect

    def fetch_side_effect(uid, spec):
        folder_name = mock_imap.select.call_args[0][0]
        if folder_name == "INBOX":
            return ("OK", _make_fetch_response(actual_inbox_raw))
        elif folder_name == "Junk":
            return ("OK", _make_fetch_response(actual_junk_raw))
        return ("NO", [])

    mock_imap.fetch.side_effect = fetch_side_effect
    mock_imap.store.return_value = ("OK", [b"1"])
    mock_imap.logout.return_value = ("BYE", [b"Logging out"])

    return mock_imap


class TestFetchUnreadEmails(unittest.TestCase):
    def _run_fetch(self, mock_imap: MagicMock) -> list:
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            return fetch_unread_emails("imap.outlook.com", "user@outlook.com", "pass")

    # Test 1: INBOX fetch returns correctly shaped EmailMessage with is_junk_rescue=False
    def test_inbox_fetch_returns_correct_shape(self):
        raw = _make_raw_email(
            subject="Hello INBOX",
            from_addr="Alice <alice@example.com>",
            body="This is the body",
            message_id="<inbox1@mail.example.com>",
        )
        mock_imap = _make_imap_mock(inbox_uids=b"1", junk_uids=b"", inbox_raw=raw)

        results = self._run_fetch(mock_imap)

        inbox_msgs = [m for m in results if not m["is_junk_rescue"]]
        self.assertEqual(len(inbox_msgs), 1)

        msg = inbox_msgs[0]
        self.assertEqual(msg["source"], "outlook")
        self.assertEqual(msg["message_id"], "inbox1@mail.example.com")
        self.assertEqual(msg["subject"], "Hello INBOX")
        self.assertIn("alice@example.com", msg["from_addr"])
        self.assertFalse(msg["is_junk_rescue"])
        self.assertIn("body_preview", msg)
        self.assertIn("received_at", msg)
        self.assertIsInstance(msg["headers"], dict)
        self.assertIn("subject", msg["headers"])

    # Test 2: Junk folder fetch returns emails with is_junk_rescue=True
    def test_junk_fetch_returns_junk_rescue_true(self):
        raw = _make_raw_email(
            subject="Junk Subject",
            message_id="<junkmail@mail.example.com>",
        )
        mock_imap = _make_imap_mock(inbox_uids=b"", junk_uids=b"2", junk_raw=raw)

        results = self._run_fetch(mock_imap)

        junk_msgs = [m for m in results if m["is_junk_rescue"]]
        self.assertEqual(len(junk_msgs), 1)
        self.assertTrue(junk_msgs[0]["is_junk_rescue"])
        self.assertEqual(junk_msgs[0]["subject"], "Junk Subject")

    # Test 3: Both folders fetched - total results include both
    def test_both_folders_fetched_combined_results(self):
        inbox_raw = _make_raw_email(
            subject="Inbox Email",
            message_id="<inbox@mail.example.com>",
        )
        junk_raw = _make_raw_email(
            subject="Junk Email",
            message_id="<junk@mail.example.com>",
        )
        mock_imap = _make_imap_mock(
            inbox_uids=b"1",
            junk_uids=b"2",
            inbox_raw=inbox_raw,
            junk_raw=junk_raw,
        )

        results = self._run_fetch(mock_imap)

        self.assertEqual(len(results), 2)
        inbox_msgs = [m for m in results if not m["is_junk_rescue"]]
        junk_msgs = [m for m in results if m["is_junk_rescue"]]
        self.assertEqual(len(inbox_msgs), 1)
        self.assertEqual(len(junk_msgs), 1)

        # Confirm select was called for both folders
        select_calls = [c[0][0] for c in mock_imap.select.call_args_list]
        self.assertIn("INBOX", select_calls)
        self.assertIn("Junk", select_calls)

    # Test 4: Empty INBOX + empty Junk returns []
    def test_empty_folders_returns_empty_list(self):
        mock_imap = _make_imap_mock(inbox_uids=b"", junk_uids=b"")

        results = self._run_fetch(mock_imap)

        self.assertEqual(results, [])

    # Test 5: Login failure raises RuntimeError
    def test_login_failure_raises_runtime_error(self):
        mock_imap = MagicMock()
        mock_imap.login.side_effect = imaplib.IMAP4.error("Authentication failed")
        mock_imap.logout.return_value = ("BYE", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            with self.assertRaises(RuntimeError) as ctx:
                fetch_unread_emails("imap.outlook.com", "user@outlook.com", "wrongpass")

        self.assertIn("login failed", str(ctx.exception).lower())

    # Test 6: Missing Junk folder (select fails) is skipped - INBOX results still returned
    def test_missing_junk_folder_skipped_inbox_returned(self):
        inbox_raw = _make_raw_email(
            subject="INBOX Only",
            message_id="<inboxonly@mail.example.com>",
        )
        mock_imap = _make_imap_mock(
            inbox_uids=b"1",
            junk_uids=b"",
            inbox_raw=inbox_raw,
            junk_select_status="NO",
        )

        results = self._run_fetch(mock_imap)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["is_junk_rescue"])
        self.assertEqual(results[0]["subject"], "INBOX Only")

    # Test 7: logout() is called even when an exception occurs during fetch
    def test_logout_called_on_exception(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [b"Logged in"])
        mock_imap.select.return_value = ("OK", [b"1"])
        mock_imap.search.side_effect = Exception("Network error")
        mock_imap.logout.return_value = ("BYE", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            with self.assertRaises(Exception):
                fetch_unread_emails("imap.outlook.com", "user@outlook.com", "pass")

        mock_imap.logout.assert_called_once()

    # Test 8: UNSEEN search returning no UIDs returns [] for that folder
    def test_unseen_search_no_uids_returns_empty(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [b"Logged in"])
        mock_imap.select.return_value = ("OK", [b"1"])
        # search returns OK but with empty data
        mock_imap.search.return_value = ("OK", [b""])
        mock_imap.logout.return_value = ("BYE", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = fetch_unread_emails("imap.outlook.com", "user@outlook.com", "pass")

        self.assertEqual(results, [])
        mock_imap.fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
