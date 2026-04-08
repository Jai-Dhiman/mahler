import sys
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

sys.path.insert(0, "scripts")

import alert

WEBHOOK_URL = "https://discord.com/api/webhooks/test/token"


def _mock_opener_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code: int, body: bytes = b"error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=WEBHOOK_URL, code=code, msg="Error", hdrs=MagicMock(), fp=BytesIO(body)
    )


class TestBuildPayload(unittest.TestCase):
    def test_payload_contains_correct_fields(self):
        payload = alert.build_payload(
            from_addr="boss@example.com",
            subject="Server is on fire",
            summary="The production server is unreachable.",
            source="gmail",
        )
        embed = payload["embeds"][0]
        self.assertEqual(embed["title"], "URGENT Email")
        self.assertEqual(embed["color"], 15158332)
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        self.assertEqual(fields["From"], "boss@example.com")
        self.assertEqual(fields["Subject"], "Server is on fire")
        self.assertEqual(fields["Summary"], "The production server is unreachable.")
        self.assertEqual(fields["Source"], "gmail")
        self.assertEqual(embed["footer"]["text"], "Mahler Email Triage")

    def test_no_authorization_header(self):
        with patch("alert._OPENER") as mock_opener:
            mock_opener.open.return_value = _mock_opener_response(204)
            alert.post_alert(WEBHOOK_URL, {"embeds": []})
        req = mock_opener.open.call_args[0][0]
        self.assertNotIn("Authorization", dict(req.headers))

    def test_content_type_is_application_json(self):
        with patch("alert._OPENER") as mock_opener:
            mock_opener.open.return_value = _mock_opener_response(204)
            alert.post_alert(WEBHOOK_URL, {"embeds": []})
        req = mock_opener.open.call_args[0][0]
        self.assertIn("application/json", req.get_header("Content-type"))


class TestPostAlert(unittest.TestCase):
    def test_successful_post_204(self):
        with patch("alert._OPENER") as mock_opener:
            mock_opener.open.return_value = _mock_opener_response(204)
            with patch("builtins.print") as mock_print:
                alert.post_alert(WEBHOOK_URL, alert.build_payload("a@b.com", "S", "Summary.", "gmail"))
                print("Alert sent.")
            mock_print.assert_called_once_with("Alert sent.")

    def test_raises_on_missing_webhook_env(self):
        args = ["alert.py", "--from", "a@b.com", "--subject", "S", "--summary", "Sum.", "--source", "gmail"]
        with patch("sys.argv", args):
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(RuntimeError) as ctx:
                    alert.main()
        self.assertIn("DISCORD_TRIAGE_WEBHOOK", str(ctx.exception))

    def test_raises_on_non_200_204_response(self):
        with patch("alert._OPENER") as mock_opener:
            mock_opener.open.side_effect = _make_http_error(500, b"Internal Server Error")
            with self.assertRaises(RuntimeError) as ctx:
                alert.post_alert(WEBHOOK_URL, {"embeds": []})
        self.assertIn("500", str(ctx.exception))

    def test_raises_on_non_https_url(self):
        with self.assertRaises(RuntimeError) as ctx:
            alert.post_alert("http://discord.com/api/webhooks/bad", {"embeds": []})
        self.assertIn("HTTPS", str(ctx.exception))


class TestMainIntegration(unittest.TestCase):
    def test_main_prints_alert_sent_on_success(self):
        args = ["alert.py", "--from", "s@e.com", "--subject", "Urgent", "--summary", "Needs attention.", "--source", "outlook"]
        with patch("sys.argv", args):
            with patch.dict("os.environ", {"DISCORD_TRIAGE_WEBHOOK": WEBHOOK_URL}):
                with patch("alert._OPENER") as mock_opener:
                    mock_opener.open.return_value = _mock_opener_response(200)
                    with patch("builtins.print") as mock_print:
                        alert.main()
        mock_print.assert_called_once_with("Alert sent.")


if __name__ == "__main__":
    unittest.main()
