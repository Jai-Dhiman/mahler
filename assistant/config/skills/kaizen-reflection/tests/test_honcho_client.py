import json
import sys
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from honcho_client import conclude, _OPENER


def _make_response(status: int = 201) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = b"{}"
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("", code, "err", {}, BytesIO(b""))


class TestConclude(unittest.TestCase):

    def test_conclude_sends_content_to_metamessages_endpoint(self):
        captured = {}

        def fake_open(req, **kwargs):
            if "metamessages" in req.full_url:
                captured["url"] = req.full_url
                captured["body"] = json.loads(req.data)
                return _make_response(201)
            raise _http_error(409)

        with patch.object(_OPENER, "open", side_effect=fake_open):
            conclude(
                "traderjoe had 5 blockers this week",
                "test-key",
                "https://api.honcho.dev",
                "mahler",
                "jai",
            )

        self.assertIn("metamessages", captured["url"])
        self.assertIn("mahler", captured["url"])
        self.assertIn("jai", captured["url"])
        self.assertEqual(captured["body"]["content"], "traderjoe had 5 blockers this week")
        self.assertEqual(captured["body"]["metamessage_type"], "honcho_conclude")

    def test_conclude_raises_runtime_error_on_metamessage_http_error(self):
        def fake_open(req, **kwargs):
            if "metamessages" in req.full_url:
                raise _http_error(500)
            raise _http_error(409)

        with patch.object(_OPENER, "open", side_effect=fake_open):
            with self.assertRaises(RuntimeError) as ctx:
                conclude("text", "key", "https://api.honcho.dev", "mahler", "jai")

        self.assertIn("Honcho conclude failed", str(ctx.exception))

    def test_conclude_raises_runtime_error_on_session_creation_http_error(self):
        with patch.object(_OPENER, "open", side_effect=_http_error(401)):
            with self.assertRaises(RuntimeError) as ctx:
                conclude("text", "key", "https://api.honcho.dev", "mahler", "jai")

        self.assertIn("Honcho session creation failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
