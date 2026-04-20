import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from d1_client import D1Client, _OPENER


def _make_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _success_payload(rows: list) -> dict:
    return {
        "result": [{"results": rows, "success": True}],
        "success": True,
        "errors": [],
        "messages": [],
    }


def _make_client() -> D1Client:
    return D1Client(
        account_id="test-account-123",
        database_id="test-db-456",
        api_token="test-token-abc",
    )


class TestInsertReflection(unittest.TestCase):

    def test_insert_reflection_sends_week_of_and_raw_text_to_d1(self):
        captured = {}

        def fake_open(req, **kwargs):
            body = json.loads(req.data)
            captured["sql"] = body["sql"]
            captured["params"] = body["params"]
            return _make_response(_success_payload([]))

        with patch.object(_OPENER, "open", side_effect=fake_open):
            client = _make_client()
            client.insert_reflection("2026-W16", "Good week overall. Meetings were tiring.")

        self.assertIn("reflection_log", captured["sql"])
        self.assertIn("INSERT", captured["sql"].upper())
        self.assertIn("2026-W16", captured["params"])
        self.assertIn("Good week overall. Meetings were tiring.", captured["params"])

    def test_insert_reflection_raises_on_d1_error(self):
        error_payload = {
            "result": [],
            "success": False,
            "errors": [{"message": "no such table: reflection_log"}],
            "messages": [],
        }
        with patch.object(_OPENER, "open", return_value=_make_response(error_payload)):
            client = _make_client()
            with self.assertRaises(RuntimeError) as ctx:
                client.insert_reflection("2026-W16", "Some text")

        self.assertIn("D1 query failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
