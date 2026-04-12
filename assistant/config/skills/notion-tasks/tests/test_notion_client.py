import sys
sys.path.insert(0, 'scripts')

import json
import unittest
from unittest.mock import MagicMock, patch

from notion_client import NotionClient, _OPENER


def _make_response(payload: dict, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _page_fixture(
    page_id: str = "page-abc123",
    title: str = "Test task",
    status: str = "Todo",
    due: str | None = None,
    priority: str | None = None,
) -> dict:
    return {
        "object": "page",
        "id": page_id,
        "properties": {
            "Name": {"title": [{"plain_text": title}]},
            "Status": {"select": {"name": status}},
            "Due": {"date": {"start": due} if due else None},
            "Priority": {"select": {"name": priority} if priority else None},
        },
    }


def _list_response(
    pages: list,
    has_more: bool = False,
    next_cursor: str | None = None,
) -> dict:
    return {
        "object": "list",
        "results": pages,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


def _make_client() -> NotionClient:
    return NotionClient(api_token="test-token", database_id="test-db-id")


class TestNotionClientInit(unittest.TestCase):

    def test_empty_api_token_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionClient(api_token="", database_id="test-db-id")
        self.assertIn("NOTION_API_TOKEN", str(ctx.exception))

    def test_empty_database_id_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionClient(api_token="test-token", database_id="")
        self.assertIn("NOTION_DATABASE_ID", str(ctx.exception))

    def test_valid_credentials_succeed_without_network_calls(self):
        with patch.object(_OPENER, "open") as mock_open:
            client = NotionClient(api_token="tok", database_id="db-id")
            mock_open.assert_not_called()
        self.assertIsNotNone(client)


if __name__ == "__main__":
    unittest.main()
