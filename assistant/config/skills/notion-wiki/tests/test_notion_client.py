import sys
sys.path.insert(0, 'scripts')

import json
import unittest
from unittest.mock import MagicMock, patch

from notion_client import NotionWikiReader, _OPENER


def _make_response(payload: dict, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_reader() -> NotionWikiReader:
    return NotionWikiReader(token="t", sources_db_id="src", concepts_db_id="con")


class TestReaderInit(unittest.TestCase):
    def test_missing_token_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiReader(token="", sources_db_id="s", concepts_db_id="c")
        self.assertIn("NOTION_WIKI_READ_TOKEN", str(ctx.exception))

    def test_missing_sources_db_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiReader(token="t", sources_db_id="", concepts_db_id="c")
        self.assertIn("NOTION_WIKI_SOURCES_DB_ID", str(ctx.exception))

    def test_missing_concepts_db_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiReader(token="t", sources_db_id="s", concepts_db_id="")
        self.assertIn("NOTION_WIKI_CONCEPTS_DB_ID", str(ctx.exception))

    def test_valid_init_does_not_open_network(self):
        with patch.object(_OPENER, "open") as mock_open:
            reader = _make_reader()
            mock_open.assert_not_called()
        self.assertIsNotNone(reader)


class TestReaderSearch(unittest.TestCase):
    def test_search_returns_ranked_list(self):
        api_response = {
            "results": [
                {
                    "object": "page",
                    "id": "con-1",
                    "parent": {"type": "database_id", "database_id": "con"},
                    "properties": {
                        "Title": {"type": "title", "title": [{"plain_text": "Speculative Decoding"}]},
                    },
                },
                {
                    "object": "page",
                    "id": "src-1",
                    "parent": {"type": "database_id", "database_id": "src"},
                    "properties": {
                        "Title": {"type": "title", "title": [{"plain_text": "Fast Inference Paper"}]},
                    },
                },
            ],
            "has_more": False,
            "next_cursor": None,
        }
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(api_response)

        with patch.object(_OPENER, "open", side_effect=capture):
            reader = _make_reader()
            results = reader.search("speculative decoding", limit=5)

        self.assertIn("/search", captured[0].full_url)
        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(body["query"], "speculative decoding")
        self.assertEqual(body["page_size"], 5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "con-1")
        self.assertEqual(results[0]["title"], "Speculative Decoding")
        self.assertEqual(results[0]["db"], "concepts")
        self.assertEqual(results[1]["db"], "sources")


if __name__ == "__main__":
    unittest.main()
