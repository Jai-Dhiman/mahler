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


class TestReaderReadPage(unittest.TestCase):
    def test_read_page_returns_title_and_type(self):
        page_response = {
            "object": "page",
            "id": "src-1",
            "parent": {"type": "database_id", "database_id": "src"},
            "properties": {
                "Title": {"type": "title", "title": [{"plain_text": "Paper X"}]},
                "Type": {"type": "select", "select": {"name": "paper"}},
                "URL": {"type": "url", "url": "https://example.com/x"},
            },
        }
        blocks_response = {"results": [], "has_more": False, "next_cursor": None}
        responses = [_make_response(page_response), _make_response(blocks_response)]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            reader = _make_reader()
            result = reader.read_page("src-1")

        self.assertEqual(result["id"], "src-1")
        self.assertEqual(result["title"], "Paper X")
        self.assertEqual(result["type"], "paper")
        self.assertEqual(result["url"], "https://example.com/x")
        self.assertEqual(result["body_markdown"], "")
        self.assertEqual(result["related_sources"], [])
        self.assertIn("/pages/src-1", calls[0].full_url)
        self.assertIn("/blocks/src-1/children", calls[1].full_url)


    def test_paragraph_blocks_become_markdown_paragraphs(self):
        page_response = {
            "id": "src-p",
            "properties": {
                "Title": {"type": "title", "title": [{"plain_text": "T"}]},
            },
        }
        blocks_response = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "First."}]},
                },
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "Second line."}]},
                },
            ],
            "has_more": False,
            "next_cursor": None,
        }
        responses = [_make_response(page_response), _make_response(blocks_response)]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            reader = _make_reader()
            result = reader.read_page("src-p")

        self.assertEqual(result["body_markdown"], "First.\n\nSecond line.")


    def test_heading_blocks_become_markdown_headers(self):
        page_response = {
            "id": "src-h",
            "properties": {"Title": {"type": "title", "title": [{"plain_text": "T"}]}},
        }
        blocks_response = {
            "results": [
                {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Top"}]}},
                {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "Sub"}]}},
                {"type": "heading_3", "heading_3": {"rich_text": [{"plain_text": "Sub-sub"}]}},
            ],
            "has_more": False,
            "next_cursor": None,
        }
        responses = [_make_response(page_response), _make_response(blocks_response)]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            reader = _make_reader()
            result = reader.read_page("src-h")

        self.assertEqual(result["body_markdown"], "# Top\n\n## Sub\n\n### Sub-sub")


    def test_bulleted_list_items_render_as_dashes(self):
        page_response = {
            "id": "src-l",
            "properties": {"Title": {"type": "title", "title": [{"plain_text": "T"}]}},
        }
        blocks_response = {
            "results": [
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "first"}]}},
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "second"}]}},
            ],
            "has_more": False,
            "next_cursor": None,
        }
        responses = [_make_response(page_response), _make_response(blocks_response)]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            reader = _make_reader()
            result = reader.read_page("src-l")

        self.assertEqual(result["body_markdown"], "- first\n\n- second")


if __name__ == "__main__":
    unittest.main()
