import sys
sys.path.insert(0, 'scripts')

import json
import unittest
from unittest.mock import MagicMock, patch

from notion_client import NotionWikiWriter, _OPENER


def _make_response(payload: dict, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_writer() -> NotionWikiWriter:
    return NotionWikiWriter(
        token="test-token",
        sources_db_id="src-db",
        concepts_db_id="con-db",
        log_db_id="log-db",
    )


class TestNotionWikiWriterInit(unittest.TestCase):
    def test_missing_token_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiWriter(token="", sources_db_id="s", concepts_db_id="c", log_db_id="l")
        self.assertIn("NOTION_WIKI_WRITE_TOKEN", str(ctx.exception))

    def test_missing_sources_db_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiWriter(token="t", sources_db_id="", concepts_db_id="c", log_db_id="l")
        self.assertIn("NOTION_WIKI_SOURCES_DB_ID", str(ctx.exception))

    def test_missing_concepts_db_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiWriter(token="t", sources_db_id="s", concepts_db_id="", log_db_id="l")
        self.assertIn("NOTION_WIKI_CONCEPTS_DB_ID", str(ctx.exception))

    def test_missing_log_db_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            NotionWikiWriter(token="t", sources_db_id="s", concepts_db_id="c", log_db_id="")
        self.assertIn("NOTION_WIKI_LOG_DB_ID", str(ctx.exception))

    def test_valid_args_construct_without_network(self):
        with patch.object(_OPENER, "open") as mock_open:
            writer = _make_writer()
            mock_open.assert_not_called()
        self.assertIsNotNone(writer)


class TestFindSourceByURL(unittest.TestCase):
    def test_returns_page_dict_when_url_matches(self):
        matching_page = {
            "id": "page-src-1",
            "properties": {
                "URL": {"url": "https://example.com/paper"},
                "Title": {"title": [{"plain_text": "Example Paper"}]},
            },
        }
        query_response = {"results": [matching_page], "has_more": False, "next_cursor": None}
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(query_response)

        with patch.object(_OPENER, "open", side_effect=capture):
            writer = _make_writer()
            result = writer.find_source_by_url("https://example.com/paper")

        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "page-src-1")
        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(
            body["filter"],
            {"property": "URL", "url": {"equals": "https://example.com/paper"}},
        )
        self.assertIn("/databases/src-db/query", captured[0].full_url)

    def test_returns_none_when_url_not_found(self):
        empty_response = {"results": [], "has_more": False, "next_cursor": None}
        with patch.object(_OPENER, "open", return_value=_make_response(empty_response)):
            writer = _make_writer()
            result = writer.find_source_by_url("https://example.com/missing")
        self.assertIsNone(result)


class TestFindConceptByTitle(unittest.TestCase):
    def test_case_insensitive_match_returns_page(self):
        page = {
            "id": "page-con-1",
            "properties": {
                "Title": {"title": [{"plain_text": "speculative decoding"}]},
            },
        }
        query_response = {"results": [page], "has_more": False, "next_cursor": None}
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(query_response)

        with patch.object(_OPENER, "open", side_effect=capture):
            writer = _make_writer()
            result = writer.find_concept_by_title("Speculative Decoding")

        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "page-con-1")
        self.assertIn("/databases/con-db/query", captured[0].full_url)

    def test_returns_none_when_no_exact_match(self):
        page = {
            "id": "page-con-near",
            "properties": {
                "Title": {"title": [{"plain_text": "Speculative Decoding Variants"}]},
            },
        }
        response = {"results": [page], "has_more": False, "next_cursor": None}
        with patch.object(_OPENER, "open", return_value=_make_response(response)):
            writer = _make_writer()
            result = writer.find_concept_by_title("Speculative Decoding")
        self.assertIsNone(result)


class TestCreateSource(unittest.TestCase):
    def test_basic_create_sends_correct_payload(self):
        created_page = {
            "id": "new-src-id",
            "properties": {
                "Title": {"title": [{"plain_text": "Paper"}]},
                "URL": {"url": "https://example.com/p"},
            },
        }
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(created_page)

        with patch.object(_OPENER, "open", side_effect=capture):
            writer = _make_writer()
            result = writer.create_source(
                url="https://example.com/p",
                title="Paper",
                type_="paper",
                summary="First paragraph.\n\nSecond paragraph.",
                ingested="2026-04-12",
            )

        self.assertIn("/pages", captured[0].full_url)
        self.assertEqual(captured[0].method, "POST")
        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(body["parent"], {"database_id": "src-db"})
        props = body["properties"]
        self.assertEqual(props["Title"]["title"][0]["text"]["content"], "Paper")
        self.assertEqual(props["URL"]["url"], "https://example.com/p")
        self.assertEqual(props["Type"]["select"]["name"], "paper")
        self.assertEqual(props["Ingested"]["date"]["start"], "2026-04-12")
        self.assertNotIn("Tags", props)
        self.assertNotIn("Concepts", props)
        children = body["children"]
        self.assertEqual(len(children), 2)
        self.assertEqual(
            children[0]["paragraph"]["rich_text"][0]["text"]["content"],
            "First paragraph.",
        )
        self.assertEqual(
            children[1]["paragraph"]["rich_text"][0]["text"]["content"],
            "Second paragraph.",
        )
        self.assertEqual(result["id"], "new-src-id")

    def test_create_with_tags_sets_multi_select(self):
        page = {"id": "src-2", "properties": {"Title": {"title": [{"plain_text": "p"}]}}}
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(page)

        with patch.object(_OPENER, "open", side_effect=capture):
            writer = _make_writer()
            writer.create_source(
                url="https://example.com/p2",
                title="p",
                type_="article",
                summary="x",
                tags=["llm", "inference"],
                ingested="2026-04-12",
            )

        body = json.loads(captured[0].data.decode("utf-8"))
        tag_names = [t["name"] for t in body["properties"]["Tags"]["multi_select"]]
        self.assertEqual(tag_names, ["llm", "inference"])

    def test_tag_with_comma_raises(self):
        with patch.object(_OPENER, "open") as mock_open:
            writer = _make_writer()
            with self.assertRaises(RuntimeError) as ctx:
                writer.create_source(
                    url="https://example.com/p3",
                    title="p",
                    type_="article",
                    summary="x",
                    tags=["llm,inference"],
                    ingested="2026-04-12",
                )
            mock_open.assert_not_called()
        self.assertIn("comma", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
