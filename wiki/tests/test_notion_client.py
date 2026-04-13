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

    def test_create_with_concept_ids_sets_relation(self):
        page = {"id": "src-3", "properties": {"Title": {"title": [{"plain_text": "p"}]}}}
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(page)

        with patch.object(_OPENER, "open", side_effect=capture):
            writer = _make_writer()
            writer.create_source(
                url="https://example.com/p4",
                title="p",
                type_="paper",
                summary="x",
                concept_ids=["con-1", "con-2"],
                ingested="2026-04-12",
            )

        body = json.loads(captured[0].data.decode("utf-8"))
        relation = body["properties"]["Concepts"]["relation"]
        self.assertEqual(relation, [{"id": "con-1"}, {"id": "con-2"}])

    def test_non_2xx_response_raises(self):
        error_body = {"object": "error", "status": 400, "message": "validation"}
        with patch.object(_OPENER, "open", return_value=_make_response(error_body, status=400)):
            writer = _make_writer()
            with self.assertRaises(RuntimeError) as ctx:
                writer.create_source(
                    url="https://example.com/p5",
                    title="p",
                    type_="paper",
                    summary="x",
                    ingested="2026-04-12",
                )
        self.assertIn("400", str(ctx.exception))
        self.assertIn("validation", str(ctx.exception))


class TestAppendLog(unittest.TestCase):
    def test_append_log_sends_correct_payload(self):
        page = {"id": "log-1", "properties": {}}
        captured = []

        def capture(req):
            captured.append(req)
            return _make_response(page)

        with patch.object(_OPENER, "open", side_effect=capture):
            writer = _make_writer()
            result = writer.append_log(kind="INGEST", detail="added paper X", when="2026-04-12")

        self.assertIn("/pages", captured[0].full_url)
        body = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(body["parent"], {"database_id": "log-db"})
        self.assertEqual(body["properties"]["Kind"]["select"]["name"], "INGEST")
        self.assertEqual(
            body["properties"]["Detail"]["rich_text"][0]["text"]["content"],
            "added paper X",
        )
        self.assertEqual(body["properties"]["When"]["date"]["start"], "2026-04-12")
        self.assertEqual(result["id"], "log-1")


class TestRetryOn429(unittest.TestCase):
    def test_429_then_success_retries(self):
        success = {"id": "src-retry", "properties": {"Title": {"title": [{"plain_text": "p"}]}}}
        responses = [
            _make_response({"message": "rate limited"}, status=429),
            _make_response(success, status=200),
        ]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch("notion_client.time.sleep") as mock_sleep:
            with patch.object(_OPENER, "open", side_effect=side_effect):
                writer = _make_writer()
                result = writer.create_source(
                    url="https://example.com/retry",
                    title="p",
                    type_="paper",
                    summary="x",
                    ingested="2026-04-12",
                )
            mock_sleep.assert_called()
        self.assertEqual(result["id"], "src-retry")
        self.assertEqual(len(calls), 2)

    def test_429_exhausted_raises(self):
        always_429 = _make_response({"message": "rate limited"}, status=429)
        with patch("notion_client.time.sleep"):
            with patch.object(_OPENER, "open", return_value=always_429):
                writer = _make_writer()
                with self.assertRaises(RuntimeError) as ctx:
                    writer.create_source(
                        url="https://example.com/retry2",
                        title="p",
                        type_="paper",
                        summary="x",
                        ingested="2026-04-12",
                    )
        self.assertIn("429", str(ctx.exception))


class TestListAllConcepts(unittest.TestCase):
    def _make_concept_page(self, page_id: str, title: str, source_ids: list) -> dict:
        return {
            "id": page_id,
            "properties": {
                "Title": {"title": [{"plain_text": title}]},
                "Sources": {"relation": [{"id": sid} for sid in source_ids]},
            },
        }

    def _make_blocks_response(self, texts: list) -> dict:
        return {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": t}]
                    },
                }
                for t in texts
            ]
        }

    def test_single_page_returns_all_concepts(self):
        page1 = self._make_concept_page("con-1", "Speculative Decoding", ["src-a"])
        page2 = self._make_concept_page("con-2", "LLM Efficiency", ["src-b", "src-c"])

        query_response = {"results": [page1, page2], "has_more": False, "next_cursor": None}
        blocks_1 = self._make_blocks_response(["Speculative decoding speeds up inference."])
        blocks_2 = self._make_blocks_response(["Efficiency matters.", "Second paragraph."])

        responses = [
            _make_response(query_response),
            _make_response(blocks_1),
            _make_response(blocks_2),
        ]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            writer = _make_writer()
            result = writer.list_all_concepts()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "con-1")
        self.assertEqual(result[0]["title"], "Speculative Decoding")
        self.assertEqual(result[0]["source_ids"], ["src-a"])
        self.assertIn("Speculative decoding speeds up inference.", result[0]["body_markdown"])
        self.assertEqual(result[1]["id"], "con-2")
        self.assertEqual(result[1]["title"], "LLM Efficiency")
        self.assertEqual(result[1]["source_ids"], ["src-b", "src-c"])
        self.assertIn("Efficiency matters.", result[1]["body_markdown"])
        self.assertIn("/databases/con-db/query", calls[0].full_url)
        self.assertIn("/blocks/con-1/children", calls[1].full_url)
        self.assertIn("/blocks/con-2/children", calls[2].full_url)

    def test_follows_pagination_cursor(self):
        page1 = self._make_concept_page("con-1", "Alpha", [])
        page2 = self._make_concept_page("con-2", "Beta", [])

        query_page1 = {"results": [page1], "has_more": True, "next_cursor": "cursor-abc"}
        query_page2 = {"results": [page2], "has_more": False, "next_cursor": None}
        blocks_1 = self._make_blocks_response(["Alpha body."])
        blocks_2 = self._make_blocks_response(["Beta body."])

        responses = [
            _make_response(query_page1),
            _make_response(blocks_1),
            _make_response(query_page2),
            _make_response(blocks_2),
        ]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            writer = _make_writer()
            result = writer.list_all_concepts()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Alpha")
        self.assertEqual(result[1]["title"], "Beta")

        # First query has no cursor, second query must include next_cursor
        first_body = json.loads(calls[0].data.decode("utf-8"))
        self.assertNotIn("start_cursor", first_body)
        third_body = json.loads(calls[2].data.decode("utf-8"))
        self.assertEqual(third_body["start_cursor"], "cursor-abc")

    def test_raises_when_has_more_but_no_cursor(self):
        page1 = self._make_concept_page("con-1", "Alpha", [])
        query_response = {"results": [page1], "has_more": True, "next_cursor": None}
        blocks_1 = self._make_blocks_response([])

        responses = [
            _make_response(query_response),
            _make_response(blocks_1),
        ]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            writer = _make_writer()
            with self.assertRaises(RuntimeError) as ctx:
                writer.list_all_concepts()
        self.assertIn("next_cursor", str(ctx.exception))


class TestFetchConceptBodyMarkdown(unittest.TestCase):
    def test_joins_paragraph_blocks_into_markdown(self):
        blocks_response = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "First paragraph."}]
                    },
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "Second paragraph."}]
                    },
                },
            ]
        }
        with patch.object(_OPENER, "open", return_value=_make_response(blocks_response)):
            writer = _make_writer()
            result = writer._fetch_concept_body_markdown("page-abc")

        self.assertIn("First paragraph.", result)
        self.assertIn("Second paragraph.", result)
        self.assertEqual(result, "First paragraph.\n\nSecond paragraph.")

    def test_skips_non_paragraph_blocks(self):
        blocks_response = {
            "results": [
                {
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"plain_text": "A Heading"}]
                    },
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "Only this paragraph."}]
                    },
                },
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"plain_text": "A bullet."}]
                    },
                },
            ]
        }
        with patch.object(_OPENER, "open", return_value=_make_response(blocks_response)):
            writer = _make_writer()
            result = writer._fetch_concept_body_markdown("page-xyz")

        self.assertEqual(result, "Only this paragraph.")
        self.assertNotIn("A Heading", result)
        self.assertNotIn("A bullet.", result)


if __name__ == "__main__":
    unittest.main()
