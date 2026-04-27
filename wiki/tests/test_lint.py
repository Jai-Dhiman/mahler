# mahler/wiki/tests/test_lint.py
import sys
sys.path.insert(0, 'scripts')

import json
import os
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import lint


def _concept(page_id: str, title: str, body: str = "", sources: list | None = None) -> dict:
    return {
        "id": page_id,
        "title": title,
        "body_markdown": body,
        "source_ids": sources or [],
    }


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestLintBrokenWikilinks(unittest.TestCase):
    def test_detects_broken_wikilink(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("p1", "Alpha", body="See also [[Missing Concept]]."),
            _concept("p2", "Beta"),
        ]
        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["lint"])
        output = out.getvalue()
        self.assertIn("Broken wikilink", output)
        self.assertIn("Missing Concept", output)
        self.assertIn("Alpha", output)


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestLintOrphans(unittest.TestCase):
    def test_detects_orphan_concept(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("p1", "Alpha", body="Linked to [[Beta]].", sources=["src-1"]),
            _concept("p2", "Beta", body="", sources=["src-2"]),
            _concept("p3", "Orphan", body="", sources=[]),
        ]
        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["lint"])
        output = out.getvalue()
        self.assertIn("Orphan concept", output)
        self.assertIn("Orphan", output)
        self.assertNotIn("Orphan concept: 'Alpha'", output)
        self.assertNotIn("Orphan concept: 'Beta'", output)


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestLintSourceless(unittest.TestCase):
    def test_detects_sourceless_concept(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("p1", "Alpha", body="See [[Sourceless]].", sources=["src-1"]),
            _concept("p2", "Sourceless", body="", sources=[]),
        ]
        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["lint"])
        output = out.getvalue()
        self.assertIn("Sourceless concept", output)
        self.assertIn("Sourceless", output)
        self.assertNotIn("Orphan concept: 'Sourceless'", output)


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestLintDuplicates(unittest.TestCase):
    def test_detects_case_insensitive_duplicates(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("p1", "Speculative Decoding", body="x", sources=["s1"]),
            _concept("p2", "speculative decoding", body="x", sources=["s2"]),
        ]
        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["lint"])
        output = out.getvalue()
        self.assertIn("Duplicate title", output)
        self.assertIn("Speculative Decoding", output)
        self.assertIn("p1", output)
        self.assertIn("p2", output)


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestLintLog(unittest.TestCase):
    def test_appends_log_with_counts(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("p1", "Alpha", body="[[Missing]]", sources=["s1"]),
            _concept("p2", "Orphan", body="", sources=[]),
        ]
        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO):
                lint.main(["lint"])
        fake_writer.append_log.assert_called_once()
        kwargs = fake_writer.append_log.call_args.kwargs
        self.assertEqual(kwargs["kind"], "LINT")
        self.assertIn("broken", kwargs["detail"])
        self.assertIn("orphan", kwargs["detail"])


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestDumpCommand(unittest.TestCase):
    def test_emits_concepts_with_source_bodies(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("con-1", "Agent Harnesses", body="About harnesses.", sources=["src-a", "src-b"]),
            _concept("con-2", "Skill Design", body="About skills.", sources=["src-b"]),
        ]

        def get_source_side_effect(sid):
            data = {
                "src-a": {"title": "Thin Harness", "body": "Thin body."},
                "src-b": {"title": "Fat Skills", "body": "Fat body."},
            }
            return data[sid]

        fake_writer.get_source.side_effect = get_source_side_effect

        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["dump"])

        data = json.loads(out.getvalue())
        self.assertEqual(len(data), 2)

        con1 = next(c for c in data if c["title"] == "Agent Harnesses")
        self.assertEqual(con1["body"], "About harnesses.")
        self.assertEqual(len(con1["sources"]), 2)
        source_titles = [s["title"] for s in con1["sources"]]
        self.assertIn("Thin Harness", source_titles)
        self.assertIn("Fat Skills", source_titles)
        self.assertIn("Agent Harnesses", con1["all_concept_titles"])
        self.assertIn("Skill Design", con1["all_concept_titles"])

        con2 = next(c for c in data if c["title"] == "Skill Design")
        self.assertEqual(len(con2["sources"]), 1)
        self.assertEqual(con2["sources"][0]["title"], "Fat Skills")

    def test_deduplicates_source_fetches(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("con-1", "Alpha", body=".", sources=["src-shared"]),
            _concept("con-2", "Beta", body=".", sources=["src-shared"]),
        ]
        fake_writer.get_source.return_value = {"title": "Shared Source", "body": "Shared body."}

        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO):
                lint.main(["dump"])

        self.assertEqual(fake_writer.get_source.call_count, 1)

    def test_concept_with_no_sources_emits_empty_list_without_fetching(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("con-1", "Empty Concept", body="No sources yet.", sources=[]),
        ]

        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["dump"])

        data = json.loads(out.getvalue())
        self.assertEqual(data[0]["sources"], [])
        fake_writer.get_source.assert_not_called()
