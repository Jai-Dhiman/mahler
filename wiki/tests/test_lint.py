# mahler/wiki/tests/test_lint.py
import sys
sys.path.insert(0, 'scripts')

import os
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import lint


def _concept(page_id: str, title: str, body: str = "", sources: list = None) -> dict:
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
