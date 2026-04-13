# assistant/config/skills/notion-wiki/tests/test_wiki.py
import sys
sys.path.insert(0, 'scripts')

import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import wiki


class TestWikiEnvLoader(unittest.TestCase):
    def test_missing_token_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                wiki._get_reader()
            self.assertIn("NOTION_WIKI_READ_TOKEN", str(ctx.exception))

    def test_supplement_env_reads_hermes_dotenv(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("NOTION_WIKI_READ_TOKEN=tok-from-hermes\n")
            f.write("NOTION_WIKI_SOURCES_DB_ID=src-from-hermes\n")
            path = f.name
        try:
            with patch.dict(os.environ, {}, clear=True):
                wiki._load_env_file(path)
                self.assertEqual(os.environ["NOTION_WIKI_READ_TOKEN"], "tok-from-hermes")
                self.assertEqual(os.environ["NOTION_WIKI_SOURCES_DB_ID"], "src-from-hermes")
        finally:
            os.unlink(path)


@patch.dict(os.environ, {
    "NOTION_WIKI_READ_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
})
class TestWikiSearch(unittest.TestCase):
    def test_search_dispatches_and_prints_results(self):
        fake_reader = MagicMock()
        fake_reader.search.return_value = [
            {"id": "con-1", "title": "Alpha", "db": "concepts"},
            {"id": "src-1", "title": "Paper X", "db": "sources"},
        ]
        with patch("wiki.NotionWikiReader", return_value=fake_reader):
            with patch("sys.stdout", new_callable=StringIO) as out:
                wiki.main(["search", "--query", "alpha", "--limit", "3"])
        fake_reader.search.assert_called_once_with("alpha", limit=3)
        output = out.getvalue()
        self.assertIn("[con-1] (concepts) Alpha", output)
        self.assertIn("[src-1] (sources) Paper X", output)


if __name__ == "__main__":
    unittest.main()
