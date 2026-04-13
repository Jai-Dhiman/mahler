# mahler/wiki/tests/test_ingest.py
import sys
sys.path.insert(0, 'scripts')

import io
import os
import tempfile
import unittest
from unittest.mock import patch

import ingest


class TestIngestEnvLoader(unittest.TestCase):
    def test_load_env_from_dotenv_populates_missing_keys(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# comment\n")
            f.write("NOTION_WIKI_WRITE_TOKEN=abc123\n")
            f.write("NOTION_WIKI_SOURCES_DB_ID=src\n")
            path = f.name
        try:
            with patch.dict(os.environ, {}, clear=True):
                ingest._load_env_from_dotenv(path)
                self.assertEqual(os.environ["NOTION_WIKI_WRITE_TOKEN"], "abc123")
                self.assertEqual(os.environ["NOTION_WIKI_SOURCES_DB_ID"], "src")
        finally:
            os.unlink(path)

    def test_load_env_does_not_override_existing_env(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("NOTION_WIKI_WRITE_TOKEN=from-file\n")
            path = f.name
        try:
            with patch.dict(os.environ, {"NOTION_WIKI_WRITE_TOKEN": "from-env"}, clear=True):
                ingest._load_env_from_dotenv(path)
                self.assertEqual(os.environ["NOTION_WIKI_WRITE_TOKEN"], "from-env")
        finally:
            os.unlink(path)

    def test_missing_token_raises_runtime_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                ingest._get_writer()
            self.assertIn("NOTION_WIKI_WRITE_TOKEN", str(ctx.exception))


from io import StringIO
from unittest.mock import MagicMock


def _summary_tmpfile(text: str = "A summary.") -> str:
    tf = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    tf.write(text)
    tf.close()
    return tf.name


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestIngestSkipDuplicate(unittest.TestCase):
    def test_skips_when_url_already_ingested(self):
        existing = {"id": "existing-src", "properties": {}}
        fake_writer = MagicMock()
        fake_writer.find_source_by_url.return_value = existing
        summary_path = _summary_tmpfile()
        try:
            with patch("ingest.NotionWikiWriter", return_value=fake_writer):
                with patch("sys.stdout", new_callable=StringIO) as out:
                    ingest.main([
                        "ingest",
                        "--url", "https://example.com/dup",
                        "--title", "dup",
                        "--type", "paper",
                        "--summary-file", summary_path,
                    ])
            fake_writer.find_source_by_url.assert_called_once_with("https://example.com/dup")
            fake_writer.create_source.assert_not_called()
            self.assertIn("Already ingested", out.getvalue())
            self.assertIn("existing-src", out.getvalue())
        finally:
            os.unlink(summary_path)


@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestIngestCreate(unittest.TestCase):
    def test_creates_source_for_new_url(self):
        fake_writer = MagicMock()
        fake_writer.find_source_by_url.return_value = None
        fake_writer.create_source.return_value = {"id": "new-src-id"}
        summary_path = _summary_tmpfile("Paragraph one.\n\nParagraph two.")
        try:
            with patch("ingest.NotionWikiWriter", return_value=fake_writer):
                with patch("sys.stdout", new_callable=StringIO) as out:
                    ingest.main([
                        "ingest",
                        "--url", "https://example.com/new",
                        "--title", "New Paper",
                        "--type", "paper",
                        "--summary-file", summary_path,
                        "--ingested", "2026-04-12",
                    ])
            fake_writer.create_source.assert_called_once()
            kwargs = fake_writer.create_source.call_args.kwargs
            self.assertEqual(kwargs["url"], "https://example.com/new")
            self.assertEqual(kwargs["title"], "New Paper")
            self.assertEqual(kwargs["type_"], "paper")
            self.assertEqual(kwargs["summary"], "Paragraph one.\n\nParagraph two.")
            self.assertEqual(kwargs["ingested"], "2026-04-12")
            self.assertIn("new-src-id", out.getvalue())
        finally:
            os.unlink(summary_path)


if __name__ == "__main__":
    unittest.main()
