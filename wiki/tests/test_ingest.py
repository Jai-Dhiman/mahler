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


if __name__ == "__main__":
    unittest.main()
