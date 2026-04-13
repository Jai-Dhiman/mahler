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


if __name__ == "__main__":
    unittest.main()
