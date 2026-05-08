import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

import inputs


class TestLoadAllEmpty(unittest.TestCase):
    def test_returns_empty_bundle_and_creates_tables(self):
        d1 = MagicMock()
        d1.query.return_value = []
        honcho = MagicMock()
        honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        self.assertEqual(bundle.recent_items, [])
        self.assertEqual(bundle.context_items, [])
        self.assertEqual(bundle.past_briefs, [])
        self.assertEqual(bundle.identifiers, set())

        executed_sqls = [call.args[0] for call in d1.query.call_args_list]
        joined = "\n".join(executed_sqls)
        self.assertIn("CREATE TABLE IF NOT EXISTS local_capture", joined)
        self.assertIn("CREATE TABLE IF NOT EXISTS synthesis_brief", joined)


if __name__ == "__main__":
    unittest.main()
