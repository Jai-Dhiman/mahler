import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct123",
    "CF_D1_DATABASE_ID": "db123",
    "CF_API_TOKEN": "cftoken",
}


def _patch_env():
    return patch.dict("os.environ", _BASE_ENV, clear=True)


class TestMigrate(unittest.TestCase):

    def _write_temp_map(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_inserts_file_content_into_d1_when_table_is_empty(self):
        tmp = self._write_temp_map("## URGENT\nTest priority map content.")
        mock_d1 = MagicMock()
        mock_d1.get_priority_map.side_effect = RuntimeError("priority_map table is empty")

        with (
            _patch_env(),
            patch("migrate.D1Client", return_value=mock_d1),
        ):
            import migrate
            migrate.main(["--file", tmp])

        mock_d1.set_priority_map.assert_called_once()
        inserted_content = mock_d1.set_priority_map.call_args[0][0]
        self.assertIn("URGENT", inserted_content)
        self.assertIn("Test priority map content", inserted_content)

    def test_raises_if_priority_map_already_exists_in_d1(self):
        tmp = self._write_temp_map("## URGENT\nContent.")
        mock_d1 = MagicMock()
        mock_d1.get_priority_map.return_value = "existing content"

        with (
            _patch_env(),
            patch("migrate.D1Client", return_value=mock_d1),
        ):
            import migrate
            with self.assertRaises(RuntimeError) as ctx:
                migrate.main(["--file", tmp])
        self.assertIn("already seeded", str(ctx.exception))
        mock_d1.set_priority_map.assert_not_called()


if __name__ == "__main__":
    unittest.main()
