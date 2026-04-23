import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestEmptyQueue(unittest.TestCase):
    def test_empty_queue_prints_no_work_and_returns_zero(self):
        import orchestrate
        d1 = MagicMock()
        d1.fetch_pending.return_value = []
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = orchestrate.main(
                argv=[],
                d1_client=d1,
                runner=MagicMock(),
                llm_caller=MagicMock(),
                discord_poster=MagicMock(),
            )
        self.assertEqual(rc, 0)
        self.assertIn("NO_WORK", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
