import unittest
from pathlib import Path


class TestEntrypointSynthesisBriefCron(unittest.TestCase):
    def test_registers_synthesis_brief_cron(self):
        text = (Path(__file__).parent.parent / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn("'synthesis-brief'", text)
        self.assertIn("synthesize.py --run", text)
        self.assertIn("'0 13 * * 1-5'", text)


if __name__ == "__main__":
    unittest.main()
