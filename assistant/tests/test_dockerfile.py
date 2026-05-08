import unittest
from pathlib import Path


class TestDockerfileSynthesisBrief(unittest.TestCase):
    def test_copies_synthesis_brief_skill(self):
        path = Path(__file__).parent.parent / "Dockerfile"
        text = path.read_text(encoding="utf-8")
        expected = (
            "COPY --chown=hermes:hermes config/skills/synthesis-brief "
            "/home/hermes/.hermes/skills/synthesis-brief"
        )
        self.assertIn(expected, text, f"missing line in Dockerfile:\n{expected}")


if __name__ == "__main__":
    unittest.main()
