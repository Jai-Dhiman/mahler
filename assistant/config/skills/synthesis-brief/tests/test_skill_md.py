import unittest
from pathlib import Path


class TestSkillManifest(unittest.TestCase):
    def test_skill_md_has_required_frontmatter(self):
        path = Path(__file__).parent.parent / "SKILL.md"
        self.assertTrue(path.exists(), f"missing: {path}")
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"), "missing YAML frontmatter")
        end = text.find("\n---\n", 4)
        self.assertGreater(end, 0, "unterminated YAML frontmatter")
        front = text[4:end]
        self.assertIn("name: synthesis-brief", front)
        self.assertIn("description:", front)
        self.assertIn("version:", front)
        self.assertIn("author:", front)
        self.assertIn("license:", front)
        self.assertIn("metadata:", front)
        self.assertIn("hermes:", front)
        self.assertIn("tags:", front)


if __name__ == "__main__":
    unittest.main()
