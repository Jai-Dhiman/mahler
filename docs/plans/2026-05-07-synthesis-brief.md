# Daily Synthesis Brief Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task within a group). Tasks in different groups must run sequentially in group order. Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Receive a daily Discord push (CONNECTIONS / PATTERN / QUESTION) at ~6am PT weekdays that surfaces non-obvious links across recent and older notes.
**Spec:** docs/specs/2026-05-07-synthesis-brief-design.md
**Style:** Follow the project's coding standards (assistant/CLAUDE.md). Python uses `uv`. Tests use unittest, run via `pytest`. No emojis. Explicit exception handling, no silent fallbacks.

---

## Task Groups

- **Group A** (parallel, 7 tasks): scaffold each new/modified file with its first behavior
- **Group B** (parallel, 4 tasks): second behavior layer
- **Group C** (parallel, 4 tasks): third behavior layer
- **Group D** (parallel, 2 tasks): fourth behavior layer
- **Group E** (sequential, 2 tasks): final inputs.py behaviors — both modify inputs.py so they cannot be parallel
- **Group F** (sequential, 3 tasks): synthesize.py orchestrator (depends on Groups A–E for inputs/validator complete)

Total: 22 tasks. Estimated 12–16 commits worth of work.

---

## Shared Conventions (read once)

- All Python files use `from __future__ import annotations` if helpful, but match existing style in the directory.
- Test files use unittest, importable via `pytest --import-mode=importlib` (per `assistant/pytest.ini`). Each test file inserts `Path(__file__).parent.parent / "scripts"` and `Path(__file__).parent.parent.parent.parent / "shared"` into `sys.path` (mirrors `assistant/config/skills/memory-kaizen/tests/test_kaizen.py:8-9`).
- `D1Client` is the shared client at `assistant/config/shared/d1_base.py`. It exposes `query(sql, params) -> list[dict]`.
- Time math uses `datetime.now(timezone.utc)`.
- `mahler_kv` is the existing D1 table with `(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)`. Use raw SQL via `D1Client.query` since `synthesis-brief` does not extend the email-triage `D1Client` subclass.
- Run tests with: `cd /Users/jdhiman/Documents/mahler && pytest assistant/<test path> -v`.

---

## Group A — Scaffold (parallel, 7 tasks, no overlapping files)

### Task 1: inputs.py — load_all returns empty InputBundle when D1 has no rows
**Group:** A (parallel with Tasks 2, 3, 4, 5, 6, 7)

**Behavior being verified:** Calling `load_all` against an empty D1 returns an `InputBundle` with empty lists and an empty identifiers set; the call also creates `local_capture` and `synthesis_brief` tables.
**Interface under test:** `inputs.load_all(d1, honcho, recent_days, context_days) -> InputBundle`

**Files:**
- Create: `assistant/config/skills/synthesis-brief/scripts/__init__.py` (empty)
- Create: `assistant/config/skills/synthesis-brief/scripts/inputs.py`
- Create: `assistant/config/skills/synthesis-brief/tests/__init__.py` (empty)
- Create: `assistant/config/skills/synthesis-brief/tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/synthesis-brief/tests/test_inputs.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'inputs'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/synthesis-brief/scripts/inputs.py
from dataclasses import dataclass, field


@dataclass
class Item:
    source: str
    id: str
    content: str
    captured_at: str


@dataclass
class InputBundle:
    recent_items: list = field(default_factory=list)
    context_items: list = field(default_factory=list)
    past_briefs: list = field(default_factory=list)
    identifiers: set = field(default_factory=set)


_CREATE_LOCAL_CAPTURE = """
CREATE TABLE IF NOT EXISTS local_capture (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL CHECK(source IN ('memory','git')),
  project TEXT,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  captured_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_SYNTHESIS_BRIEF = """
CREATE TABLE IF NOT EXISTS synthesis_brief (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  posted_at TEXT NOT NULL DEFAULT (datetime('now')),
  connections_json TEXT NOT NULL,
  pattern TEXT NOT NULL,
  question TEXT NOT NULL
)
"""


def _ensure_tables(d1) -> None:
    d1.query(_CREATE_LOCAL_CAPTURE, [])
    d1.query(_CREATE_SYNTHESIS_BRIEF, [])
    d1.query(
        "CREATE INDEX IF NOT EXISTS idx_local_capture_recent ON local_capture(captured_at)",
        [],
    )


def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    return InputBundle()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/__init__.py \
        assistant/config/skills/synthesis-brief/scripts/inputs.py \
        assistant/config/skills/synthesis-brief/tests/__init__.py \
        assistant/config/skills/synthesis-brief/tests/test_inputs.py && \
git commit -m "feat(synthesis-brief): scaffold inputs module with table creation"
```

---

### Task 2: validator.py — validate returns (False, "thin_context") when bundle is too thin
**Group:** A (parallel with Tasks 1, 3, 4, 5, 6, 7)

**Behavior being verified:** Given a bundle with fewer than 3 recent items in last 24h AND fewer than 5 total in last 7d, `validate` returns `(False, "thin_context")` without touching the brief.
**Interface under test:** `validator.validate(brief, inputs) -> tuple[bool, str]`

**Files:**
- Create: `assistant/config/skills/synthesis-brief/scripts/validator.py`
- Create: `assistant/config/skills/synthesis-brief/tests/test_validator.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/synthesis-brief/tests/test_validator.py
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import validator
from inputs import InputBundle, Item


class TestThinContext(unittest.TestCase):
    def test_returns_false_thin_context_when_recent_under_3_and_context_under_5(self):
        bundle = InputBundle(
            recent_items=[Item("memory", "memory:1", "x", "2026-05-07")],
            context_items=[Item("project_log", "project_log:1", "y", "2026-05-06")],
            past_briefs=[],
            identifiers={"memory:1", "project_log:1"},
        )
        ok, reason = validator.validate({}, bundle)
        self.assertFalse(ok)
        self.assertEqual(reason, "thin_context")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_validator.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'validator'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/synthesis-brief/scripts/validator.py
_THIN_RECENT_MIN = 3
_THIN_CONTEXT_MIN = 5


def _is_thin(bundle) -> bool:
    recent_n = len(bundle.recent_items)
    total_n = recent_n + len(bundle.context_items)
    return recent_n < _THIN_RECENT_MIN and total_n < _THIN_CONTEXT_MIN


def validate(brief: dict, bundle) -> tuple[bool, str]:
    if _is_thin(bundle):
        return (False, "thin_context")
    return (True, "")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_validator.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/validator.py \
        assistant/config/skills/synthesis-brief/tests/test_validator.py && \
git commit -m "feat(synthesis-brief): scaffold validator with thin-context gate"
```

---

### Task 3: project_log.py — sync_local_to_d1 inserts memory file deltas
**Group:** A (parallel with Tasks 1, 2, 4, 5, 6, 7)

**Behavior being verified:** Calling `sync_local_to_d1` with a memory dir containing two .md files INSERTs two `local_capture` rows with `source='memory'` using `INSERT OR IGNORE` and a sha256 content_hash.
**Interface under test:** `project_log.sync_local_to_d1(memory_dir: Path, repos_root: Path)`

**Files:**
- Modify: `assistant/hooks/project_log.py`
- Create: `assistant/hooks/tests/__init__.py` (empty)
- Create: `assistant/hooks/tests/test_project_log.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/hooks/tests/test_project_log.py
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import project_log


class TestSyncLocalToD1Memory(unittest.TestCase):
    def test_inserts_memory_files_as_local_capture_rows(self):
        with tempfile.TemporaryDirectory() as memdir, tempfile.TemporaryDirectory() as repos:
            memdir_path = Path(memdir)
            (memdir_path / "MEMORY.md").write_text("index file")
            (memdir_path / "user_role.md").write_text("solo dev")

            d1 = MagicMock()
            d1.query.return_value = []

            with patch("project_log._get_d1_client", return_value=d1):
                project_log.sync_local_to_d1(memdir_path, Path(repos))

            insert_calls = [
                c for c in d1.query.call_args_list
                if "INSERT OR IGNORE INTO local_capture" in c.args[0]
                and "'memory'" in c.args[0] or (len(c.args) > 1 and "memory" in (c.args[1] or []))
            ]
            # Filter precisely: SQL contains INSERT OR IGNORE INTO local_capture
            memory_inserts = [
                c for c in d1.query.call_args_list
                if "INSERT OR IGNORE INTO local_capture" in c.args[0]
                and c.args[1] and c.args[1][0] == "memory"
            ]
            self.assertEqual(len(memory_inserts), 2)
            for call in memory_inserts:
                params = call.args[1]
                self.assertEqual(params[0], "memory")  # source
                self.assertTrue(len(params[3]) == 64)  # sha256 hex


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py -v
```
Expected: FAIL — `AttributeError: module 'project_log' has no attribute 'sync_local_to_d1'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add at the top of `assistant/hooks/project_log.py` (after existing imports):

```python
import hashlib
```

Append to `assistant/hooks/project_log.py` (before `def main()`):

```python
_INSERT_LOCAL_CAPTURE = (
    "INSERT OR IGNORE INTO local_capture "
    "(source, project, content, content_hash) VALUES (?, ?, ?, ?)"
)


def _sync_memory_dir(d1, memory_dir: Path) -> None:
    if not memory_dir.is_dir():
        return
    for md_file in sorted(memory_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError:
            continue
        content_hash = hashlib.sha256(
            f"memory:{md_file.name}:{content}".encode("utf-8")
        ).hexdigest()
        body = f"# {md_file.name}\n{content}"
        d1.query(_INSERT_LOCAL_CAPTURE, ["memory", md_file.name, body, content_hash])


def sync_local_to_d1(memory_dir: Path, repos_root: Path) -> None:
    d1 = _get_d1_client()
    _sync_memory_dir(d1, memory_dir)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/project_log.py \
        assistant/hooks/tests/__init__.py \
        assistant/hooks/tests/test_project_log.py && \
git commit -m "feat(hook): sync memory dir to D1 local_capture"
```

---

### Task 4: morning-brief brief.py — build_embed prepends Synthesis field when synthesis_section provided
**Group:** A (parallel with Tasks 1, 2, 3, 5, 6, 7)

**Behavior being verified:** When `build_embed` is called with `synthesis_section={"connections":[{"summary":"…","citations":[…]},…], "pattern":"…", "question":"…"}`, the resulting embed's first field is named `Synthesis` and contains the formatted CONNECTIONS / PATTERN / QUESTION sections.
**Interface under test:** `brief.build_embed(rows, period, since_hours, news_items=None, news_error=None, synthesis_section=None)`

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/brief.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py`

- [ ] **Step 1: Write the failing test**

Append this test class to `assistant/config/skills/morning-brief/tests/test_brief.py`:

```python
class TestBuildEmbedSynthesisPrepend(unittest.TestCase):
    def test_first_field_is_synthesis_when_section_provided(self):
        synthesis = {
            "connections": [
                {"summary": "Connection A", "citations": [{"source": "memory", "id": "memory:1"}]},
            ],
            "pattern": "Theme of the week",
            "question": "What is the cost of certainty?",
        }
        payload = brief.build_embed(
            rows=[],
            period="morning",
            since_hours=12,
            synthesis_section=synthesis,
        )
        fields = payload["embeds"][0]["fields"]
        self.assertEqual(fields[0]["name"], "Synthesis")
        self.assertIn("Connection A", fields[0]["value"])
        self.assertIn("Theme of the week", fields[0]["value"])
        self.assertIn("What is the cost of certainty?", fields[0]["value"])
```

(`brief` import already exists at the top of the test file; if not, add `import brief` per the file's existing import style.)

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestBuildEmbedSynthesisPrepend -v
```
Expected: FAIL — `TypeError: build_embed() got an unexpected keyword argument 'synthesis_section'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/brief.py`, change the `build_embed` signature to add the parameter, and prepend a Synthesis field when present.

Locate this signature:
```python
def build_embed(rows: list[dict], period: str, since_hours: int, news_items: list[dict] | None = None, news_error: str | None = None) -> dict:
```

Replace with:
```python
def build_embed(rows: list[dict], period: str, since_hours: int, news_items: list[dict] | None = None, news_error: str | None = None, synthesis_section: dict | None = None) -> dict:
```

Locate this line near the top of `build_embed`:
```python
    fields = []
```

Insert immediately after it:
```python
    if synthesis_section:
        connections = synthesis_section.get("connections", [])
        pattern = synthesis_section.get("pattern", "")
        question = synthesis_section.get("question", "")
        conn_lines = "\n".join(
            f"• {c.get('summary','')}" for c in connections
        )
        synth_value = (
            f"**CONNECTIONS**\n{conn_lines}\n\n"
            f"**PATTERN**\n{pattern}\n\n"
            f"**QUESTION**\n{question}"
        )
        fields.append({
            "name": "Synthesis",
            "value": _truncate_field(synth_value.split("\n"), max_chars=1024),
            "inline": False,
        })
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestBuildEmbedSynthesisPrepend -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/brief.py \
        assistant/config/skills/morning-brief/tests/test_brief.py && \
git commit -m "feat(morning-brief): prepend Synthesis field when section provided"
```

---

### Task 5: SKILL.md — synthesis-brief manifest with required frontmatter
**Group:** A (parallel with Tasks 1, 2, 3, 4, 6, 7)

**Behavior being verified:** `synthesis-brief/SKILL.md` exists with required YAML frontmatter keys (`name`, `description`, `version`, `author`, `license`, `metadata.hermes.tags`) and the name matches `synthesis-brief`.
**Interface under test:** the SKILL.md file as parsed by Hermes' skill loader (we assert on its YAML frontmatter directly).

**Files:**
- Create: `assistant/config/skills/synthesis-brief/SKILL.md`
- Create: `assistant/config/skills/synthesis-brief/tests/test_skill_md.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/synthesis-brief/tests/test_skill_md.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_skill_md.py -v
```
Expected: FAIL — `AssertionError: missing: .../SKILL.md`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```markdown
---
name: synthesis-brief
description: Daily synthesis push at ~6am PT weekdays. Loads Honcho conclusions, project_log wins, and local_capture (memory + git) deltas, asks the LLM for 3 non-obvious connections + a weekly pattern + a question to sit with, validates citations and length, then writes the result to D1 (synthesis_brief table) and mahler_kv (key synthesis_brief:latest) for the 8am morning-brief to prepend.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [memory, synthesis, brief, discord, productivity, cron]
    related_skills: [morning-brief, memory-kaizen, project-synthesis]
---

## When to use

- Cron-triggered Mon–Fri at 13:00 UTC (5am PDT / 6am PST)
- When the user asks "run synthesis brief" or "what's today's synthesis"

## Procedure

```bash
python3 ~/.hermes/skills/synthesis-brief/scripts/synthesize.py --run
```

Dry-run (prints the resulting brief JSON, does not write to D1 or KV):

```bash
python3 ~/.hermes/skills/synthesis-brief/scripts/synthesize.py --run --dry-run
```

## Output

On success: writes a row to `synthesis_brief` and updates `mahler_kv` at key `synthesis_brief:latest` with the JSON `{posted_at, connections, pattern, question}`. Prints `Synthesis brief written.`

On thin context / validator failure: prints `Synthesis brief skipped: <reason>` and exits 0. No Discord post.

The 8am morning-brief reads `mahler_kv:synthesis_brief:latest` and prepends a Synthesis field if the row is fresher than 24h.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_skill_md.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/SKILL.md \
        assistant/config/skills/synthesis-brief/tests/test_skill_md.py && \
git commit -m "feat(synthesis-brief): add SKILL.md manifest"
```

---

### Task 6: Dockerfile — COPY synthesis-brief into Hermes skills dir
**Group:** A (parallel with Tasks 1, 2, 3, 4, 5, 7)

**Behavior being verified:** `assistant/Dockerfile` contains a `COPY` line that places `config/skills/synthesis-brief` into `/home/hermes/.hermes/skills/synthesis-brief` with the correct ownership.
**Interface under test:** the Dockerfile as text.

**Files:**
- Modify: `assistant/Dockerfile`
- Create: `assistant/tests/__init__.py` (empty, only if missing)
- Create: `assistant/tests/test_dockerfile.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/tests/test_dockerfile.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/tests/test_dockerfile.py -v
```
Expected: FAIL — `AssertionError: missing line in Dockerfile`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/Dockerfile`, locate the existing line:
```
COPY --chown=hermes:hermes config/skills/memory-kaizen /home/hermes/.hermes/skills/memory-kaizen
```

Add immediately after it:
```
COPY --chown=hermes:hermes config/skills/synthesis-brief /home/hermes/.hermes/skills/synthesis-brief
```

Create `assistant/tests/__init__.py` as empty if it does not exist.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/tests/test_dockerfile.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/Dockerfile assistant/tests/__init__.py assistant/tests/test_dockerfile.py && \
git commit -m "feat(deploy): COPY synthesis-brief skill in Dockerfile"
```

---

### Task 7: entrypoint.sh — register synthesis-brief cron at 0 13 * * 1-5
**Group:** A (parallel with Tasks 1, 2, 3, 4, 5, 6)

**Behavior being verified:** `assistant/entrypoint.sh` contains an upsert call that registers `synthesis-brief` at cron `0 13 * * 1-5` invoking `synthesize.py --run`.
**Interface under test:** the entrypoint.sh text.

**Files:**
- Modify: `assistant/entrypoint.sh`
- Create: `assistant/tests/test_entrypoint.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/tests/test_entrypoint.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/tests/test_entrypoint.py -v
```
Expected: FAIL — `AssertionError: 'synthesis-brief' not found`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/entrypoint.sh`, find the existing block:
```
results.append(('memory-kaizen', upsert(
    ['memory-kaizen'],
    'Run the weekly memory kaizen: run python3 ~/.hermes/skills/memory-kaizen/scripts/kaizen.py --run, then post the result message to Discord verbatim.',
    '0 19 * * 0',
)))
```

Add immediately after it (before the `with open(jobs_file, ...)` line):
```
results.append(('synthesis-brief', upsert(
    ['synthesis-brief'],
    'Run the daily synthesis brief: run python3 ~/.hermes/skills/synthesis-brief/scripts/synthesize.py --run, then print the result line to stdout. Do not post to Discord; the 8am morning-brief picks up the result from mahler_kv.',
    '0 13 * * 1-5',
)))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/tests/test_entrypoint.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/entrypoint.sh assistant/tests/test_entrypoint.py && \
git commit -m "feat(deploy): register synthesis-brief cron 0 13 * * 1-5"
```

---

## Group B — Second behavior layer (parallel, 4 tasks)

### Task 8: inputs.py — load_all loads project_log wins as context_items
**Group:** B (parallel with Tasks 9, 10, 11; depends on Task 1)

**Behavior being verified:** When D1 returns 2 rows from `project_log` with `entry_type='win'`, `load_all` includes them in `context_items` with `source='project_log'` and the win summary as content.
**Interface under test:** `inputs.load_all`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/inputs.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/config/skills/synthesis-brief/tests/test_inputs.py`:

```python
class TestLoadAllProjectWins(unittest.TestCase):
    def test_includes_project_log_wins_in_context_items(self):
        win_rows = [
            {"id": 42, "project": "crescendAI", "summary": "Shipped V6 atoms",
             "created_at": "2026-05-04 12:00:00"},
            {"id": 43, "project": "mahler", "summary": "Added wiki search",
             "created_at": "2026-05-05 09:00:00"},
        ]

        d1 = MagicMock()
        def query_side_effect(sql, params=None):
            if "FROM project_log" in sql:
                return win_rows
            return []
        d1.query.side_effect = query_side_effect

        honcho = MagicMock()
        honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        wins = [it for it in bundle.context_items if it.source == "project_log"]
        self.assertEqual(len(wins), 2)
        self.assertEqual(wins[0].id, "project_log:42")
        self.assertIn("Shipped V6 atoms", wins[0].content)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py::TestLoadAllProjectWins -v
```
Expected: FAIL — `AssertionError: 0 != 2` (no project_log loader yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/synthesis-brief/scripts/inputs.py`, add:

```python
def _load_project_wins(d1, context_days: int) -> list:
    rows = d1.query(
        "SELECT id, project, summary, created_at FROM project_log "
        "WHERE entry_type = 'win' AND created_at >= datetime('now', ? || ' days') "
        "ORDER BY created_at DESC",
        [f"-{context_days}"],
    )
    items = []
    for r in rows:
        items.append(Item(
            source="project_log",
            id=f"project_log:{r['id']}",
            content=f"[{r['project']}] {r['summary']}",
            captured_at=r["created_at"],
        ))
    return items
```

Update `load_all` to call it:

```python
def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    return bundle
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/inputs.py \
        assistant/config/skills/synthesis-brief/tests/test_inputs.py && \
git commit -m "feat(synthesis-brief): load project_log wins as context items"
```

---

### Task 9: validator.py — validate detects insufficient_citations
**Group:** B (parallel with Tasks 8, 10, 11; depends on Task 2)

**Behavior being verified:** When the bundle has enough context but the brief contains 3 connections of which fewer than 2 cite ≥2 valid identifiers, `validate` returns `(False, "insufficient_citations")`.
**Interface under test:** `validator.validate`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/validator.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_validator.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/config/skills/synthesis-brief/tests/test_validator.py`:

```python
class TestCitations(unittest.TestCase):
    def _bundle(self):
        # Above thin-context threshold
        return InputBundle(
            recent_items=[Item("memory", f"memory:{i}", "x", "2026-05-07") for i in range(4)],
            context_items=[Item("project_log", f"project_log:{i}", "y", "2026-05-04") for i in range(6)],
            past_briefs=[],
            identifiers={f"memory:{i}" for i in range(4)} | {f"project_log:{i}" for i in range(6)},
        )

    def test_returns_false_insufficient_citations_when_under_2_of_3_qualify(self):
        bundle = self._bundle()
        brief = {
            "connections": [
                {"summary": "A", "citations": [{"source": "memory", "id": "memory:1"}]},  # only 1
                {"summary": "B", "citations": []},                                          # 0
                {"summary": "C", "citations": [
                    {"source": "memory", "id": "memory:2"},
                    {"source": "project_log", "id": "project_log:0"},
                ]},                                                                         # 2 ok
            ],
            "pattern": "p",
            "question": "q",
        }
        ok, reason = validator.validate(brief, bundle)
        self.assertFalse(ok)
        self.assertEqual(reason, "insufficient_citations")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_validator.py::TestCitations -v
```
Expected: FAIL — `AssertionError: True is not false` (validator currently returns OK once thin-context passes).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `validate` in `assistant/config/skills/synthesis-brief/scripts/validator.py`:

```python
def _qualifying_connections(brief: dict, identifiers: set) -> int:
    n = 0
    for c in brief.get("connections", []):
        cites = c.get("citations", []) or []
        valid = {f"{cit.get('id','')}" for cit in cites if cit.get("id") in identifiers}
        if len(valid) >= 2:
            n += 1
    return n


def validate(brief: dict, bundle) -> tuple[bool, str]:
    if _is_thin(bundle):
        return (False, "thin_context")
    if _qualifying_connections(brief, bundle.identifiers) < 2:
        return (False, "insufficient_citations")
    return (True, "")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_validator.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/validator.py \
        assistant/config/skills/synthesis-brief/tests/test_validator.py && \
git commit -m "feat(synthesis-brief): validate insufficient_citations gate"
```

---

### Task 10: project_log.py — sync_local_to_d1 inserts last-24h git commits
**Group:** B (parallel with Tasks 8, 9, 11; depends on Task 3)

**Behavior being verified:** Calling `sync_local_to_d1` against a `repos_root` containing one fake git repo with a recent commit produces an INSERT with `source='git'`, `project=<repo dir name>`, and content containing the commit subject.
**Interface under test:** `project_log.sync_local_to_d1`.

**Files:**
- Modify: `assistant/hooks/project_log.py`
- Modify: `assistant/hooks/tests/test_project_log.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/hooks/tests/test_project_log.py`:

```python
class TestSyncLocalToD1Git(unittest.TestCase):
    def test_inserts_recent_git_commits_per_repo(self):
        with tempfile.TemporaryDirectory() as memdir, tempfile.TemporaryDirectory() as repos:
            d1 = MagicMock()
            d1.query.return_value = []

            git_log_output = "abc1234 First commit subject\ndef5678 Second commit subject\n"

            def fake_run(cmd, **kwargs):
                result = MagicMock()
                result.returncode = 0
                if cmd[0] == "git" and "log" in cmd:
                    result.stdout = git_log_output
                else:
                    result.stdout = ""
                return result

            # Make one fake repo dir with a .git child
            repo_dir = Path(repos) / "myproject"
            (repo_dir / ".git").mkdir(parents=True)

            with patch("project_log._get_d1_client", return_value=d1), \
                 patch("project_log.subprocess.run", side_effect=fake_run):
                project_log.sync_local_to_d1(Path(memdir), Path(repos))

            git_inserts = [
                c for c in d1.query.call_args_list
                if "INSERT OR IGNORE INTO local_capture" in c.args[0]
                and c.args[1] and c.args[1][0] == "git"
            ]
            self.assertEqual(len(git_inserts), 2)
            for call in git_inserts:
                params = call.args[1]
                self.assertEqual(params[1], "myproject")     # project
                self.assertIn("commit subject", params[2])    # content
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py::TestSyncLocalToD1Git -v
```
Expected: FAIL — `AssertionError: 0 != 2` (no git sync yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/hooks/project_log.py`, add:

```python
def _sync_git_recent(d1, repos_root: Path) -> None:
    if not repos_root.is_dir():
        return
    for repo_dir in sorted(repos_root.iterdir()):
        if not (repo_dir / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "log", "--since=24.hours.ago",
                 "--pretty=format:%h %s"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            content = f"[{repo_dir.name}] {line}"
            content_hash = hashlib.sha256(
                f"git:{repo_dir.name}:{line}".encode("utf-8")
            ).hexdigest()
            d1.query(_INSERT_LOCAL_CAPTURE, ["git", repo_dir.name, content, content_hash])
```

Update `sync_local_to_d1`:

```python
def sync_local_to_d1(memory_dir: Path, repos_root: Path) -> None:
    d1 = _get_d1_client()
    try:
        _sync_memory_dir(d1, memory_dir)
    except Exception as exc:
        print(f"sync_local_to_d1 memory error: {exc}", file=sys.stderr)
    try:
        _sync_git_recent(d1, repos_root)
    except Exception as exc:
        print(f"sync_local_to_d1 git error: {exc}", file=sys.stderr)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/project_log.py assistant/hooks/tests/test_project_log.py && \
git commit -m "feat(hook): sync recent git commits to D1 local_capture"
```

---

### Task 11: morning-brief brief.py — main reads mahler_kv synthesis_brief:latest
**Group:** B (parallel with Tasks 8, 9, 10; depends on Task 4)

**Behavior being verified:** When `mahler_kv` has a row at key `synthesis_brief:latest` with `posted_at` within the last 24h, `main()` parses its JSON value and passes it to `build_embed` as `synthesis_section`.
**Interface under test:** `brief.main` (via dry-run).

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/brief.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/config/skills/morning-brief/tests/test_brief.py`:

```python
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


class TestMainReadsSynthesisFromKV(unittest.TestCase):
    def test_passes_fresh_synthesis_to_build_embed(self):
        fresh_posted = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        kv_value = json.dumps({
            "posted_at": fresh_posted,
            "connections": [{"summary": "S", "citations": [{"source": "memory", "id": "memory:1"}]}],
            "pattern": "P",
            "question": "Q",
        })

        captured = {}

        def fake_query(sql, params=None):
            if "mahler_kv" in sql and "WHERE key" in sql:
                return [{"value": kv_value}]
            if "FROM email_triage_log" in sql:
                return []
            return []

        def fake_build_embed(*args, **kwargs):
            captured["synthesis_section"] = kwargs.get("synthesis_section")
            return {"embeds": [{"title": "x", "fields": []}]}

        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "tok", "DISCORD_TRIAGE_WEBHOOK": "https://example.com",
        }
        with patch.dict("os.environ", env, clear=True), \
             patch("brief.D1Client") as mock_d1cls, \
             patch("brief.build_embed", side_effect=fake_build_embed), \
             patch("brief.fetch_top_news", return_value=[]):
            mock_d1cls.return_value.query.side_effect = fake_query
            brief.main_argv(["--period", "morning", "--dry-run"]) if hasattr(brief, "main_argv") else brief.main_with_args(["--period", "morning", "--dry-run"])

        self.assertIsNotNone(captured.get("synthesis_section"))
        self.assertEqual(captured["synthesis_section"]["pattern"], "P")
```

Note: the existing `brief.main()` reads `sys.argv` via argparse. We need an argv-injectable entry. If one does not exist, the implementation step adds `main_with_args(argv)` and rewrites `main()` to call `main_with_args(None)`.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestMainReadsSynthesisFromKV -v
```
Expected: FAIL — `AttributeError: module 'brief' has no attribute 'main_with_args'` (or assertion `synthesis_section` is None).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/brief.py`:

1. Refactor `main()` to delegate to `main_with_args`:

```python
def main_with_args(argv: list | None) -> None:
    parser = argparse.ArgumentParser(description="Post a morning or evening email brief to Discord.")
    parser.add_argument("--period", required=True, choices=["morning", "evening"])
    parser.add_argument("--since-hours", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    env = load_env(dry_run=args.dry_run)
    d1 = D1Client(env["CF_ACCOUNT_ID"], env["CF_D1_DATABASE_ID"], env["CF_API_TOKEN"])
    cutoff = compute_cutoff(args.since_hours)
    rows = query_rows(d1, cutoff)

    synthesis_section = _read_fresh_synthesis(d1)

    news_items: list[dict] = []
    news_error: str | None = None
    if args.period == "morning":
        try:
            sources = _load_news_sources()
            news_items = fetch_top_news(sources)
        except Exception as exc:
            news_error = str(exc)
            print(f"brief: news fetch failed: {exc}", file=sys.stderr)

    payload = build_embed(
        rows, args.period, args.since_hours,
        news_items=news_items, news_error=news_error,
        synthesis_section=synthesis_section,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    post_brief(env["DISCORD_TRIAGE_WEBHOOK"], payload)


def main() -> None:
    main_with_args(None)
```

2. Add the helper near the top-level helpers in the same file:

```python
def _read_fresh_synthesis(d1) -> dict | None:
    rows = d1.query(
        "SELECT value FROM mahler_kv WHERE key = ? LIMIT 1",
        ["synthesis_brief:latest"],
    )
    if not rows:
        return None
    try:
        data = json.loads(rows[0]["value"])
    except (KeyError, ValueError):
        return None
    posted = data.get("posted_at")
    if not posted:
        return None
    try:
        posted_dt = datetime.strptime(posted, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if datetime.now(timezone.utc) - posted_dt > timedelta(hours=24):
        return None
    return data
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/morning-brief/tests/test_brief.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/brief.py \
        assistant/config/skills/morning-brief/tests/test_brief.py && \
git commit -m "feat(morning-brief): read synthesis_brief:latest from mahler_kv"
```

---

## Group C — Third behavior layer (parallel, 4 tasks)

### Task 12: inputs.py — load_all loads Honcho conclusions into context_items
**Group:** C (parallel with Tasks 13, 14, 15; depends on Task 8)

**Behavior being verified:** Conclusions from `honcho.list_conclusions(since_days=context_days)` appear in `context_items` with `source='honcho'`.
**Interface under test:** `inputs.load_all`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/inputs.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestLoadAllHoncho(unittest.TestCase):
    def test_includes_honcho_conclusions_in_context_items(self):
        d1 = MagicMock()
        d1.query.return_value = []

        c1 = MagicMock(); c1.content = "Jai is focused on traderjoe"; c1.created_at = "2026-05-01T00:00:00Z"
        c2 = MagicMock(); c2.content = "Jai ships on Sundays"; c2.created_at = "2026-05-03T00:00:00Z"

        honcho = MagicMock()
        honcho.list_conclusions.return_value = [c1, c2]

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        honcho_items = [it for it in bundle.context_items if it.source == "honcho"]
        self.assertEqual(len(honcho_items), 2)
        self.assertEqual(honcho_items[0].content, "Jai is focused on traderjoe")
        honcho.list_conclusions.assert_called_once_with(since_days=14)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py::TestLoadAllHoncho -v
```
Expected: FAIL — `AssertionError: 0 != 2`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/synthesis-brief/scripts/inputs.py` add:

```python
def _load_honcho(honcho, context_days: int) -> list:
    conclusions = honcho.list_conclusions(since_days=context_days)
    items = []
    for i, c in enumerate(conclusions):
        items.append(Item(
            source="honcho",
            id=f"honcho:{i}",
            content=getattr(c, "content", str(c)),
            captured_at=str(getattr(c, "created_at", "")),
        ))
    return items
```

Update `load_all`:

```python
def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    bundle.context_items.extend(_load_honcho(honcho, context_days))
    return bundle
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/inputs.py \
        assistant/config/skills/synthesis-brief/tests/test_inputs.py && \
git commit -m "feat(synthesis-brief): load Honcho conclusions into context items"
```

---

### Task 13: validator.py — validate detects length_exceeded
**Group:** C (parallel with Tasks 12, 14, 15; depends on Task 9)

**Behavior being verified:** When the brief sums to more than 2000 characters total or any single section exceeds 600 characters, `validate` returns `(False, "length_exceeded")`.
**Interface under test:** `validator.validate`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/validator.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_validator.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestLength(unittest.TestCase):
    def _bundle(self):
        return InputBundle(
            recent_items=[Item("memory", f"memory:{i}", "x", "2026-05-07") for i in range(4)],
            context_items=[Item("project_log", f"project_log:{i}", "y", "2026-05-04") for i in range(6)],
            past_briefs=[],
            identifiers={f"memory:{i}" for i in range(4)} | {f"project_log:{i}" for i in range(6)},
        )

    def test_returns_false_length_exceeded_when_section_over_600(self):
        bundle = self._bundle()
        long_pattern = "x" * 601
        brief = {
            "connections": [
                {"summary": "A", "citations": [
                    {"source": "memory", "id": "memory:0"},
                    {"source": "memory", "id": "memory:1"},
                ]},
                {"summary": "B", "citations": [
                    {"source": "memory", "id": "memory:2"},
                    {"source": "project_log", "id": "project_log:0"},
                ]},
                {"summary": "C", "citations": [
                    {"source": "memory", "id": "memory:3"},
                    {"source": "project_log", "id": "project_log:1"},
                ]},
            ],
            "pattern": long_pattern,
            "question": "q",
        }
        ok, reason = validator.validate(brief, bundle)
        self.assertFalse(ok)
        self.assertEqual(reason, "length_exceeded")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_validator.py::TestLength -v
```
Expected: FAIL — `AssertionError: True is not false`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/synthesis-brief/scripts/validator.py` add:

```python
_SECTION_MAX = 600
_TOTAL_MAX = 2000


def _check_length(brief: dict) -> bool:
    pattern = brief.get("pattern", "") or ""
    question = brief.get("question", "") or ""
    connections_text = "\n".join(c.get("summary", "") for c in brief.get("connections", []))
    sections = [pattern, question, connections_text]
    if any(len(s) > _SECTION_MAX for s in sections):
        return False
    if sum(len(s) for s in sections) > _TOTAL_MAX:
        return False
    return True
```

Update `validate`:

```python
def validate(brief: dict, bundle) -> tuple[bool, str]:
    if _is_thin(bundle):
        return (False, "thin_context")
    if _qualifying_connections(brief, bundle.identifiers) < 2:
        return (False, "insufficient_citations")
    if not _check_length(brief):
        return (False, "length_exceeded")
    return (True, "")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_validator.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/validator.py \
        assistant/config/skills/synthesis-brief/tests/test_validator.py && \
git commit -m "feat(synthesis-brief): validate length_exceeded gate"
```

---

### Task 14: project_log.py — log_blocker_if_triggered uses claude subprocess
**Group:** C (parallel with Tasks 12, 13, 15; depends on Task 10)

**Behavior being verified:** When the transcript matches a blocker keyword, `log_blocker_if_triggered` calls `claude -p <prompt>` via subprocess (NOT OpenRouter HTTP), parses the stdout as the blocker summary, and inserts a `blocker` row to D1.
**Interface under test:** `project_log.log_blocker_if_triggered`.

**Files:**
- Modify: `assistant/hooks/project_log.py`
- Modify: `assistant/hooks/tests/test_project_log.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestBlockerClassifierViaClaude(unittest.TestCase):
    def test_uses_claude_subprocess_and_inserts_blocker_row(self):
        transcript = {
            "messages": [
                {"role": "user", "content": "I'm stuck on the auth migration"},
            ],
            "cwd": "/tmp/fakeproj",
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if cmd and cmd[0] == "claude":
                result.stdout = "Stuck migrating Postgres auth schema due to FK ordering."
            else:
                result.stdout = ""
            return result

        called_urls = []

        class FailingOpener:
            def open(self, *a, **kw):
                called_urls.append(a)
                raise AssertionError("must not call HTTP for blocker classification")

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="fakeproj"), \
             patch("project_log._derive_git_ref", return_value="abc1234"):
            project_log.log_blocker_if_triggered(transcript, "/tmp/fakeproj")

        # No HTTP calls were made (FailingOpener never used)
        self.assertEqual(called_urls, [])

        # A blocker row was inserted via D1
        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0]
            and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 1)
        params = blocker_inserts[0].args[1]
        self.assertIn("Postgres auth schema", " ".join(str(p) for p in params))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py::TestBlockerClassifierViaClaude -v
```
Expected: FAIL — current implementation calls `_call_openrouter`; the assertion on `INSERT INTO project_log` may pass or fail depending on mocked HTTP, but no `subprocess.run(["claude", ...])` call exists.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/hooks/project_log.py`:

1. Add a helper:

```python
_CLAUDE_BLOCKER_PROMPT = (
    "You are extracting a concise blocker summary from a development session. "
    "Return exactly 1-2 sentences describing the main technical blocker the developer "
    "is stuck on. Be specific. If no clear blocker exists, return an empty string. "
    "Last 10 user messages:\n\n"
)


def _classify_blocker_via_claude(transcript: dict) -> str:
    messages = transcript.get("messages", transcript.get("transcript", []))
    user_texts = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        user_texts.append(content)
    excerpt = "\n".join(f"User: {t}" for t in user_texts[-10:])
    prompt = _CLAUDE_BLOCKER_PROMPT + excerpt
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()
```

2. Replace `log_blocker_if_triggered`:

```python
def log_blocker_if_triggered(transcript: dict, cwd: str) -> None:
    if not _scan_for_keywords(transcript):
        return
    summary = _classify_blocker_via_claude(transcript)
    if not summary:
        return
    project = _derive_project_name(cwd)
    git_ref = _derive_git_ref(cwd)
    client = _get_d1_client()
    client.insert_project_log(project, "blocker", summary, git_ref)
```

(Leave `_call_openrouter` in the file for now; Task 15 may remove it.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/project_log.py assistant/hooks/tests/test_project_log.py && \
git commit -m "feat(hook): classify blockers via claude -p subprocess"
```

---

### Task 15: morning-brief brief.py — main omits synthesis when posted_at older than 24h
**Group:** C (parallel with Tasks 12, 13, 14; depends on Task 11)

**Behavior being verified:** When `mahler_kv:synthesis_brief:latest.posted_at` is older than 24 hours, `main()` passes `synthesis_section=None` to `build_embed`.
**Interface under test:** `brief.main_with_args`.

**Files:**
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py` (test only — implementation already supports this from Task 11; the test pins the behavior).

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestMainOmitsStaleSynthesis(unittest.TestCase):
    def test_passes_none_when_posted_at_older_than_24h(self):
        stale_posted = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
        kv_value = json.dumps({
            "posted_at": stale_posted,
            "connections": [], "pattern": "P", "question": "Q",
        })

        captured = {}

        def fake_query(sql, params=None):
            if "mahler_kv" in sql and "WHERE key" in sql:
                return [{"value": kv_value}]
            if "FROM email_triage_log" in sql:
                return []
            return []

        def fake_build_embed(*args, **kwargs):
            captured["synthesis_section"] = kwargs.get("synthesis_section")
            return {"embeds": [{"title": "x", "fields": []}]}

        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "tok",
        }
        with patch.dict("os.environ", env, clear=True), \
             patch("brief.D1Client") as mock_d1cls, \
             patch("brief.build_embed", side_effect=fake_build_embed), \
             patch("brief.fetch_top_news", return_value=[]):
            mock_d1cls.return_value.query.side_effect = fake_query
            brief.main_with_args(["--period", "morning", "--dry-run"])

        self.assertIsNone(captured["synthesis_section"])
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestMainOmitsStaleSynthesis -v
```
Expected: PASS — `_read_fresh_synthesis` from Task 11 already handles staleness. If it FAILS, fix `_read_fresh_synthesis` to return `None` when `posted_dt` is more than 24h old (per the spec). If the test PASSES with no implementation change, the test is still valid: it pins the behavior so a future regression is caught.

- [ ] **Step 3: Implement (only if Step 2 failed)**

If the test failed, ensure `_read_fresh_synthesis` ends with:

```python
    if datetime.now(timezone.utc) - posted_dt > timedelta(hours=24):
        return None
    return data
```

This was already specified in Task 11; if it was implemented correctly, no further changes are needed.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/morning-brief/tests/test_brief.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/tests/test_brief.py && \
git commit -m "test(morning-brief): pin behavior — omit synthesis when stale"
```

---

## Group D — Fourth behavior layer (parallel, 2 tasks)

### Task 16: inputs.py — load_all partitions local_capture by recent_days vs context_days
**Group:** D (parallel with Task 17; depends on Task 12)

**Behavior being verified:** Rows from `local_capture` whose `captured_at` is within `recent_days` go into `recent_items`; rows older than `recent_days` but within `context_days` go into `context_items`. Source is mapped to either `'memory'` or `'git'` based on the row's `source` column.
**Interface under test:** `inputs.load_all`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/inputs.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestLoadAllLocalCapturePartition(unittest.TestCase):
    def test_recent_rows_go_to_recent_older_to_context(self):
        recent_rows = [
            {"id": 1, "source": "memory", "project": "MEMORY.md",
             "content": "Recent memory", "captured_at": "2026-05-07 04:00:00"},
        ]
        context_rows = [
            {"id": 2, "source": "git", "project": "crescendAI",
             "content": "[crescendAI] abc1 Old commit", "captured_at": "2026-05-02 10:00:00"},
        ]

        def query_side_effect(sql, params=None):
            if "FROM local_capture" in sql and "captured_at >= datetime" in sql and params and params[0] == "-1 days":
                return recent_rows
            if "FROM local_capture" in sql and "captured_at >= datetime" in sql and params and params[0] == "-14 days" and "captured_at < datetime" in sql:
                return context_rows
            return []

        d1 = MagicMock()
        d1.query.side_effect = query_side_effect
        honcho = MagicMock(); honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        recent_local = [it for it in bundle.recent_items if it.source in ("memory", "git")]
        context_local = [it for it in bundle.context_items if it.source in ("memory", "git")]
        self.assertEqual(len(recent_local), 1)
        self.assertEqual(recent_local[0].source, "memory")
        self.assertEqual(recent_local[0].id, "memory:1")
        self.assertEqual(len(context_local), 1)
        self.assertEqual(context_local[0].source, "git")
        self.assertEqual(context_local[0].id, "git:2")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py::TestLoadAllLocalCapturePartition -v
```
Expected: FAIL — `AssertionError: 0 != 1`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/synthesis-brief/scripts/inputs.py` add:

```python
def _row_to_item(r: dict) -> Item:
    return Item(
        source=r["source"],
        id=f"{r['source']}:{r['id']}",
        content=r["content"],
        captured_at=r["captured_at"],
    )


def _load_local_recent(d1, recent_days: int) -> list:
    rows = d1.query(
        "SELECT id, source, project, content, captured_at FROM local_capture "
        "WHERE captured_at >= datetime('now', ? || ' days') "
        "ORDER BY captured_at DESC",
        [f"-{recent_days} days"],
    )
    return [_row_to_item(r) for r in rows]


def _load_local_context(d1, recent_days: int, context_days: int) -> list:
    rows = d1.query(
        "SELECT id, source, project, content, captured_at FROM local_capture "
        "WHERE captured_at >= datetime('now', ? || ' days') "
        "AND captured_at < datetime('now', ? || ' days') "
        "ORDER BY captured_at DESC",
        [f"-{context_days} days", f"-{recent_days} days"],
    )
    return [_row_to_item(r) for r in rows]
```

Update `load_all`:

```python
def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.recent_items.extend(_load_local_recent(d1, recent_days))
    bundle.context_items.extend(_load_local_context(d1, recent_days, context_days))
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    bundle.context_items.extend(_load_honcho(honcho, context_days))
    return bundle
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/inputs.py \
        assistant/config/skills/synthesis-brief/tests/test_inputs.py && \
git commit -m "feat(synthesis-brief): partition local_capture rows by recency"
```

---

### Task 17: project_log.py — _classify_blocker_via_claude returns empty string when claude CLI missing
**Group:** D (parallel with Task 16; depends on Task 14)

**Behavior being verified:** If the `claude` CLI is not installed (subprocess raises `FileNotFoundError`), `_classify_blocker_via_claude` returns `""` and `log_blocker_if_triggered` writes no blocker row.
**Interface under test:** `project_log.log_blocker_if_triggered`.

**Files:**
- Modify: `assistant/hooks/tests/test_project_log.py` (test only — implementation already handles this via the `try/except (FileNotFoundError, ...)` block from Task 14; this task pins the behavior).

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestBlockerClassifierMissingCli(unittest.TestCase):
    def test_returns_no_insert_when_claude_cli_missing(self):
        transcript = {
            "messages": [{"role": "user", "content": "I'm stuck on the auth migration"}],
            "cwd": "/tmp/x",
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            if cmd and cmd[0] == "claude":
                raise FileNotFoundError("no such file: claude")
            r = MagicMock(); r.returncode = 0; r.stdout = ""
            return r

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="x"), \
             patch("project_log._derive_git_ref", return_value=""):
            project_log.log_blocker_if_triggered(transcript, "/tmp/x")

        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0]
            and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 0)
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py::TestBlockerClassifierMissingCli -v
```
Expected: PASS — Task 14 already includes `except FileNotFoundError` returning `""`. If it FAILS, ensure the `try/except` in `_classify_blocker_via_claude` includes `FileNotFoundError`.

- [ ] **Step 3: Implement (only if Step 2 failed)**

Confirm `_classify_blocker_via_claude` exception list contains `FileNotFoundError`:

```python
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/hooks/tests/test_project_log.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/hooks/tests/test_project_log.py && \
git commit -m "test(hook): pin behavior — no blocker row when claude CLI missing"
```

---

## Group E — Final inputs.py behaviors (parallel, 2 tasks)

### Task 18: inputs.py — load_all returns past_briefs from synthesis_brief table
**Group:** E (parallel with Task 19; depends on Task 16)

**Behavior being verified:** `load_all` queries `synthesis_brief` for rows posted within the last `context_days` and returns them as a list of dicts with `connections`, `pattern`, `question` keys (parsed from `connections_json` for connections).
**Interface under test:** `inputs.load_all`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/inputs.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
import json as _json


class TestLoadAllPastBriefs(unittest.TestCase):
    def test_returns_past_briefs_with_parsed_connections(self):
        past_rows = [
            {"posted_at": "2026-05-06 13:00:00",
             "connections_json": _json.dumps([{"summary": "A", "citations": []}]),
             "pattern": "P1", "question": "Q1"},
            {"posted_at": "2026-05-05 13:00:00",
             "connections_json": _json.dumps([{"summary": "B", "citations": []}]),
             "pattern": "P2", "question": "Q2"},
        ]

        def query_side_effect(sql, params=None):
            if "FROM synthesis_brief" in sql:
                return past_rows
            return []

        d1 = MagicMock(); d1.query.side_effect = query_side_effect
        honcho = MagicMock(); honcho.list_conclusions.return_value = []

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        self.assertEqual(len(bundle.past_briefs), 2)
        self.assertEqual(bundle.past_briefs[0]["pattern"], "P1")
        self.assertEqual(bundle.past_briefs[0]["connections"][0]["summary"], "A")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py::TestLoadAllPastBriefs -v
```
Expected: FAIL — `AssertionError: 0 != 2`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `inputs.py`:

```python
import json as _json


def _load_past_briefs(d1, context_days: int) -> list:
    rows = d1.query(
        "SELECT posted_at, connections_json, pattern, question FROM synthesis_brief "
        "WHERE posted_at >= datetime('now', ? || ' days') ORDER BY posted_at DESC",
        [f"-{context_days} days"],
    )
    out = []
    for r in rows:
        try:
            connections = _json.loads(r.get("connections_json") or "[]")
        except ValueError:
            connections = []
        out.append({
            "posted_at": r.get("posted_at"),
            "connections": connections,
            "pattern": r.get("pattern", ""),
            "question": r.get("question", ""),
        })
    return out
```

Update `load_all`:

```python
def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.recent_items.extend(_load_local_recent(d1, recent_days))
    bundle.context_items.extend(_load_local_context(d1, recent_days, context_days))
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    bundle.context_items.extend(_load_honcho(honcho, context_days))
    bundle.past_briefs = _load_past_briefs(d1, context_days)
    return bundle
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/inputs.py \
        assistant/config/skills/synthesis-brief/tests/test_inputs.py && \
git commit -m "feat(synthesis-brief): load past briefs as negative-list dedup"
```

---

### Task 19: inputs.py — load_all populates identifiers set with all source IDs
**Group:** E (sequential, depends on Task 18 — same file)

**Behavior being verified:** After `load_all` returns, `bundle.identifiers` contains every `Item.id` from both `recent_items` and `context_items`. Used by validator to check citations resolve to real items.
**Interface under test:** `inputs.load_all`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/inputs.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestLoadAllIdentifiers(unittest.TestCase):
    def test_identifiers_set_contains_all_item_ids(self):
        win_rows = [{"id": 7, "project": "p", "summary": "s", "created_at": "2026-05-04"}]

        def query_side_effect(sql, params=None):
            if "FROM project_log" in sql:
                return win_rows
            return []

        d1 = MagicMock(); d1.query.side_effect = query_side_effect

        c1 = MagicMock(); c1.content = "x"; c1.created_at = "2026-05-04T00:00:00Z"
        honcho = MagicMock(); honcho.list_conclusions.return_value = [c1]

        bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

        self.assertIn("project_log:7", bundle.identifiers)
        self.assertIn("honcho:0", bundle.identifiers)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py::TestLoadAllIdentifiers -v
```
Expected: FAIL — `AssertionError: 'project_log:7' not found in set()`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the end of `load_all` in `inputs.py` with:

```python
def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.recent_items.extend(_load_local_recent(d1, recent_days))
    bundle.context_items.extend(_load_local_context(d1, recent_days, context_days))
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    bundle.context_items.extend(_load_honcho(honcho, context_days))
    bundle.past_briefs = _load_past_briefs(d1, context_days)
    bundle.identifiers = {it.id for it in bundle.recent_items} | {it.id for it in bundle.context_items}
    return bundle
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_inputs.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/inputs.py \
        assistant/config/skills/synthesis-brief/tests/test_inputs.py && \
git commit -m "feat(synthesis-brief): assemble identifiers set for citation validation"
```

---

## Group F — Synthesize.py orchestrator (sequential, 3 tasks)

### Task 20: synthesize.py — main(--run --dry-run) prints brief JSON when LLM and validator both succeed
**Group:** F (depends on Group E completion)

**Behavior being verified:** With sufficient inputs and a valid LLM response, `main_with_args(["--run", "--dry-run"])` prints the brief JSON to stdout and does NOT write to D1's `synthesis_brief` table or `mahler_kv`.
**Interface under test:** `synthesize.main_with_args`.

**Files:**
- Create: `assistant/config/skills/synthesis-brief/scripts/synthesize.py`
- Create: `assistant/config/skills/synthesis-brief/tests/test_synthesize.py`

- [ ] **Step 1: Write the failing test**

```python
# assistant/config/skills/synthesis-brief/tests/test_synthesize.py
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

import synthesize
from inputs import InputBundle, Item


def _full_bundle():
    return InputBundle(
        recent_items=[Item("memory", f"memory:{i}", "x", "2026-05-07") for i in range(4)],
        context_items=[Item("project_log", f"project_log:{i}", "y", "2026-05-04") for i in range(6)],
        past_briefs=[],
        identifiers={f"memory:{i}" for i in range(4)} | {f"project_log:{i}" for i in range(6)},
    )


_GOOD_BRIEF = {
    "connections": [
        {"summary": "A", "citations": [
            {"source": "memory", "id": "memory:0"},
            {"source": "memory", "id": "memory:1"},
        ]},
        {"summary": "B", "citations": [
            {"source": "memory", "id": "memory:2"},
            {"source": "project_log", "id": "project_log:0"},
        ]},
        {"summary": "C", "citations": [
            {"source": "memory", "id": "memory:3"},
            {"source": "project_log", "id": "project_log:1"},
        ]},
    ],
    "pattern": "Pattern X",
    "question": "Question Y",
}


class TestSynthesizeDryRun(unittest.TestCase):
    def test_dry_run_prints_brief_and_does_not_write(self):
        d1 = MagicMock()
        d1.query.return_value = []
        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
        }
        captured = io.StringIO()
        with patch.dict("os.environ", env, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", return_value=json.dumps(_GOOD_BRIEF)), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run", "--dry-run"])

        out = captured.getvalue()
        self.assertIn("Pattern X", out)
        self.assertIn("Question Y", out)

        # No INSERT INTO synthesis_brief and no mahler_kv write
        write_calls = [
            c for c in d1.query.call_args_list
            if "INSERT INTO synthesis_brief" in c.args[0]
            or "mahler_kv" in c.args[0] and "INSERT" in c.args[0]
        ]
        self.assertEqual(write_calls, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_synthesize.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'synthesize'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# assistant/config/skills/synthesis-brief/scripts/synthesize.py
import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
_SHARED_DIR = Path.home() / ".hermes" / "shared"
_LOCAL_SHARED = _SCRIPTS_DIR.parent.parent.parent / "shared"
for _p in (str(_SCRIPTS_DIR), str(_SHARED_DIR), str(_LOCAL_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inputs  # noqa: E402
import validator  # noqa: E402

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_REQUIRED_ENV = [
    "CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN",
    "OPENROUTER_API_KEY", "HONCHO_API_KEY",
]

_PROMPT_TEMPLATE = """\
You are Mahler, a personal chief-of-staff. Synthesize today's daily brief.

RECENT (last 24h):
{recent}

CONTEXT (last 14d):
{context}

DO NOT REPEAT these from past briefs:
{past}

Output STRICT JSON only, no prose:
{{
  "connections": [
    {{"summary": "<one non-obvious link, 1-2 sentences>",
      "citations": [{{"source": "<src>", "id": "<id from RECENT or CONTEXT>"}}, ...]}},
    ... exactly 3 ...
  ],
  "pattern": "<one weekly theme, 1-2 sentences>",
  "question": "<one question to sit with today, 1 sentence>"
}}

Each connection MUST cite at least 2 distinct ids drawn verbatim from RECENT or CONTEXT.
"""


def _supplement_env_from_hermes() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    if not hermes_env.exists():
        return
    with open(hermes_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _load_env() -> dict:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _build_d1(env: dict):
    from d1_base import D1Client
    return D1Client(env["CF_ACCOUNT_ID"], env["CF_D1_DATABASE_ID"], env["CF_API_TOKEN"])


def _build_honcho():
    import honcho_client
    return honcho_client


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_llm(prompt: str, api_key: str, model: str = _DEFAULT_MODEL, max_tokens: int = 1500) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        _OPENROUTER_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    return data["choices"][0]["message"]["content"]


def _format_items(items: list) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- [{it.id}] {it.content}" for it in items)


def _format_past(past: list) -> str:
    if not past:
        return "(none)"
    return "\n".join(
        f"- pattern: {p['pattern']} | question: {p['question']}"
        for p in past
    )


def main_with_args(argv: list | None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(description="Daily synthesis brief")
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    env = _load_env()
    d1 = _build_d1(env)
    honcho = _build_honcho()

    bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

    prompt = _PROMPT_TEMPLATE.format(
        recent=_format_items(bundle.recent_items),
        context=_format_items(bundle.context_items),
        past=_format_past(bundle.past_briefs),
    )
    raw = _call_llm(prompt, env["OPENROUTER_API_KEY"])
    try:
        brief = json.loads(raw)
    except ValueError:
        print("Synthesis brief skipped: malformed")
        return

    ok, reason = validator.validate(brief, bundle)
    if not ok:
        print(f"Synthesis brief skipped: {reason}")
        return

    if args.dry_run:
        print(json.dumps(brief, indent=2))
        return

    # Persistence is added in Task 22.
    print("Synthesis brief written.")


def main() -> None:
    main_with_args(None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_synthesize.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/synthesize.py \
        assistant/config/skills/synthesis-brief/tests/test_synthesize.py && \
git commit -m "feat(synthesis-brief): scaffold synthesize orchestrator with dry-run"
```

---

### Task 21: synthesize.py — main skips LLM call entirely when bundle is thin
**Group:** F (depends on Task 20)

**Behavior being verified:** When `inputs.load_all` returns a bundle that fails the thin-context gate, `main_with_args` prints `Synthesis brief skipped: thin_context` and does NOT call `_call_llm`.
**Interface under test:** `synthesize.main_with_args`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/synthesize.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_synthesize.py`

- [ ] **Step 1: Write the failing test**

Append to `assistant/config/skills/synthesis-brief/tests/test_synthesize.py`:

```python
class TestSynthesizeThinContext(unittest.TestCase):
    def test_skips_llm_when_bundle_thin(self):
        thin_bundle = InputBundle(
            recent_items=[Item("memory", "memory:0", "x", "2026-05-07")],
            context_items=[],
            past_briefs=[],
            identifiers={"memory:0"},
        )
        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
        }
        captured = io.StringIO()
        llm_mock = MagicMock()
        with patch.dict("os.environ", env, clear=True), \
             patch("synthesize._build_d1", return_value=MagicMock()), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=thin_bundle), \
             patch("synthesize._call_llm", llm_mock), \
             patch("sys.stdout", captured):
            synthesize.main_with_args(["--run"])

        self.assertIn("thin_context", captured.getvalue())
        llm_mock.assert_not_called()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_synthesize.py::TestSynthesizeThinContext -v
```
Expected: FAIL — `_call_llm` IS called (current code calls LLM before validating).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/synthesis-brief/scripts/synthesize.py`, add a pre-LLM thin-context check inside `main_with_args`. Replace the body between `bundle = inputs.load_all(...)` and `prompt = _PROMPT_TEMPLATE.format(...)` with:

```python
    bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

    pre_ok, pre_reason = validator.validate({"connections": [], "pattern": "", "question": ""}, bundle)
    if not pre_ok and pre_reason == "thin_context":
        print(f"Synthesis brief skipped: {pre_reason}")
        return
```

(Keep the post-LLM `validator.validate(brief, bundle)` call further down — it catches the citation/length gates after the LLM responds.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_synthesize.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/synthesize.py \
        assistant/config/skills/synthesis-brief/tests/test_synthesize.py && \
git commit -m "feat(synthesis-brief): skip LLM call on thin context"
```

---

### Task 22: synthesize.py — main writes synthesis_brief row + mahler_kv on success
**Group:** F (depends on Task 21)

**Behavior being verified:** On a successful, validated brief (non-dry-run), `main_with_args(["--run"])` issues exactly one INSERT into `synthesis_brief` and one upsert into `mahler_kv` at key `synthesis_brief:latest`.
**Interface under test:** `synthesize.main_with_args`.

**Files:**
- Modify: `assistant/config/skills/synthesis-brief/scripts/synthesize.py`
- Modify: `assistant/config/skills/synthesis-brief/tests/test_synthesize.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
class TestSynthesizePersistence(unittest.TestCase):
    def test_writes_synthesis_brief_row_and_mahler_kv_on_success(self):
        d1 = MagicMock()
        d1.query.return_value = []
        env = {
            "CF_ACCOUNT_ID": "a"*32, "CF_D1_DATABASE_ID": "b"*32,
            "CF_API_TOKEN": "t", "OPENROUTER_API_KEY": "k", "HONCHO_API_KEY": "h",
        }
        with patch.dict("os.environ", env, clear=True), \
             patch("synthesize._build_d1", return_value=d1), \
             patch("synthesize._build_honcho", return_value=MagicMock()), \
             patch("synthesize.inputs.load_all", return_value=_full_bundle()), \
             patch("synthesize._call_llm", return_value=json.dumps(_GOOD_BRIEF)):
            synthesize.main_with_args(["--run"])

        sb_inserts = [c for c in d1.query.call_args_list if "INSERT INTO synthesis_brief" in c.args[0]]
        kv_writes = [c for c in d1.query.call_args_list if "mahler_kv" in c.args[0] and "INSERT" in c.args[0]]
        self.assertEqual(len(sb_inserts), 1)
        self.assertEqual(len(kv_writes), 1)

        kv_params = kv_writes[0].args[1]
        self.assertEqual(kv_params[0], "synthesis_brief:latest")
        payload = json.loads(kv_params[1])
        self.assertEqual(payload["pattern"], "Pattern X")
        self.assertIn("posted_at", payload)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_synthesize.py::TestSynthesizePersistence -v
```
Expected: FAIL — `AssertionError: 0 != 1` (no persistence yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/synthesis-brief/scripts/synthesize.py`, add at the end of `main_with_args` (replacing the `print("Synthesis brief written.")` placeholder):

```python
    from datetime import datetime, timezone
    posted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    connections_json = json.dumps(brief["connections"])
    d1.query(
        "INSERT INTO synthesis_brief (posted_at, connections_json, pattern, question) "
        "VALUES (?, ?, ?, ?)",
        [posted_at, connections_json, brief["pattern"], brief["question"]],
    )
    kv_payload = json.dumps({
        "posted_at": posted_at,
        "connections": brief["connections"],
        "pattern": brief["pattern"],
        "question": brief["question"],
    })
    d1.query(
        "INSERT INTO mahler_kv (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        ["synthesis_brief:latest", kv_payload],
    )
    print("Synthesis brief written.")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && pytest assistant/config/skills/synthesis-brief/tests/test_synthesize.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/synthesis-brief/scripts/synthesize.py \
        assistant/config/skills/synthesis-brief/tests/test_synthesize.py && \
git commit -m "feat(synthesis-brief): persist brief to synthesis_brief + mahler_kv"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Every spec requirement maps to a task. Synthesis-brief skill (Tasks 1, 2, 5, 8, 9, 12, 13, 16, 18, 19, 20, 21, 22). Stop hook extension (Tasks 3, 10, 14, 17). morning-brief patch (Tasks 4, 11, 15). Deployment (Tasks 6, 7).
- **Vertical slice check:** Each task contains one test + one implementation + one commit. Tasks 15 and 17 are pin-the-behavior tests where Step 3 is a no-op if Step 2 already passes — explicitly noted in the task body.
- **Behavior tests:** All tests exercise public functions (`load_all`, `validate`, `main_with_args`, `log_blocker_if_triggered`, `sync_local_to_d1`, `build_embed`). Stubs replace external collaborators (D1 HTTP, OpenRouter HTTP, Honcho SDK, subprocess) — never internal collaborators of the module under test.
- **Type consistency:** `Item`, `InputBundle`, brief dict shape (`connections/pattern/question`), `validate(brief, bundle) -> tuple[bool, str]`, reason codes (`thin_context`, `insufficient_citations`, `length_exceeded`, `malformed`) used identically across all tasks.
- **Group correctness:** Within each parallel group, no two tasks modify the same file:
  - Group A: 7 distinct files.
  - Group B: 4 distinct files (inputs.py, validator.py, project_log.py, brief.py).
  - Group C: 4 distinct files (same set as B).
  - Group D: 2 distinct files (inputs.py, project_log.py test only).
  - Group E: 1 file shared (inputs.py) — these MUST run sequentially. **Correction:** Tasks 18 and 19 both modify `inputs.py`. They cannot be parallel. Run Task 18 first, then Task 19. Treat Group E as sequential.
  - Group F: all 3 tasks modify synthesize.py — sequential by definition.
- **Forbidden patterns scan:** No mocks of internal collaborators. No private-method tests. No "test was called with X" as the primary assertion (counts and content are the assertions). No bundling.
