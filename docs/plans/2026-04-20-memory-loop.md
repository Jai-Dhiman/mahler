# Memory Loop Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Hermes learns from coding sessions and improves memory quality via two Sunday cron skills backed by a shared honcho-ai SDK client.
**Spec:** docs/specs/2026-04-20-memory-loop-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

Group A (sequential): Task 1
Group B (parallel, depends on Group A): Task 2, Task 3, Task 4, Task 5, Task 6
Group C (sequential, depends on Group B): Task 7

---

### Task 1: Shared honcho_client.py using honcho-ai SDK
**Group:** A

**Behavior being verified:** `conclude(text, session_id)` calls the Honcho SDK conclusions client with the correct content and session_id; `list_conclusions(since_days)` filters out entries older than the window.

**Interface under test:** `honcho_client.conclude`, `honcho_client.list_conclusions`

**Files:**
- Create: `config/shared/honcho_client.py`
- Create: `config/shared/tests/__init__.py`
- Create: `config/shared/tests/test_honcho_client.py`

---

- [ ] **Step 1: Write the failing tests**

```python
# config/shared/tests/test_honcho_client.py
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_CFG = {
    "workspace_id": "mahler",
    "ai_peer_id": "mahler",
    "user_peer_id": "jai",
    "api_key": "test-key",
}


def _mock_sdk():
    mock_conclusions = MagicMock()
    mock_peer = MagicMock()
    mock_peer.conclusions_of.return_value = mock_conclusions
    mock_honcho = MagicMock()
    mock_honcho.peer.return_value = mock_peer
    return mock_honcho, mock_conclusions


class TestConclude(unittest.TestCase):

    def test_conclude_writes_content_and_session_id_to_sdk(self):
        mock_honcho, mock_conclusions = _mock_sdk()
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            honcho_client.conclude("Jai is focused on traderjoe", session_id="project-synthesis")

        mock_conclusions.create.assert_called_once_with([
            {"content": "Jai is focused on traderjoe", "session_id": "project-synthesis"}
        ])

    def test_conclude_raises_runtime_error_on_sdk_exception(self):
        _, mock_conclusions = _mock_sdk()
        mock_conclusions.create.side_effect = Exception("connection refused")
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            with self.assertRaises(RuntimeError) as ctx:
                honcho_client.conclude("text")
        self.assertIn("Honcho conclude failed", str(ctx.exception))


class TestListConclusions(unittest.TestCase):

    def test_list_conclusions_filters_out_entries_older_than_since_days(self):
        now = datetime.now(timezone.utc)
        old = MagicMock()
        old.content = "old fact"
        old.created_at = (now - timedelta(days=31)).isoformat()
        recent = MagicMock()
        recent.content = "recent fact"
        recent.created_at = (now - timedelta(days=5)).isoformat()

        _, mock_conclusions = _mock_sdk()
        mock_conclusions.list.return_value = [old, recent]

        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            result = honcho_client.list_conclusions(since_days=30)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "recent fact")

    def test_list_conclusions_raises_runtime_error_on_sdk_exception(self):
        _, mock_conclusions = _mock_sdk()
        mock_conclusions.list.side_effect = Exception("API down")
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            with self.assertRaises(RuntimeError) as ctx:
                honcho_client.list_conclusions()
        self.assertIn("Honcho list_conclusions failed", str(ctx.exception))


class TestBuildClient(unittest.TestCase):

    def test_build_conclusions_client_initializes_honcho_with_workspace_and_key(self):
        mock_honcho_module = MagicMock()
        mock_honcho_class = MagicMock()
        mock_honcho_module.Honcho = mock_honcho_class
        mock_instance = MagicMock()
        mock_honcho_class.return_value = mock_instance

        with patch.dict("sys.modules", {"honcho": mock_honcho_module}):
            import honcho_client
            import importlib
            importlib.reload(honcho_client)
            honcho_client._build_conclusions_client(_CFG)

        mock_honcho_class.assert_called_with(workspace_id="mahler", api_key="test-key")
        mock_instance.peer.assert_called_with("mahler")
        mock_instance.peer.return_value.conclusions_of.assert_called_with("jai")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/shared/tests/test_honcho_client.py -v 2>&1 | head -30
```
Expected: FAIL — `ModuleNotFoundError: No module named 'honcho_client'`

- [ ] **Step 3: Implement honcho_client.py**

```python
# config/shared/honcho_client.py
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_hermes_env() -> None:
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


def _load_config() -> dict:
    honcho_json = Path.home() / ".hermes" / "honcho.json"
    if not honcho_json.exists():
        raise RuntimeError(f"honcho.json not found at {honcho_json}")
    with open(honcho_json) as f:
        cfg = json.load(f)
    api_key = os.environ.get("HONCHO_API_KEY", "")
    if not api_key:
        raise RuntimeError("HONCHO_API_KEY environment variable not set")
    return {
        "workspace_id": cfg["workspace"],
        "ai_peer_id": cfg["aiPeer"],
        "user_peer_id": cfg["peerName"],
        "api_key": api_key,
    }


def _build_conclusions_client(cfg: dict):
    from honcho import Honcho
    honcho = Honcho(workspace_id=cfg["workspace_id"], api_key=cfg["api_key"])
    return honcho.peer(cfg["ai_peer_id"]).conclusions_of(cfg["user_peer_id"])


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def conclude(text: str, session_id: str = "mahler-memory") -> None:
    """Write one durable conclusion to Honcho. Raises RuntimeError on failure."""
    _load_hermes_env()
    cfg = _load_config()
    conclusions = _build_conclusions_client(cfg)
    try:
        conclusions.create([{"content": text, "session_id": session_id}])
    except Exception as exc:
        raise RuntimeError(f"Honcho conclude failed: {exc}") from exc


def list_conclusions(since_days: int = 30) -> list:
    """Return conclusions written within the last since_days days. Raises RuntimeError on failure."""
    _load_hermes_env()
    cfg = _load_config()
    conclusions = _build_conclusions_client(cfg)
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    try:
        all_items = list(conclusions.list())
    except Exception as exc:
        raise RuntimeError(f"Honcho list_conclusions failed: {exc}") from exc
    return [
        c for c in all_items
        if getattr(c, "created_at", None) is None
        or _parse_dt(c.created_at) >= cutoff
    ]


def query_conclusions(query: str, top_k: int = 10) -> list:
    """Semantic search over conclusions. Raises RuntimeError on failure."""
    _load_hermes_env()
    cfg = _load_config()
    conclusions = _build_conclusions_client(cfg)
    try:
        return list(conclusions.query(query))
    except Exception as exc:
        raise RuntimeError(f"Honcho query_conclusions failed: {exc}") from exc
```

Also create `config/shared/__init__.py` (empty) and `config/shared/tests/__init__.py` (empty).

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/shared/tests/test_honcho_client.py -v
```
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/shared/ && git commit -m "feat(memory-loop): add shared honcho_client using honcho-ai SDK"
```

---

### Task 2: Migrate reflection-journal to shared honcho_client
**Group:** B (parallel with Tasks 3, 4, 5, 6)

**Behavior being verified:** `journal.py --record` still stores reflection and concludes facts to Honcho after removing per-skill honcho_client.py and updating the import.

**Interface under test:** `journal.main(["--record", text])`

**Files:**
- Delete: `config/skills/reflection-journal/scripts/honcho_client.py`
- Delete: `config/skills/reflection-journal/tests/test_honcho_client.py`
- Modify: `config/skills/reflection-journal/scripts/journal.py`
- Modify: `config/skills/reflection-journal/tests/test_journal.py`

---

- [ ] **Step 1: Write the failing test**

Add to `config/skills/reflection-journal/tests/test_journal.py` at top (after existing sys.path insert):

```python
# Add this line immediately after the existing sys.path.insert for scripts:
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
```

The existing test `test_record_stores_raw_text_in_d1_and_concludes_facts_to_honcho` is the regression guard — it must pass after migration.

Run it now to confirm it passes before touching files:
```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/reflection-journal/tests/test_journal.py -v
```

- [ ] **Step 2: Run test — verify it currently PASSES (pre-migration baseline)**

Expected: PASS (4 tests green — this is the baseline to protect)

- [ ] **Step 3: Apply migration**

**Delete** `config/skills/reflection-journal/scripts/honcho_client.py`

**Delete** `config/skills/reflection-journal/tests/test_honcho_client.py`

**Modify** `config/skills/reflection-journal/scripts/journal.py`:

Replace the import block (after `from pathlib import Path`):
```python
# OLD — remove these three constants:
_HONCHO_BASE_URL = "https://api.honcho.dev"
_HONCHO_APP_NAME = "mahler"
_HONCHO_USER_ID = "jai"
```

Add shared path lookup before `import honcho_client`:
```python
_SHARED_DIR = str(Path.home() / ".hermes" / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
```

Replace the conclude call site in `_record()`:
```python
# OLD:
honcho_client.conclude(
    fact,
    env["HONCHO_API_KEY"],
    _HONCHO_BASE_URL,
    _HONCHO_APP_NAME,
    _HONCHO_USER_ID,
)
# NEW:
honcho_client.conclude(fact, session_id="reflection-journal")
```

**Modify** `config/skills/reflection-journal/tests/test_journal.py` — add shared path at top (after existing sys.path.insert):
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/reflection-journal/tests/test_journal.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/reflection-journal/ && git commit -m "refactor(reflection-journal): migrate honcho_client to shared SDK module"
```

---

### Task 3: Migrate kaizen-reflection — remove _run_combined_analysis
**Group:** B (parallel with Tasks 2, 4, 5, 6)

**Behavior being verified:** `reflect.py --run` completes without error and without any honcho-related output when `HONCHO_API_KEY` is absent — `_run_combined_analysis` is fully removed.

**Interface under test:** `reflect.main(["--run"])`

**Files:**
- Delete: `config/skills/kaizen-reflection/scripts/honcho_client.py`
- Delete: `config/skills/kaizen-reflection/tests/test_honcho_client.py`
- Modify: `config/skills/kaizen-reflection/scripts/reflect.py`
- Modify: `config/skills/kaizen-reflection/tests/test_reflect.py`

---

- [ ] **Step 1: Write the failing test**

Add to `config/skills/kaizen-reflection/tests/test_reflect.py` in `TestReflectRunNoProposals`:

```python
def test_run_produces_no_honcho_warning_without_honcho_api_key(self):
    mock_d1 = MagicMock()
    mock_d1.get_triage_patterns_with_reply_rate.return_value = []

    captured_err = io.StringIO()
    with (
        patch.dict("os.environ", _BASE_ENV, clear=True),
        patch("reflect.D1Client", return_value=mock_d1),
        patch("sys.stdout", io.StringIO()),
        patch("sys.stderr", captured_err),
    ):
        import reflect
        reflect.main(["--run"])

    self.assertNotIn("HONCHO", captured_err.getvalue())
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/kaizen-reflection/tests/test_reflect.py::TestReflectRunNoProposals::test_run_produces_no_honcho_warning_without_honcho_api_key -v
```
Expected: FAIL — test finds "HONCHO_API_KEY not set" in stderr (from existing `_run_combined_analysis` WARNING)

- [ ] **Step 3: Remove _run_combined_analysis from reflect.py**

In `config/skills/kaizen-reflection/scripts/reflect.py`:

**Remove** the entire `_run_combined_analysis` function (lines containing `def _run_combined_analysis` through its closing).

**Remove** from `_run()`:
```python
    try:
        _run_combined_analysis(d1, env)
    except Exception as exc:
        sys.stderr.write(f"WARNING: combined analysis failed: {exc}\n")
```

**Remove** these three constants:
```python
_HONCHO_BASE_URL = "https://api.honcho.dev"
_HONCHO_APP_NAME = "mahler"
_HONCHO_USER_ID = "jai"
```

**Remove** the `import honcho_client` line and the `_COMBINED_ANALYSIS_PROMPT` constant.

**Delete** `config/skills/kaizen-reflection/scripts/honcho_client.py`

**Delete** `config/skills/kaizen-reflection/tests/test_honcho_client.py`

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/kaizen-reflection/tests/test_reflect.py -v
```
Expected: PASS (all tests including the new one)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/kaizen-reflection/ && git commit -m "refactor(kaizen-reflection): remove _run_combined_analysis, delete duplicate honcho_client"
```

---

### Task 4: Migrate email-triage to shared honcho_client
**Group:** B (parallel with Tasks 2, 3, 5, 6)

**Behavior being verified:** `triage.py` attribution path still calls `honcho_client.conclude` with correct content after migrating to shared client interface.

**Interface under test:** `triage.py` attribution flow (via existing integration test)

**Files:**
- Delete: `config/skills/email-triage/scripts/honcho_client.py`
- Delete: `config/skills/email-triage/tests/test_honcho_client.py`
- Modify: `config/skills/email-triage/scripts/triage.py`
- Modify: `config/skills/email-triage/tests/test_triage_integration.py`

---

- [ ] **Step 1: Verify baseline**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/email-triage/tests/test_triage_integration.py -v 2>&1 | tail -20
```
Expected: existing tests pass (baseline before migration)

- [ ] **Step 2: Apply migration**

**Delete** `config/skills/email-triage/scripts/honcho_client.py`

**Delete** `config/skills/email-triage/tests/test_honcho_client.py`

**Modify** `config/skills/email-triage/scripts/triage.py`:

Remove these three constants near the top:
```python
_HONCHO_BASE_URL = "https://api.honcho.dev"
_HONCHO_APP_NAME = "mahler"
_HONCHO_USER_ID = "jai"
```

Add shared path before `import honcho_client` (after `_SCRIPTS_DIR` is defined or near other imports):
```python
_SHARED_DIR = str(Path.home() / ".hermes" / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
```

Replace the conclude call site in the attribution function:
```python
# OLD:
honcho_client.conclude(
    fact,
    api_key=honcho_api_key,
    base_url=_HONCHO_BASE_URL,
    app_name=_HONCHO_APP_NAME,
    user_id=_HONCHO_USER_ID,
)
# NEW:
honcho_client.conclude(fact, session_id="email-triage-attribution")
```

**Modify** `config/skills/email-triage/tests/test_triage_integration.py` — add at top after existing sys.path.insert:
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
```

- [ ] **Step 3: Run tests — verify they PASS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/email-triage/tests/test_triage_integration.py -v
```
Expected: PASS (all existing tests)

- [ ] **Step 4: Commit**

```bash
git add assistant/config/skills/email-triage/ && git commit -m "refactor(email-triage): migrate honcho_client to shared SDK module"
```

---

### Task 5: Build project-synthesis skill
**Group:** B (parallel with Tasks 2, 3, 4, 6)

**Behavior being verified:** `synthesize.py --run` calls `honcho_client.conclude` exactly once with the LLM synthesis when D1 has entries; skips `conclude` and prints "No project activity" when D1 is empty.

**Interface under test:** `synthesize.main(["--run"])`

**Files:**
- Create: `config/skills/project-synthesis/SKILL.md`
- Create: `config/skills/project-synthesis/scripts/synthesize.py`
- Create: `config/skills/project-synthesis/scripts/__init__.py`
- Create: `config/skills/project-synthesis/tests/__init__.py`
- Create: `config/skills/project-synthesis/tests/test_synthesize.py`

---

- [ ] **Step 1: Write the failing tests**

```python
# config/skills/project-synthesis/tests/test_synthesize.py
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "CF_ACCOUNT_ID": "acct",
    "CF_D1_DATABASE_ID": "db",
    "CF_API_TOKEN": "cftoken",
    "OPENROUTER_API_KEY": "orkey",
    "HONCHO_API_KEY": "hkey",
}

_SAMPLE_ROWS = [
    {
        "project": "mahler",
        "entry_type": "win",
        "summary": "Shipped morning brief news extension",
        "git_ref": "abc123",
        "created_at": "2026-04-18",
    },
    {
        "project": "traderjoe",
        "entry_type": "blocker",
        "summary": "Spread calculation off by factor of 2",
        "git_ref": "def456",
        "created_at": "2026-04-19",
    },
]


class TestSynthesizeRun(unittest.TestCase):

    def test_run_concludes_llm_output_once_when_entries_exist(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.return_value = _SAMPLE_ROWS
        synthesis = "Jai split attention between mahler and traderjoe this week, shipping a news feature while hitting a spread calculation blocker."

        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
            patch("synthesize._call_llm", return_value=synthesis),
            patch("synthesize.honcho_client") as mock_honcho,
            patch("sys.stdout", io.StringIO()),
        ):
            import synthesize
            synthesize.main(["--run"])

        mock_honcho.conclude.assert_called_once_with(synthesis, session_id="project-synthesis")

    def test_run_skips_conclude_and_prints_no_activity_when_d1_empty(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.return_value = []

        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
            patch("synthesize.honcho_client") as mock_honcho,
            patch("sys.stdout", captured),
        ):
            import synthesize
            synthesize.main(["--run"])

        self.assertIn("No project activity", captured.getvalue())
        mock_honcho.conclude.assert_not_called()

    def test_run_raises_when_d1_query_fails(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.side_effect = RuntimeError("D1 query failed: 500")

        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
        ):
            import synthesize
            with self.assertRaises(RuntimeError) as ctx:
                synthesize.main(["--run"])

        self.assertIn("D1 query failed", str(ctx.exception))

    def test_run_raises_when_llm_fails_and_does_not_conclude(self):
        mock_d1 = MagicMock()
        mock_d1.get_recent_project_log.return_value = _SAMPLE_ROWS

        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("synthesize.D1Client", return_value=mock_d1),
            patch("synthesize._call_llm", side_effect=RuntimeError("OpenRouter error: HTTP 429")),
            patch("synthesize.honcho_client") as mock_honcho,
        ):
            import synthesize
            with self.assertRaises(RuntimeError) as ctx:
                synthesize.main(["--run"])

        self.assertIn("OpenRouter error", str(ctx.exception))
        mock_honcho.conclude.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/project-synthesis/tests/test_synthesize.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'synthesize'`

- [ ] **Step 3: Implement synthesize.py and SKILL.md**

```python
# config/skills/project-synthesis/scripts/synthesize.py
import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_EMAIL_TRIAGE_SCRIPTS = Path.home() / ".hermes" / "skills" / "email-triage" / "scripts"
_SHARED_DIR = Path.home() / ".hermes" / "shared"

for _p in [str(_EMAIL_TRIAGE_SCRIPTS), str(_SHARED_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_client import D1Client
import honcho_client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_REQUIRED_ENV = [
    "CF_ACCOUNT_ID",
    "CF_D1_DATABASE_ID",
    "CF_API_TOKEN",
    "OPENROUTER_API_KEY",
    "HONCHO_API_KEY",
]
_SESSION_ID = "project-synthesis"

_SYNTHESIS_PROMPT = """\
You are Mahler, a personal chief-of-staff. Analyze this week's development activity.

Project log entries (last 7 days, format: [project] date — TYPE: summary):
{log_text}

Write one paragraph (2-4 sentences) covering:
- Which project(s) received the most attention
- Overall trajectory (making progress / stuck / shipping features)
- Any recurring friction pattern visible across sessions

Write in third person starting with "Jai". Be specific about project names. Return only the paragraph.\
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


def _build_https_opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_llm(prompt: str, api_key: str, model: str = _DEFAULT_MODEL, max_tokens: int = 200) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def _format_log(rows: list) -> str:
    return "\n".join(
        "[{project}] {created_at} — {entry_type}: {summary}".format(**r)
        for r in rows
    )


def run(env: dict) -> str:
    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    rows = d1.get_recent_project_log(days=7)
    if not rows:
        msg = "No project activity this week."
        print(msg)
        return msg
    log_text = _format_log(rows)
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    synthesis = _call_llm(
        _SYNTHESIS_PROMPT.format(log_text=log_text),
        env["OPENROUTER_API_KEY"],
        model,
    )
    honcho_client.conclude(synthesis, session_id=_SESSION_ID)
    summary = f"Project synthesis: {len(rows)} entries synthesized."
    print(summary)
    return summary


def main(argv: list | None = None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(description="Project synthesis — weekly coding session summary")
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    env = _load_env()
    run(env)


if __name__ == "__main__":
    main()
```

```markdown
---
name: project-synthesis
description: Weekly coding session synthesis. Reads the past 7 days of project_log wins and blockers from D1, synthesizes one cross-project paragraph covering attention, trajectory, and friction, and writes it to Honcho memory.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [memory, honcho, synthesis, coding, productivity, cron]
    related_skills: [memory-kaizen, kaizen-reflection, reflection-journal]
---

## When to use

- Cron-triggered every Sunday at 18:00 UTC
- When the user asks "synthesize my coding week" or "update project memory"

## Prerequisites

| Variable | Purpose |
|---|---|
| `CF_ACCOUNT_ID` | Cloudflare account ID for D1 API calls |
| `CF_D1_DATABASE_ID` | D1 database ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 read permission |
| `OPENROUTER_API_KEY` | API key for LLM synthesis |
| `HONCHO_API_KEY` | Honcho API key for memory write |

## Procedure

```bash
python3 ~/.hermes/skills/project-synthesis/scripts/synthesize.py --run
```

Prints `"Project synthesis: N entries synthesized."` on success, or `"No project activity this week."` if D1 is empty.

## Cron flow

1. Every Sunday at 18:00 UTC, Mahler runs `synthesize.py --run`
2. If entries exist, Mahler posts the confirmation message to Discord
3. If empty, Mahler posts "No project activity this week" to Discord
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/project-synthesis/tests/test_synthesize.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/project-synthesis/ && git commit -m "feat(project-synthesis): add weekly coding session synthesis skill"
```

---

### Task 6: Build memory-kaizen skill
**Group:** B (parallel with Tasks 2, 3, 4, 5)

**Behavior being verified:** `kaizen.py --run` calls `honcho_client.conclude` N times when LLM returns N PATTERN lines; exits without concluding when conclusions < 5 or LLM returns NO_PATTERNS.

**Interface under test:** `kaizen.main(["--run"])`

**Files:**
- Create: `config/skills/memory-kaizen/SKILL.md`
- Create: `config/skills/memory-kaizen/scripts/kaizen.py`
- Create: `config/skills/memory-kaizen/scripts/__init__.py`
- Create: `config/skills/memory-kaizen/tests/__init__.py`
- Create: `config/skills/memory-kaizen/tests/test_kaizen.py`

---

- [ ] **Step 1: Write the failing tests**

```python
# config/skills/memory-kaizen/tests/test_kaizen.py
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_BASE_ENV = {
    "OPENROUTER_API_KEY": "orkey",
    "HONCHO_API_KEY": "hkey",
}


def _make_conclusion(content: str):
    c = MagicMock()
    c.content = content
    return c


_SEVEN_CONCLUSIONS = [_make_conclusion(f"Jai fact {i}") for i in range(7)]


class TestKaizenRun(unittest.TestCase):

    def test_run_writes_each_pattern_as_separate_conclusion(self):
        llm_response = (
            "PATTERN: Jai consistently ships on Sundays.\n"
            "PATTERN: Jai finds auth-related issues recurring across projects."
        )
        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
            patch("kaizen._call_llm", return_value=llm_response),
            patch("sys.stdout", captured),
        ):
            mock_honcho.list_conclusions.return_value = _SEVEN_CONCLUSIONS
            import kaizen
            kaizen.main(["--run"])

        self.assertEqual(mock_honcho.conclude.call_count, 2)
        mock_honcho.conclude.assert_any_call(
            "Jai consistently ships on Sundays.", session_id="memory-kaizen"
        )
        mock_honcho.conclude.assert_any_call(
            "Jai finds auth-related issues recurring across projects.", session_id="memory-kaizen"
        )
        self.assertIn("2 patterns", captured.getvalue())

    def test_run_skips_conclude_and_reports_insufficient_when_fewer_than_5(self):
        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
            patch("sys.stdout", captured),
        ):
            mock_honcho.list_conclusions.return_value = [_make_conclusion("only one")]
            import kaizen
            kaizen.main(["--run"])

        self.assertIn("Insufficient data", captured.getvalue())
        mock_honcho.conclude.assert_not_called()

    def test_run_skips_conclude_when_llm_returns_no_patterns(self):
        captured = io.StringIO()
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
            patch("kaizen._call_llm", return_value="NO_PATTERNS"),
            patch("sys.stdout", captured),
        ):
            mock_honcho.list_conclusions.return_value = _SEVEN_CONCLUSIONS
            import kaizen
            kaizen.main(["--run"])

        mock_honcho.conclude.assert_not_called()
        self.assertIn("no multi-entry patterns", captured.getvalue())

    def test_run_raises_when_list_conclusions_fails(self):
        with (
            patch.dict("os.environ", _BASE_ENV, clear=True),
            patch("kaizen.honcho_client") as mock_honcho,
        ):
            mock_honcho.list_conclusions.side_effect = RuntimeError("Honcho list_conclusions failed: 503")
            import kaizen
            with self.assertRaises(RuntimeError) as ctx:
                kaizen.main(["--run"])

        self.assertIn("Honcho list_conclusions failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/memory-kaizen/tests/test_kaizen.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'kaizen'`

- [ ] **Step 3: Implement kaizen.py and SKILL.md**

```python
# config/skills/memory-kaizen/scripts/kaizen.py
import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SHARED_DIR = Path.home() / ".hermes" / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

import honcho_client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_REQUIRED_ENV = ["OPENROUTER_API_KEY", "HONCHO_API_KEY"]
_SESSION_ID = "memory-kaizen"
_MIN_CONCLUSIONS = 5

_KAIZEN_PROMPT = """\
You are Mahler reviewing memory conclusions about Jai from the past 30 days.

Conclusions (oldest first):
{conclusions_text}

Identify 2-4 high-signal patterns that appear across multiple entries. Each pattern must be supported by at least 2 different conclusions above.

Write each as one plain-English sentence starting with "Jai ". Return each prefixed with "PATTERN: ".
If fewer than 2 meaningful multi-entry patterns exist, return "NO_PATTERNS".\
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


def _build_https_opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_llm(prompt: str, api_key: str, model: str = _DEFAULT_MODEL, max_tokens: int = 400) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        _OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def run(env: dict) -> str:
    conclusions = honcho_client.list_conclusions(since_days=30)
    if len(conclusions) < _MIN_CONCLUSIONS:
        msg = f"Insufficient data for memory kaizen ({len(conclusions)} conclusions, need {_MIN_CONCLUSIONS})."
        print(msg)
        return msg
    conclusions_text = "\n".join(
        f"{i + 1}. {getattr(c, 'content', str(c))}"
        for i, c in enumerate(conclusions)
    )
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    raw = _call_llm(
        _KAIZEN_PROMPT.format(conclusions_text=conclusions_text),
        env["OPENROUTER_API_KEY"],
        model,
    )
    if raw.strip() == "NO_PATTERNS":
        msg = "Memory kaizen: no multi-entry patterns found."
        print(msg)
        return msg
    patterns = [
        line[len("PATTERN: "):].strip()
        for line in raw.splitlines()
        if line.startswith("PATTERN: ")
    ]
    for pattern in patterns:
        honcho_client.conclude(pattern, session_id=_SESSION_ID)
    summary = f"Memory kaizen: {len(patterns)} patterns written to Honcho."
    print(summary)
    return summary


def main(argv: list | None = None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(description="Memory kaizen — weekly Honcho conclusion distillation")
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    env = _load_env()
    run(env)


if __name__ == "__main__":
    main()
```

```markdown
---
name: memory-kaizen
description: Weekly Honcho memory distillation. Reads the last 30 days of conclusions, identifies 2-4 high-signal patterns that appear across multiple entries, and writes each as a new conclusion. Runs one hour after project-synthesis.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [memory, honcho, kaizen, synthesis, productivity, cron]
    related_skills: [project-synthesis, reflection-journal, kaizen-reflection]
---

## When to use

- Cron-triggered every Sunday at 19:00 UTC (one hour after project-synthesis)
- When the user asks "run memory kaizen" or "distill my Honcho memories"

## Prerequisites

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | API key for LLM pattern synthesis |
| `HONCHO_API_KEY` | Honcho API key for read + write |

## Procedure

```bash
python3 ~/.hermes/skills/memory-kaizen/scripts/kaizen.py --run
```

Prints `"Memory kaizen: N patterns written to Honcho."` on success.
Prints `"Insufficient data for memory kaizen (N conclusions, need 5)."` if fewer than 5 conclusions exist.
Prints `"Memory kaizen: no multi-entry patterns found."` if LLM finds no cross-entry patterns.

## Cron flow

1. Every Sunday at 19:00 UTC, Mahler runs `kaizen.py --run`
2. Mahler posts the result message to Discord
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/memory-kaizen/tests/test_kaizen.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/memory-kaizen/ && git commit -m "feat(memory-kaizen): add weekly Honcho conclusion distillation skill"
```

---

### Task 7: Update Dockerfile
**Group:** C (depends on Group B)

**Behavior being verified:** Dockerfile contains `pip install honcho-ai` and COPY lines for `config/shared`, `config/skills/project-synthesis`, and `config/skills/memory-kaizen`.

**Interface under test:** `grep` assertions on Dockerfile content

**Files:**
- Modify: `assistant/Dockerfile`

---

- [ ] **Step 1: Write the test**

```bash
grep -q "pip install honcho-ai" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS pip" || echo "FAIL pip"
grep -q "config/shared" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS shared" || echo "FAIL shared"
grep -q "project-synthesis" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS project-synthesis" || echo "FAIL project-synthesis"
grep -q "memory-kaizen" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS memory-kaizen" || echo "FAIL memory-kaizen"
```
Expected: all four FAIL before changes.

- [ ] **Step 2: Apply Dockerfile changes**

In `assistant/Dockerfile`, after the `apt-get` RUN block and before the first `COPY`, add:

```dockerfile
# Install Python dependencies for Hermes skills
RUN pip install honcho-ai
```

After the existing skill COPY lines (after the `meeting-followthrough` line), add:

```dockerfile
COPY --chown=hermes:hermes config/shared /home/hermes/.hermes/shared
COPY --chown=hermes:hermes config/skills/project-synthesis /home/hermes/.hermes/skills/project-synthesis
COPY --chown=hermes:hermes config/skills/memory-kaizen /home/hermes/.hermes/skills/memory-kaizen
```

- [ ] **Step 3: Run verification — verify PASS**

```bash
grep -q "pip install honcho-ai" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS pip" || echo "FAIL pip"
grep -q "config/shared" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS shared" || echo "FAIL shared"
grep -q "project-synthesis" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS project-synthesis" || echo "FAIL project-synthesis"
grep -q "memory-kaizen" /Users/jdhiman/Documents/mahler/assistant/Dockerfile && echo "PASS memory-kaizen" || echo "FAIL memory-kaizen"
```
Expected: all four PASS

- [ ] **Step 4: Commit**

```bash
git add assistant/Dockerfile && git commit -m "feat(dockerfile): add honcho-ai install and COPY lines for shared client and new skills"
```
