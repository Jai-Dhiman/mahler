# Wiki Semantic Lint Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Claude Code can semantically analyze the wiki's concepts by running `lint.py dump` to get all concept+source data as JSON, then fanning out per-concept subagents to surface sub-concept suggestions and cross-source contradictions.
**Spec:** docs/specs/2026-04-27-wiki-semantic-lint-design.md
**Style:** Follow the project's coding standards (wiki/CLAUDE.md). All commands run from repo root (`mahler/`). Tests run from `wiki/` directory: `cd wiki && python3 -m unittest discover tests`.

---

## Task Groups

Group A (parallel): Task 1, Task 3
Group B (sequential, depends on Group A): Task 2

---

### Task 1: Add `get_source` to `NotionWikiWriter`
**Group:** A (parallel with Task 3)

**Behavior being verified:** `get_source(id)` returns the source's title and body text by making two Notion API calls — one for page properties, one for block children.
**Interface under test:** `NotionWikiWriter.get_source(page_id: str) -> dict`

**Files:**
- Modify: `wiki/scripts/notion_client.py`
- Test: `wiki/tests/test_notion_client.py`

- [ ] **Step 1: Write the failing test**

Add the following class to `wiki/tests/test_notion_client.py` (after the existing `TestListAllConcepts` class):

```python
class TestGetSource(unittest.TestCase):
    def test_returns_title_and_body(self):
        page_response = {
            "id": "src-abc",
            "properties": {
                "Title": {"title": [{"plain_text": "Thin Harness, Fat Skills"}]}
            },
        }
        blocks_response = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "First paragraph."}]},
                },
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "Second paragraph."}]},
                },
            ]
        }
        responses = [_make_response(page_response), _make_response(blocks_response)]
        calls = []

        def side_effect(req):
            calls.append(req)
            return responses[len(calls) - 1]

        with patch.object(_OPENER, "open", side_effect=side_effect):
            writer = _make_writer()
            result = writer.get_source("src-abc")

        self.assertEqual(result["title"], "Thin Harness, Fat Skills")
        self.assertIn("First paragraph.", result["body"])
        self.assertIn("Second paragraph.", result["body"])
        self.assertIn("/pages/src-abc", calls[0].full_url)
        self.assertIn("/blocks/src-abc/children", calls[1].full_url)

    def test_raises_on_notion_api_error(self):
        error_response = {"object": "error", "status": 404, "message": "not found"}
        with patch.object(_OPENER, "open", return_value=_make_response(error_response, status=404)):
            writer = _make_writer()
            with self.assertRaises(RuntimeError) as ctx:
                writer.get_source("missing-id")
        self.assertIn("404", str(ctx.exception))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd wiki && python3 -m unittest tests.test_notion_client.TestGetSource
```
Expected: FAIL — `AttributeError: 'NotionWikiWriter' object has no attribute 'get_source'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the following method to `NotionWikiWriter` in `wiki/scripts/notion_client.py`, after the `list_all_concepts` method:

```python
def get_source(self, page_id: str) -> dict:
    page = self._request("GET", f"/pages/{page_id}", None)
    title_parts = page.get("properties", {}).get("Title", {}).get("title", [])
    title = "".join(p.get("plain_text", "") for p in title_parts).strip()
    body = self._fetch_concept_body_markdown(page_id)
    return {"title": title, "body": body}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd wiki && python3 -m unittest tests.test_notion_client.TestGetSource
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wiki/scripts/notion_client.py wiki/tests/test_notion_client.py && git commit -m "feat(wiki): add get_source to NotionWikiWriter"
```

---

### Task 2: Add `lint.py dump` subcommand
**Group:** B (depends on Group A — uses `get_source` from Task 1)

**Behavior being verified:** `lint.py dump` fetches all concepts and their linked source bodies, deduplicates source fetches, and prints a valid JSON array to stdout where each element contains `title`, `body`, `sources`, and `all_concept_titles`.
**Interface under test:** `lint.main(["dump"])` via captured stdout

**Files:**
- Modify: `wiki/scripts/lint.py`
- Test: `wiki/tests/test_lint.py`

- [ ] **Step 1: Write the failing test**

Add the following imports at the top of `wiki/tests/test_lint.py` (after the existing imports):

```python
import json
```

Add the following class after the existing test classes in `wiki/tests/test_lint.py`:

```python
@patch.dict(os.environ, {
    "NOTION_WIKI_WRITE_TOKEN": "t",
    "NOTION_WIKI_SOURCES_DB_ID": "s",
    "NOTION_WIKI_CONCEPTS_DB_ID": "c",
    "NOTION_WIKI_LOG_DB_ID": "l",
})
class TestDumpCommand(unittest.TestCase):
    def test_emits_concepts_with_source_bodies(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("con-1", "Agent Harnesses", body="About harnesses.", sources=["src-a", "src-b"]),
            _concept("con-2", "Skill Design", body="About skills.", sources=["src-b"]),
        ]

        def get_source_side_effect(sid):
            data = {
                "src-a": {"title": "Thin Harness", "body": "Thin body."},
                "src-b": {"title": "Fat Skills", "body": "Fat body."},
            }
            return data[sid]

        fake_writer.get_source.side_effect = get_source_side_effect

        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["dump"])

        data = json.loads(out.getvalue())
        self.assertEqual(len(data), 2)

        con1 = next(c for c in data if c["title"] == "Agent Harnesses")
        self.assertEqual(con1["body"], "About harnesses.")
        self.assertEqual(len(con1["sources"]), 2)
        source_titles = [s["title"] for s in con1["sources"]]
        self.assertIn("Thin Harness", source_titles)
        self.assertIn("Fat Skills", source_titles)
        self.assertIn("Agent Harnesses", con1["all_concept_titles"])
        self.assertIn("Skill Design", con1["all_concept_titles"])

        con2 = next(c for c in data if c["title"] == "Skill Design")
        self.assertEqual(len(con2["sources"]), 1)
        self.assertEqual(con2["sources"][0]["title"], "Fat Skills")

    def test_deduplicates_source_fetches(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("con-1", "Alpha", body=".", sources=["src-shared"]),
            _concept("con-2", "Beta", body=".", sources=["src-shared"]),
        ]
        fake_writer.get_source.return_value = {"title": "Shared Source", "body": "Shared body."}

        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO):
                lint.main(["dump"])

        self.assertEqual(fake_writer.get_source.call_count, 1)

    def test_concept_with_no_sources_emits_empty_list_without_fetching(self):
        fake_writer = MagicMock()
        fake_writer.list_all_concepts.return_value = [
            _concept("con-1", "Empty Concept", body="No sources yet.", sources=[]),
        ]

        with patch("lint.NotionWikiWriter", return_value=fake_writer):
            with patch("sys.stdout", new_callable=StringIO) as out:
                lint.main(["dump"])

        data = json.loads(out.getvalue())
        self.assertEqual(data[0]["sources"], [])
        fake_writer.get_source.assert_not_called()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd wiki && python3 -m unittest tests.test_lint.TestDumpCommand
```
Expected: FAIL — `SystemExit: 2` (argparse: invalid choice `dump`)

- [ ] **Step 3: Implement the minimum to make the test pass**

In `wiki/scripts/lint.py`, add `import json` to the imports at the top (after the existing imports).

Add the `cmd_dump` function after the `cmd_lint` function:

```python
def cmd_dump(args: argparse.Namespace) -> None:
    writer = _get_writer()
    concepts = writer.list_all_concepts()
    all_concept_titles = [c["title"] for c in concepts]

    unique_source_ids = list(dict.fromkeys(
        sid for c in concepts for sid in c["source_ids"]
    ))
    sources_by_id = {sid: writer.get_source(sid) for sid in unique_source_ids}

    output = []
    for concept in concepts:
        output.append({
            "title": concept["title"],
            "body": concept["body_markdown"],
            "all_concept_titles": all_concept_titles,
            "sources": [sources_by_id[sid] for sid in concept["source_ids"]],
        })

    print(json.dumps(output, ensure_ascii=False, indent=2))
```

In the `main` function, register the subcommand and dispatch it. In the `sub = parser.add_subparsers(...)` block, add after `sub.add_parser("lint")`:

```python
sub.add_parser("dump")
```

In the `if args.command == "lint":` block, add after the existing branch:

```python
elif args.command == "dump":
    cmd_dump(args)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd wiki && python3 -m unittest tests.test_lint.TestDumpCommand
```
Expected: PASS

Run the full test suite to verify no regressions:

```bash
cd wiki && python3 -m unittest discover tests
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add wiki/scripts/lint.py wiki/tests/test_lint.py && git commit -m "feat(wiki): add lint.py dump subcommand"
```

---

### Task 3: Add `## Semantic Analysis` section to `wiki/CLAUDE.md`
**Group:** A (parallel with Task 1)

**Behavior being verified:** When Claude reads `wiki/CLAUDE.md` and the user asks to semantically lint the wiki, it has an unambiguous playbook for the dump-then-subagent workflow.

**Files:**
- Modify: `wiki/CLAUDE.md`

- [ ] **Step 1: Add the section**

Append the following section to the end of `wiki/CLAUDE.md`:

```markdown
## Semantic Analysis

When asked to "semantically lint", "analyze", or "find concept gaps" in the wiki:

1. Run from the repo root:

   ```bash
   python3 wiki/scripts/lint.py dump
   ```

   This prints a JSON array of all concepts with their linked source bodies. It may
   take a minute — it fetches each linked source from Notion.

2. Parse the JSON. Each element has:
   - `title`: concept name
   - `body`: the concept's current synthesized text
   - `sources`: array of `{title, body}` for every source linked to this concept
   - `all_concept_titles`: list of every concept title currently in the wiki

3. Fan out one subagent per concept in parallel. For each concept, pass its element
   from the JSON and ask the subagent two questions:

   a) Given these sources and the existing concept list, is there a sub-concept hiding
      here that warrants its own wiki page? Suggest a title and a one-sentence rationale.
      If no clear sub-concept exists, say "No sub-concept found."

   b) Do any of these sources make contradictory claims? Name the two sources and
      describe the specific tension in one sentence. If no contradiction exists, say
      "No contradictions."

   Concepts with empty `sources` arrays should return "No sources to analyze."

4. Aggregate all subagent findings and print grouped by concept title to session stdout.
   Do not write findings to Notion — the user acts on them manually.
```

- [ ] **Step 2: Verify the section is present**

```bash
grep -c "Semantic Analysis" wiki/CLAUDE.md
```
Expected: `1`

- [ ] **Step 3: Commit**

```bash
git add wiki/CLAUDE.md && git commit -m "docs(wiki): add semantic analysis playbook to CLAUDE.md"
```
