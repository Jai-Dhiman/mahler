# E10 Morning Brief News Section Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** The morning Discord brief includes a "What's Worth Reading" section with the top 5 curated headlines from RSS feeds across five categories, ranked by cross-source overlap then recency.
**Spec:** docs/specs/2026-04-19-e10-morning-brief-news-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## Task Groups

```
Group A (parallel): Task 1, Task 4
Group B (parallel, depends on A): Task 2, Task 5
Group C (parallel, depends on B): Task 3, Task 6
Group D (sequential, depends on C): Task 7
```

---

## Shared test helpers (define once at top of `test_news_fetcher.py`)

The following helpers are used across Tasks 1, 2, and 3. Include them verbatim at module scope in `tests/test_news_fetcher.py`:

```python
import sys
import unittest
import unittest.mock
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from news_fetcher import fetch_top_news


def _rfc2822(hours_ago: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _make_feed_xml(items: list) -> bytes:
    """items: list of (title, url, hours_ago) tuples"""
    item_strs = []
    for title, url, hours_ago in items:
        item_strs.append(
            f"    <item>\n"
            f"      <title>{title}</title>\n"
            f"      <link>{url}</link>\n"
            f"      <pubDate>{_rfc2822(hours_ago)}</pubDate>\n"
            f"    </item>"
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<rss version=\"2.0\">\n"
        "  <channel>\n"
        "    <title>Feed</title>\n"
        + "\n".join(item_strs) + "\n"
        "  </channel>\n"
        "</rss>"
    )
    return xml.encode()


def _make_response(content: bytes):
    mock_resp = unittest.mock.MagicMock()
    mock_resp.read.return_value = content
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
    return mock_resp
```

---

### Task 1: `news_fetcher.py` — fetch, deduplicate, rank
**Group:** A (parallel with Task 4)

**Behavior being verified:** `fetch_top_news()` merges stories with similar titles from different feeds into a single item with `source_count > 1`, and ranks merged stories above unique ones.
**Interface under test:** `fetch_top_news(sources: dict[str, list[str]], max_items: int = 5) -> list[dict]`

**Files:**
- Create: `assistant/config/skills/morning-brief/scripts/news_fetcher.py`
- Create: `assistant/config/skills/morning-brief/tests/test_news_fetcher.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_news_fetcher.py` (after the shared helpers above):

```python
class TestFetchTopNewsDedup(unittest.TestCase):
    def test_same_story_from_two_feeds_merges_and_ranks_first(self):
        feed_a = _make_feed_xml([
            ("OpenAI releases GPT-5 with new capabilities", "https://feeda.example.com/gpt5", 2),
            ("Anthropic raises funding for AI safety research", "https://feeda.example.com/anthropic", 3),
        ])
        feed_b = _make_feed_xml([
            ("OpenAI releases GPT-5 model with enhanced capabilities", "https://feedb.example.com/gpt5", 1),
        ])

        def mock_open(req, timeout=None):
            if "feeda" in req.full_url:
                return _make_response(feed_a)
            return _make_response(feed_b)

        sources = {
            "AI/Tech": [
                "https://feeda.example.com/rss",
                "https://feedb.example.com/rss",
            ]
        }

        with unittest.mock.patch("news_fetcher._OPENER") as mock_opener:
            mock_opener.open.side_effect = mock_open
            items = fetch_top_news(sources)

        self.assertEqual(items[0]["source_count"], 2)
        self.assertIn("GPT-5", items[0]["title"])
        self.assertEqual(items[0]["category"], "AI/Tech")
        self.assertEqual(items[1]["source_count"], 1)
        self.assertIn("Anthropic", items[1]["title"])

    def test_returns_at_most_max_items(self):
        unique_items = [
            (f"Unique story number {i} about artificial intelligence models", f"https://feed.example.com/{i}", i * 0.1)
            for i in range(10)
        ]
        feed_xml = _make_feed_xml(unique_items)

        with unittest.mock.patch("news_fetcher._OPENER") as mock_opener:
            mock_opener.open.return_value = _make_response(feed_xml)
            items = fetch_top_news({"AI/Tech": ["https://feed.example.com/rss"]}, max_items=5)

        self.assertLessEqual(len(items), 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_news_fetcher.py -v 2>&1 | head -30
```
Expected: FAIL — `ModuleNotFoundError: No module named 'news_fetcher'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `assistant/config/skills/morning-brief/scripts/news_fetcher.py`:

```python
import email.utils
import re
import ssl
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_opener()

_STOPWORDS = {
    "about", "above", "after", "again", "against", "their", "there",
    "these", "those", "where", "which", "while", "would", "could",
    "should", "being", "having", "doing", "other", "first", "second",
}


def _significant_words(title: str) -> frozenset:
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return frozenset(w for w in words if len(w) > 4 and w not in _STOPWORDS)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_pubdate(date_str: str):
    if not date_str:
        return None
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _fetch_feed(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mahler/1.0"})
    with _OPENER.open(req, timeout=10) as resp:
        content = resp.read()
    root = ET.fromstring(content)
    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")
        if title_el is None or link_el is None:
            continue
        title = (title_el.text or "").strip()
        url_val = (link_el.text or "").strip()
        pubdate_str = (pubdate_el.text or "").strip() if pubdate_el is not None else ""
        pubdate = _parse_pubdate(pubdate_str) or datetime.min.replace(tzinfo=timezone.utc)
        if not title or not url_val:
            continue
        items.append({"title": title, "url": url_val, "pubdate": pubdate})
    return items


def fetch_top_news(sources: dict, max_items: int = 5) -> list:
    """Fetch RSS feeds, deduplicate by title similarity, rank by source overlap then recency.

    Args:
        sources: maps category name to list of RSS feed URLs
        max_items: maximum items to return

    Returns:
        list of dicts with keys: title, url, category, source_count
    """
    all_items = []
    for category, feed_urls in sources.items():
        for url in feed_urls:
            items = _fetch_feed(url)
            for item in items:
                item["category"] = category
            all_items.extend(items)

    canonical = []
    for item in all_items:
        words = _significant_words(item["title"])
        merged = False
        for canon in canonical:
            if _jaccard(words, canon["_words"]) >= 0.4:
                canon["source_count"] += 1
                if item["pubdate"] > canon["pubdate"]:
                    canon["pubdate"] = item["pubdate"]
                    canon["title"] = item["title"]
                    canon["url"] = item["url"]
                merged = True
                break
        if not merged:
            canonical.append({
                "title": item["title"],
                "url": item["url"],
                "category": item["category"],
                "source_count": 1,
                "pubdate": item["pubdate"],
                "_words": words,
            })

    canonical.sort(key=lambda x: (-x["source_count"], -x["pubdate"].timestamp()))

    return [
        {
            "title": c["title"],
            "url": c["url"],
            "category": c["category"],
            "source_count": c["source_count"],
        }
        for c in canonical[:max_items]
    ]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_news_fetcher.py::TestFetchTopNewsDedup -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/news_fetcher.py assistant/config/skills/morning-brief/tests/test_news_fetcher.py && git commit -m "feat(e10): add news_fetcher with dedup and ranking"
```

---

### Task 2: `news_fetcher.py` — feed failure resilience
**Group:** B (depends on Group A; parallel with Task 5)

**Behavior being verified:** When one RSS feed raises an exception, `fetch_top_news()` continues and returns items from healthy feeds without propagating the error.
**Interface under test:** `fetch_top_news(sources: dict[str, list[str]]) -> list[dict]`

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/news_fetcher.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_news_fetcher.py`

- [ ] **Step 1: Write the failing test**

Add to `TestFetchTopNewsDedup` in `tests/test_news_fetcher.py`:

```python
    def test_failed_feed_does_not_block_healthy_feeds(self):
        import urllib.error

        healthy_feed = _make_feed_xml([
            ("Anthropic raises funding for AI safety research initiatives", "https://feedc.example.com/anthropic", 1),
        ])

        def mock_open(req, timeout=None):
            if "broken" in req.full_url:
                raise urllib.error.URLError("connection refused")
            return _make_response(healthy_feed)

        sources = {
            "AI/Tech": [
                "https://broken.example.com/rss",
                "https://feedc.example.com/rss",
            ]
        }

        with unittest.mock.patch("news_fetcher._OPENER") as mock_opener:
            mock_opener.open.side_effect = mock_open
            items = fetch_top_news(sources)

        self.assertEqual(len(items), 1)
        self.assertIn("Anthropic", items[0]["title"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_news_fetcher.py::TestFetchTopNewsDedup::test_failed_feed_does_not_block_healthy_feeds -v
```
Expected: FAIL — `urllib.error.URLError` propagates and raises instead of returning healthy items.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/news_fetcher.py`, replace the feed loop in `fetch_top_news()`:

```python
    for category, feed_urls in sources.items():
        for url in feed_urls:
            try:
                items = _fetch_feed(url)
            except Exception as exc:
                print(f"news_fetcher: failed to fetch {url}: {exc}", file=sys.stderr)
                continue
            for item in items:
                item["category"] = category
            all_items.extend(items)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_news_fetcher.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/news_fetcher.py assistant/config/skills/morning-brief/tests/test_news_fetcher.py && git commit -m "feat(e10): isolate per-feed failures in news_fetcher"
```

---

### Task 3: `news_fetcher.py` — 24h cutoff
**Group:** C (depends on Group B; parallel with Task 6)

**Behavior being verified:** Items published more than 24 hours ago are excluded from `fetch_top_news()` results.
**Interface under test:** `fetch_top_news(sources: dict[str, list[str]]) -> list[dict]`

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/news_fetcher.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_news_fetcher.py`

- [ ] **Step 1: Write the failing test**

Add to `TestFetchTopNewsDedup` in `tests/test_news_fetcher.py`:

```python
    def test_items_older_than_24h_are_excluded(self):
        feed_xml = _make_feed_xml([
            ("Old article about machine learning neural networks", "https://feed.example.com/old", 25),
            ("Recent article about large language model systems", "https://feed.example.com/recent", 1),
        ])

        with unittest.mock.patch("news_fetcher._OPENER") as mock_opener:
            mock_opener.open.return_value = _make_response(feed_xml)
            items = fetch_top_news({"AI/Tech": ["https://feed.example.com/rss"]})

        titles = [item["title"] for item in items]
        self.assertFalse(any("Old article" in t for t in titles))
        self.assertTrue(any("Recent article" in t for t in titles))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_news_fetcher.py::TestFetchTopNewsDedup::test_items_older_than_24h_are_excluded -v
```
Expected: FAIL — old item appears in results because `_fetch_feed` does not yet filter by age.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/news_fetcher.py`, add `timedelta` to the imports line:

```python
from datetime import datetime, timezone, timedelta
```

In `_fetch_feed()`, add a cutoff check after computing `pubdate`. Replace the final `items.append(...)` block with:

```python
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        pubdate = _parse_pubdate(pubdate_str) or datetime.min.replace(tzinfo=timezone.utc)
        if pubdate < cutoff:
            continue
        if not title or not url_val:
            continue
        items.append({"title": title, "url": url_val, "pubdate": pubdate})
```

The full updated `_fetch_feed` function (replace existing):

```python
def _fetch_feed(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mahler/1.0"})
    with _OPENER.open(req, timeout=10) as resp:
        content = resp.read()
    root = ET.fromstring(content)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")
        if title_el is None or link_el is None:
            continue
        title = (title_el.text or "").strip()
        url_val = (link_el.text or "").strip()
        pubdate_str = (pubdate_el.text or "").strip() if pubdate_el is not None else ""
        pubdate = _parse_pubdate(pubdate_str) or datetime.min.replace(tzinfo=timezone.utc)
        if pubdate < cutoff:
            continue
        if not title or not url_val:
            continue
        items.append({"title": title, "url": url_val, "pubdate": pubdate})
    return items
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_news_fetcher.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/news_fetcher.py assistant/config/skills/morning-brief/tests/test_news_fetcher.py && git commit -m "feat(e10): add 24h cutoff filter to news_fetcher"
```

---

### Task 4: `build_embed()` — accept `news_items` parameter
**Group:** A (parallel with Task 1)

**Behavior being verified:** `build_embed()` called with `news_items=None` produces no "What's Worth Reading" field — existing embed structure is unchanged.
**Interface under test:** `build_embed(rows, period, since_hours, news_items=None) -> dict`

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/brief.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py`

- [ ] **Step 1: Write the failing test**

Add to `TestBuildEmbed` in `tests/test_brief.py`:

```python
    def test_no_news_field_when_news_items_is_none(self):
        payload = build_embed([], "morning", 12, news_items=None)
        fields = payload["embeds"][0].get("fields", [])
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 0)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestBuildEmbed::test_no_news_field_when_news_items_is_none -v
```
Expected: FAIL — `TypeError: build_embed() got an unexpected keyword argument 'news_items'`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/brief.py`, change the `build_embed` signature from:

```python
def build_embed(rows: list[dict], period: str, since_hours: int) -> dict:
```

to:

```python
def build_embed(rows: list[dict], period: str, since_hours: int, news_items: list[dict] | None = None) -> dict:
```

No other changes to the function body — `news_items` is accepted but unused, so the no-field behavior is already correct.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py -v
```
Expected: all tests PASS (existing tests unchanged since `news_items` defaults to `None`)

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/brief.py assistant/config/skills/morning-brief/tests/test_brief.py && git commit -m "feat(e10): add news_items param to build_embed (no-op when None)"
```

---

### Task 5: `build_embed()` — non-empty `news_items` renders field
**Group:** B (depends on Group A; parallel with Task 2)

**Behavior being verified:** `build_embed()` with a non-empty `news_items` list appends a "What's Worth Reading" embed field containing a markdown hyperlink for each item.
**Interface under test:** `build_embed(rows, period, since_hours, news_items) -> dict`

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/brief.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py`

- [ ] **Step 1: Write the failing test**

Add to `TestBuildEmbed` in `tests/test_brief.py`:

```python
    def test_news_items_appends_worth_reading_field(self):
        news_items = [
            {
                "title": "OpenAI releases GPT-5 model",
                "url": "https://example.com/gpt5",
                "category": "AI/Tech",
                "source_count": 1,
            }
        ]
        payload = build_embed([], "morning", 12, news_items=news_items)
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 1)
        self.assertIn(
            "[OpenAI releases GPT-5 model](https://example.com/gpt5)",
            news_fields[0]["value"],
        )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestBuildEmbed::test_news_items_appends_worth_reading_field -v
```
Expected: FAIL — no "What's Worth Reading" field exists in the embed output.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/brief.py`, inside `build_embed()`, add after the `embed["fields"] = fields` line and before the `return`:

```python
    if news_items:
        lines = []
        for item in news_items:
            lines.append(f"[{item['title']}]({item['url']})")
        fields.append({
            "name": "What's Worth Reading",
            "value": "\n".join(lines),
            "inline": False,
        })

    embed["fields"] = fields
    return {"embeds": [embed]}
```

Remove the original `embed["fields"] = fields` and `return {"embeds": [embed]}` lines (they are replaced by the block above). The full tail of `build_embed` should read:

```python
    fields = []

    if needs_action or fyi:
        if needs_action:
            na_lines = []
            for r in needs_action:
                from_addr = r.get("from_addr") or "unknown"
                subject = r.get("subject") or "(no subject)"
                summary = r.get("summary") or ""
                na_lines.append(f"**From:** {from_addr} | **Subject:** {subject}\n> {summary}")
            na_value = _truncate_field(na_lines)
            fields.append({
                "name": f"Needs Action ({len(needs_action)})",
                "value": na_value,
                "inline": False,
            })

        if fyi:
            fyi_lines = [r.get("subject") or "(no subject)" for r in fyi]
            fyi_value = _truncate_field(fyi_lines)
            fields.append({
                "name": f"FYI ({len(fyi)})",
                "value": fyi_value,
                "inline": False,
            })
    else:
        embed["description"] = "Nothing needs your attention."

    noise_count = len(noise)
    fields.append({
        "name": "Noise",
        "value": f"{noise_count} emails filtered",
        "inline": False,
    })

    if news_items:
        lines = []
        for item in news_items:
            lines.append(f"[{item['title']}]({item['url']})")
        fields.append({
            "name": "What's Worth Reading",
            "value": "\n".join(lines),
            "inline": False,
        })

    embed["fields"] = fields
    return {"embeds": [embed]}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/brief.py assistant/config/skills/morning-brief/tests/test_brief.py && git commit -m "feat(e10): render What's Worth Reading field in build_embed"
```

---

### Task 6: `build_embed()` — multi-source suffix
**Group:** C (depends on Group B; parallel with Task 3)

**Behavior being verified:** A news item with `source_count > 1` displays a ` · N sources` suffix after the markdown link.
**Interface under test:** `build_embed(rows, period, since_hours, news_items) -> dict`

**Files:**
- Modify: `assistant/config/skills/morning-brief/scripts/brief.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py`

- [ ] **Step 1: Write the failing test**

Add to `TestBuildEmbed` in `tests/test_brief.py`:

```python
    def test_multi_source_item_shows_source_count_suffix(self):
        news_items = [
            {
                "title": "Big market crash affects global economies",
                "url": "https://example.com/crash",
                "category": "Macro/Markets",
                "source_count": 3,
            }
        ]
        payload = build_embed([], "morning", 12, news_items=news_items)
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertIn("· 3 sources", news_fields[0]["value"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestBuildEmbed::test_multi_source_item_shows_source_count_suffix -v
```
Expected: FAIL — value contains the link but no `· 3 sources` suffix.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `assistant/config/skills/morning-brief/scripts/brief.py`, in the `if news_items:` block inside `build_embed()`, replace:

```python
        lines.append(f"[{item['title']}]({item['url']})")
```

with:

```python
        line = f"[{item['title']}]({item['url']})"
        if item.get("source_count", 1) > 1:
            line += f" · {item['source_count']} sources"
        lines.append(line)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/scripts/brief.py assistant/config/skills/morning-brief/tests/test_brief.py && git commit -m "feat(e10): add multi-source suffix to news items in embed"
```

---

### Task 7: `news_sources.json` + `main()` wiring
**Group:** D (depends on Group C)

**Behavior being verified:** `main()` with `--dry-run` produces a Discord embed JSON payload that includes a "What's Worth Reading" field when `news_sources.json` exists and `fetch_top_news` returns items.
**Interface under test:** `main()` CLI entry point via `sys.argv` + stdout capture

**Files:**
- Create: `assistant/config/skills/morning-brief/news_sources.json`
- Modify: `assistant/config/skills/morning-brief/scripts/brief.py`
- Modify: `assistant/config/skills/morning-brief/tests/test_brief.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_brief.py` — new test class at the bottom of the file:

```python
class TestMainNewsWiring(unittest.TestCase):
    def test_dry_run_output_includes_news_field(self):
        import json
        from io import StringIO

        fixture_items = [
            {
                "title": "AI breakthrough in language model research",
                "url": "https://example.com/ai",
                "category": "AI/Tech",
                "source_count": 1,
            }
        ]
        env = {
            "CF_ACCOUNT_ID": "acc123",
            "CF_D1_DATABASE_ID": "db-abc",
            "CF_API_TOKEN": "tok_abc",
        }

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["brief.py", "--period", "morning", "--dry-run"]
        sys.stdout = StringIO()

        try:
            with unittest.mock.patch.dict("os.environ", env, clear=True), \
                 unittest.mock.patch("brief._load_news_sources", return_value={"AI/Tech": []}), \
                 unittest.mock.patch("brief.fetch_top_news", return_value=fixture_items), \
                 unittest.mock.patch("d1_client.D1Client.query", return_value=[]):
                from brief import main
                main()
                output = sys.stdout.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

        payload = json.loads(output)
        fields = payload["embeds"][0]["fields"]
        news_fields = [f for f in fields if f["name"] == "What's Worth Reading"]
        self.assertEqual(len(news_fields), 1)
        self.assertIn("AI breakthrough", news_fields[0]["value"])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/test_brief.py::TestMainNewsWiring -v
```
Expected: FAIL — `AttributeError: module 'brief' has no attribute '_load_news_sources'` (neither `_load_news_sources` nor `fetch_top_news` exist in `brief.py` yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

**3a.** Create `assistant/config/skills/morning-brief/news_sources.json`:

```json
{
  "AI/Tech": [
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml"
  ],
  "ML Research": [
    "https://arxiv.org/rss/cs.LG",
    "https://arxiv.org/rss/cs.AI"
  ],
  "Macro/Markets": [
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "https://feeds.marketwatch.com/marketwatch/topstories/"
  ],
  "Startups/VC": [
    "https://techcrunch.com/category/startups/feed/",
    "https://venturebeat.com/feed/"
  ],
  "Geopolitics": [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss"
  ]
}
```

**3b.** In `assistant/config/skills/morning-brief/scripts/brief.py`, add the following two lines to the imports block, after the existing imports:

```python
from pathlib import Path
from news_fetcher import fetch_top_news
```

**3c.** In `brief.py`, add the `_load_news_sources` function after the `_OPENER` module-level line and before `load_env`:

```python
def _load_news_sources() -> dict:
    path = Path(__file__).parent.parent / "news_sources.json"
    if not path.exists():
        raise RuntimeError(f"news_sources.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

**3d.** In `brief.py`, update `main()` to load sources, call the fetcher, and pass results to `build_embed`. Replace the existing `main()` body:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Post a morning or evening email brief to Discord.")
    parser.add_argument("--period", required=True, choices=["morning", "evening"])
    parser.add_argument("--since-hours", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = load_env(dry_run=args.dry_run)
    d1 = D1Client(env["CF_ACCOUNT_ID"], env["CF_D1_DATABASE_ID"], env["CF_API_TOKEN"])
    cutoff = compute_cutoff(args.since_hours)
    rows = query_rows(d1, cutoff)

    news_items = []
    if args.period == "morning":
        sources = _load_news_sources()
        news_items = fetch_top_news(sources)

    payload = build_embed(rows, args.period, args.since_hours, news_items=news_items)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    post_brief(env["DISCORD_TRIAGE_WEBHOOK"], payload)
    print("Brief posted.")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/ -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add assistant/config/skills/morning-brief/news_sources.json assistant/config/skills/morning-brief/scripts/brief.py assistant/config/skills/morning-brief/tests/test_brief.py && git commit -m "feat(e10): wire news_sources.json and fetch_top_news into morning brief"
```

---

## Post-build verification

After all tasks complete, run the full test suite and verify the dry-run output manually:

```bash
cd /Users/jdhiman/Documents/mahler && python -m pytest assistant/config/skills/morning-brief/tests/ -v
```

Dry-run smoke test (requires CF env vars from `~/.mahler.env`):

```bash
cd /Users/jdhiman/Documents/mahler/assistant/config/skills/morning-brief/scripts && python brief.py --period morning --dry-run
```

Verify the output JSON contains an embed with a "What's Worth Reading" field populated with 5 or fewer `[title](url)` lines.

If any feed URLs in `news_sources.json` return no items or fail to parse, swap the URL in `news_sources.json` only — no code changes required.
