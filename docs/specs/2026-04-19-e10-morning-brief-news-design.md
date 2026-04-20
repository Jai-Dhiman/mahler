# E10 Morning Brief News Section Design

**Goal:** The morning Discord brief includes a "What's Worth Reading" section with the top 5 curated headlines from RSS feeds across five categories (AI/Tech, ML Research, Macro/Markets, Startups/VC, Geopolitics).

**Not in scope:**
- Life tracking (the second half of E10 — separate spec/plan)
- LLM summarization of articles
- Per-user feed preferences or dynamic category management
- Storing fetched news in D1
- The evening brief — news section is morning-only

## Problem

The existing `morning-brief` skill (`config/skills/morning-brief/scripts/brief.py`) posts a Discord embed that only covers email triage results from D1. Phase E2 (Gmail fetch) has not shipped, so the brief currently posts with no content. Even once E2 ships, the brief has no signal about what is happening in the world — no tech news, no market context, no research updates.

## Solution (from the user's perspective)

Every morning at 8am Pacific, the Discord brief ends with a "What's Worth Reading" section: up to 5 headlines with clickable links, drawn from RSS feeds across AI/Tech, ML Research, Macro/Markets, Startups/VC, and Geopolitics. Stories covered by multiple sources appear first (cross-source overlap = importance signal). Most recent story wins ties. If all feeds are unreachable, the section is silently omitted — the email section always posts.

## Design

**Approach:** New `news_fetcher.py` module with a single public function `fetch_top_news()`. Feed URLs live in `news_sources.json` at the skill root — editable without code changes. `brief.py` loads the config, calls the fetcher, and passes results to `build_embed()` as a new optional parameter. Stdlib only — `urllib.request` (already used) + `xml.etree.ElementTree`.

**Key decisions:**

- **RSS over news API:** Free, no keys, predictable signal. At one run per day, free API tiers would work, but RSS avoids external dependencies entirely.
- **Cross-source overlap as ranking signal:** If 2+ feeds publish about the same story (Jaccard similarity ≥ 0.4 on significant words), it surfaces first. This is a free proxy for importance without an LLM call.
- **Top 5 total (not per-category):** Keeps the embed compact. A story covered by 3 sources beats a unique story regardless of category.
- **Headline + link only:** No LLM summarization, zero cost per run.
- **`news_items=None` default on `build_embed()`:** Backward-compatible — existing tests and evening-brief path are unaffected.

## Modules

### `news_fetcher.py`

- **Interface:** `fetch_top_news(sources: dict[str, list[str]], max_items: int = 5) -> list[dict]`
  - `sources`: maps category name to list of RSS feed URLs
  - Returns list of up to `max_items` dicts: `{"title": str, "url": str, "category": str, "source_count": int}`
- **Hides:** HTTP fetching (urllib, 10s timeout, SSL verification), RSS XML parsing (ElementTree), RFC 2822 / ISO 8601 pubDate parsing (email.utils), 24h item cutoff, Jaccard deduplication algorithm, cross-source overlap counting, per-feed exception isolation (failed feeds logged to stderr, others continue)
- **Tested through:** `fetch_top_news()` public function with mocked HTTP responses

### `news_sources.json`

- **Interface:** JSON object mapping category string to array of RSS URL strings
- **Hides:** Which specific feeds represent each category
- **Depth:** N/A (config file, not a module)

### `brief.py` (extended)

- **Interface change:** `build_embed(rows, period, since_hours, news_items=None)` — `news_items` defaults to `None` for full backward compatibility
- **Hides:** (unchanged) Discord embed field construction, truncation logic
- **New behavior:** When `news_items` is non-empty, appends a "What's Worth Reading" embed field. Each line: `[title](url)` with ` · N sources` suffix when `source_count > 1`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/config/skills/morning-brief/scripts/news_fetcher.py` | RSS fetch, parse, deduplicate, rank | New |
| `assistant/config/skills/morning-brief/news_sources.json` | Curated feed URLs for 5 categories | New |
| `assistant/config/skills/morning-brief/scripts/brief.py` | Add `news_items` param to `build_embed()`; add `_load_news_sources()` and `fetch_top_news` import; wire in `main()` | Modify |
| `assistant/config/skills/morning-brief/tests/test_news_fetcher.py` | Behavioral tests for `fetch_top_news()` | New |
| `assistant/config/skills/morning-brief/tests/test_brief.py` | Extend with news field tests for `build_embed()` | Modify |

## Open Questions

- Q: Are the curated RSS feed URLs reachable from the Fly.io container?
  Default: verify during build with `--dry-run`; swap any dead feed URL in `news_sources.json` without code changes.
