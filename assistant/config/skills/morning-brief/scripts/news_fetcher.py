import email.utils
import re
import ssl
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta


def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPRedirectHandler())
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
        pubdate = _parse_pubdate(pubdate_str)
        if pubdate is None:
            print(f"news_fetcher: skipping '{title}' — missing or unparseable pubDate", file=sys.stderr)
            continue
        if pubdate < cutoff:
            continue
        if not title or not url_val:
            continue
        items.append({"title": title, "url": url_val, "pubdate": pubdate})
    return items


def fetch_top_news(sources: dict, max_items: int = 5) -> list:
    all_items = []
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

    # Cross-category merges are intentional: same story from different category
    # feeds merges into one item, keeping the first-seen category label.
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
