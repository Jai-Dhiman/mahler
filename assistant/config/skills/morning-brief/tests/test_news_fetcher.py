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

    def test_returns_at_most_max_items(self):
        topics = [
            "Federal Reserve raises interest rates amid inflation concerns",
            "SpaceX launches Starship rocket on maiden orbital flight",
            "Apple unveils redesigned MacBook lineup with custom silicon",
            "Scientists discover potential treatment for Alzheimers disease",
            "Electric vehicle sales surpass gasoline cars globally",
            "Congress passes bipartisan infrastructure spending bill",
            "Earthquake strikes coastal region triggering tsunami warning",
            "Breakthrough fusion reactor achieves record energy output",
            "Major cybersecurity breach exposes millions of passwords",
            "Olympic committee announces host city for summer games",
        ]
        unique_items = [
            (topics[i], f"https://feed.example.com/{i}", i * 0.1)
            for i in range(10)
        ]
        feed_xml = _make_feed_xml(unique_items)

        with unittest.mock.patch("news_fetcher._OPENER") as mock_opener:
            mock_opener.open.return_value = _make_response(feed_xml)
            items = fetch_top_news({"AI/Tech": ["https://feed.example.com/rss"]}, max_items=5)

        self.assertEqual(len(items), 5)


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


if __name__ == "__main__":
    unittest.main()
