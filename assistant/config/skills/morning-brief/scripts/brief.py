import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
_EMAIL_TRIAGE_SCRIPTS = _SCRIPTS_DIR.parent.parent / "email-triage" / "scripts"
sys.path.insert(0, str(_EMAIL_TRIAGE_SCRIPTS))
from d1_client import D1Client
from news_fetcher import fetch_top_news


def _supplement_env_from_hermes() -> None:
    """Load missing env vars from ~/.hermes/.env.

    Hermes loads this file into its own process but does not always export
    the values into subprocess environments when running bash commands via
    the terminal tool. This supplements os.environ for any keys not already set.
    """
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


_supplement_env_from_hermes()


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _load_news_sources() -> dict:
    path = Path(__file__).parent.parent / "news_sources.json"
    if not path.exists():
        raise RuntimeError(f"news_sources.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env(dry_run: bool) -> dict:
    required = ["CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN"]
    result = {}
    missing = []
    for key in required:
        val = os.environ.get(key)
        if not val:
            missing.append(key)
        else:
            result[key] = val

    if not dry_run:
        webhook = os.environ.get("DISCORD_TRIAGE_WEBHOOK")
        if not webhook:
            missing.append("DISCORD_TRIAGE_WEBHOOK")
        else:
            result["DISCORD_TRIAGE_WEBHOOK"] = webhook

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return result


def compute_cutoff(since_hours: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def query_rows(d1: D1Client, cutoff: str) -> list[dict]:
    sql = """
        SELECT from_addr, subject, summary, classification, received_at, processed_at
        FROM email_triage_log
        WHERE classification != 'URGENT'
          AND (
            CASE
              WHEN received_at IS NOT NULL THEN received_at >= ?
              ELSE processed_at >= ?
            END
          )
        ORDER BY
          CASE classification
            WHEN 'NEEDS_ACTION' THEN 1
            WHEN 'FYI' THEN 2
            WHEN 'NOISE' THEN 3
            ELSE 4
          END
    """
    return d1.query(sql, [cutoff, cutoff])


def _truncate_field(lines: list[str], max_chars: int = 1024) -> str:
    result = []
    total = 0
    for i, line in enumerate(lines):
        candidate = line + "\n"
        if total + len(candidate) > max_chars:
            remaining = len(lines) - i
            suffix = f"... +{remaining} more"
            # Back off already-added lines if needed
            while result and total + len(suffix) > max_chars:
                removed = result.pop()
                total -= len(removed)
                remaining += 1
                suffix = f"... +{remaining} more"
            result.append(suffix)
            break
        result.append(candidate)
        total += len(candidate)
    return "".join(result).rstrip("\n")


def build_embed(rows: list[dict], period: str, since_hours: int, news_items: list[dict] | None = None, news_error: str | None = None) -> dict:
    if period == "morning":
        color = 3447003
        title_prefix = "Morning Brief"
    else:
        color = 10181046
        title_prefix = "Evening Brief"

    date_str = datetime.now(timezone.utc).strftime("%b %-d")
    title = f"{title_prefix} \u2014 {date_str}"

    needs_action = [r for r in rows if r.get("classification") == "NEEDS_ACTION"]
    fyi = [r for r in rows if r.get("classification") == "FYI"]
    noise = [r for r in rows if r.get("classification") == "NOISE"]

    embed: dict = {
        "title": title,
        "color": color,
        "footer": {"text": f"Last {since_hours}h | Mahler"},
    }

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
            link_label = f"{item['source_count']} sources" if item.get("source_count", 1) > 1 else "read"
            line = f"**{item['title']}** · [{link_label}]({item['url']})"
            lines.append(line)
        fields.append({
            "name": "What's Worth Reading",
            "value": _truncate_field(lines),
            "inline": False,
        })
    elif news_error:
        fields.append({
            "name": "What's Worth Reading",
            "value": f"Fetch failed: {news_error[:200]}",
            "inline": False,
        })

    embed["fields"] = fields
    return {"embeds": [embed]}


def post_brief(webhook_url: str, payload: dict) -> None:
    scheme = webhook_url.split("://")[0] if "://" in webhook_url else ""
    if scheme != "https":
        raise RuntimeError(
            f"DISCORD_TRIAGE_WEBHOOK must be an HTTPS URL, got scheme: {scheme!r}"
        )

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "DiscordBot (mahler, 1.0)")

    try:
        with _OPENER.open(req) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Discord webhook failed: {exc.code} {body}")

    if status not in (200, 204):
        raise RuntimeError(f"Discord webhook failed: {status}")


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

    news_items: list[dict] = []
    news_error: str | None = None
    if args.period == "morning":
        try:
            sources = _load_news_sources()
            news_items = fetch_top_news(sources)
        except Exception as exc:
            news_error = str(exc)
            print(f"brief: news fetch failed: {exc}", file=sys.stderr)

    payload = build_embed(rows, args.period, args.since_hours, news_items=news_items, news_error=news_error)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    post_brief(env["DISCORD_TRIAGE_WEBHOOK"], payload)
    print("Brief posted.")


if __name__ == "__main__":
    main()
