import argparse
import json
import os
import ssl
import urllib.error
import urllib.request


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def build_payload(from_addr: str, subject: str, summary: str, source: str) -> dict:
    return {
        "embeds": [
            {
                "title": "URGENT Email",
                "color": 15158332,
                "fields": [
                    {"name": "From", "value": from_addr, "inline": True},
                    {"name": "Source", "value": source, "inline": True},
                    {"name": "Subject", "value": subject, "inline": False},
                    {"name": "Summary", "value": summary, "inline": False},
                ],
                "footer": {"text": "Mahler Email Triage"},
            }
        ]
    }


def post_alert(webhook_url: str, payload: dict) -> None:
    parsed_scheme = webhook_url.split("://")[0] if "://" in webhook_url else ""
    if parsed_scheme != "https":
        raise RuntimeError(
            f"DISCORD_TRIAGE_WEBHOOK must be an HTTPS URL, got scheme: {parsed_scheme!r}"
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
    parser = argparse.ArgumentParser(description="Post an urgent email alert to Discord.")
    parser.add_argument("--from", dest="from_addr", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--source", required=True, choices=["gmail", "outlook"])
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_TRIAGE_WEBHOOK")
    if not webhook_url:
        raise RuntimeError("DISCORD_TRIAGE_WEBHOOK environment variable is not set")

    payload = build_payload(args.from_addr, args.subject, args.summary, args.source)
    post_alert(webhook_url, payload)
    print("Alert sent.")


if __name__ == "__main__":
    main()
