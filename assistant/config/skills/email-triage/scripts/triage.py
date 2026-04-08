"""
Email triage orchestration script.

Usage:
    python3 triage.py [--dry-run] [--since-hours N]

--dry-run     Fetch and classify but do NOT write to D1 and do NOT send alerts.
--since-hours N  Only fetch emails received in the last N hours (default: fetch all UNSEEN).
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so sibling modules are importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import subprocess

from d1_client import D1Client
from email_types import EmailMessage
from prefilter import is_noise
import gmail_client
import outlook_client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "x-ai/grok-4.1-fast"
_OUTLOOK_IMAP_HOST = "outlook.office365.com"

_REQUIRED_ENV = [
    "GMAIL_CLIENT_ID",
    "GMAIL_CLIENT_SECRET",
    "GMAIL_REFRESH_TOKEN",
    "OUTLOOK_EMAIL",
    "OUTLOOK_APP_PASSWORD",
    "CF_ACCOUNT_ID",
    "CF_D1_DATABASE_ID",
    "CF_API_TOKEN",
    "OPENROUTER_API_KEY",
]


# ---------------------------------------------------------------------------
# HTTPS opener (same pattern as d1_client)
# ---------------------------------------------------------------------------

def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


# ---------------------------------------------------------------------------
# Priority map loader
# ---------------------------------------------------------------------------

def _load_priority_map() -> str:
    custom = os.environ.get("PRIORITY_MAP_PATH")
    if custom:
        p = Path(custom).expanduser()
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    hermes_path = Path("~/.hermes/workspace/priority-map.md").expanduser()
    if hermes_path.exists():
        with open(hermes_path, "r", encoding="utf-8") as f:
            return f.read()

    fallback = Path(__file__).parent.parent.parent / "priority-map.md"
    with open(fallback, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = (
    "You are an email classifier. Given emails and a priority map, classify each email.\n\n"
    "{priority_map}\n\n"
    "Return a JSON array (no other text) where each element is:\n"
    '{{"message_id": "...", "classification": "URGENT|NEEDS_ACTION|FYI|NOISE", "summary": "one sentence"}}'
)


def classify_batch(
    emails: list[EmailMessage],
    priority_map: str,
    api_key: str,
    model: str,
) -> list[dict]:
    """
    Call OpenRouter to classify a batch of emails.
    Returns list of {"message_id": str, "classification": str, "summary": str}.
    Classification is one of: URGENT, NEEDS_ACTION, FYI, NOISE.
    On parse error, returns classification=NEEDS_ACTION with classification_error=True.
    Raises immediately on 401/403 (auth failure).
    """
    user_payload = json.dumps([
        {
            "message_id": e["message_id"],
            "from": e["from_addr"],
            "subject": e["subject"],
            "preview": e["body_preview"],
        }
        for e in emails
    ])

    body = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT_TEMPLATE.format(priority_map=priority_map),
            },
            {
                "role": "user",
                "content": user_payload,
            },
        ],
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
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                f"OpenRouter auth failure: HTTP {exc.code} — check OPENROUTER_API_KEY"
            ) from exc
        raw = exc.read()
        return _batch_error_results(emails)

    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        results = json.loads(content)
        if not isinstance(results, list):
            raise ValueError("Expected JSON array from LLM")
        # Validate required fields exist in each result
        for item in results:
            if "message_id" not in item or "classification" not in item:
                raise ValueError(f"Missing required fields in LLM result: {item!r}")
        return results
    except (KeyError, IndexError, json.JSONDecodeError, ValueError):
        return _batch_error_results(emails)


def _batch_error_results(emails: list[EmailMessage]) -> list[dict]:
    return [
        {
            "message_id": e["message_id"],
            "classification": "NEEDS_ACTION",
            "summary": "",
            "classification_error": True,
        }
        for e in emails
    ]


# ---------------------------------------------------------------------------
# Urgent alert
# ---------------------------------------------------------------------------

def send_urgent_alert(email: EmailMessage, summary: str) -> None:
    alert_script = (
        Path(__file__).parent.parent.parent / "urgent-alert" / "scripts" / "alert.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(alert_script),
            "--from", email["from_addr"],
            "--subject", email["subject"],
            "--summary", summary,
            "--source", email["source"],
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mahler email triage pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify but do not write to D1 or send alerts",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=None,
        metavar="N",
        help="Only process emails from the last N hours (default: all UNSEEN)",
    )
    return parser.parse_args(argv)


def _load_env(dry_run: bool) -> dict:
    required = list(_REQUIRED_ENV)
    if not dry_run:
        required.append("DISCORD_TRIAGE_WEBHOOK")

    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return {k: os.environ[k] for k in required}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env = _load_env(args.dry_run)

    # 3. Init D1 and ensure tables
    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    d1.ensure_tables()

    # 4. Refresh Gmail access token
    gmail_token = gmail_client.refresh_access_token(
        client_id=env["GMAIL_CLIENT_ID"],
        client_secret=env["GMAIL_CLIENT_SECRET"],
        refresh_token=env["GMAIL_REFRESH_TOKEN"],
    )

    # 5. Fetch emails
    gmail_emails: list[EmailMessage] = []
    gmail_error: Exception | None = None
    try:
        gmail_emails = gmail_client.fetch_unread_emails(gmail_token)
    except Exception as exc:
        gmail_error = exc
        print(f"WARNING: Gmail fetch failed: {exc}", file=sys.stderr)

    outlook_emails: list[EmailMessage] = []
    outlook_error: Exception | None = None
    try:
        outlook_emails = outlook_client.fetch_unread_emails(
            host=_OUTLOOK_IMAP_HOST,
            email_addr=env["OUTLOOK_EMAIL"],
            app_password=env["OUTLOOK_APP_PASSWORD"],
        )
    except Exception as exc:
        outlook_error = exc
        print(f"WARNING: Outlook fetch failed: {exc}", file=sys.stderr)

    gmail_fetched = len(gmail_emails)
    outlook_fetched = len(outlook_emails)
    outlook_junk_count = sum(1 for e in outlook_emails if e["is_junk_rescue"])

    all_emails = gmail_emails + outlook_emails

    # 6. Deduplicate: filter out message_ids already in D1
    new_emails: list[EmailMessage] = []
    for email in all_emails:
        if not d1.is_already_processed(email["message_id"]):
            new_emails.append(email)

    gmail_new = sum(1 for e in new_emails if e["source"] == "gmail")
    outlook_new = sum(1 for e in new_emails if e["source"] == "outlook")

    # 7. Pre-filter: mark deterministic NOISE
    noise_emails: list[EmailMessage] = []
    to_classify: list[EmailMessage] = []
    for email in new_emails:
        if is_noise(email):
            noise_emails.append(email)
        else:
            to_classify.append(email)

    # 8. LLM classify remaining in batches of 20
    priority_map = _load_priority_map()
    api_key = env["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)

    classified_results: list[dict] = []
    batch_size = 20
    for i in range(0, len(to_classify), batch_size):
        batch = to_classify[i : i + batch_size]
        results = classify_batch(batch, priority_map, api_key, model)
        classified_results.extend(results)

    # Build a lookup from message_id to classification result
    classification_map: dict[str, dict] = {r["message_id"]: r for r in classified_results}

    # Prepare all records to store
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    records_to_store: list[tuple[EmailMessage, str, str, bool]] = []

    # NOISE pre-filter results
    for email in noise_emails:
        records_to_store.append((email, "NOISE", "", False))

    # LLM-classified results
    for email in to_classify:
        result = classification_map.get(email["message_id"])
        if result:
            classification = result.get("classification", "NEEDS_ACTION")
            summary = result.get("summary", "")
            is_error = bool(result.get("classification_error", False))
            # Validate classification value
            if classification not in ("URGENT", "NEEDS_ACTION", "FYI", "NOISE"):
                classification = "NEEDS_ACTION"
                is_error = True
        else:
            classification = "NEEDS_ACTION"
            summary = ""
            is_error = True
        records_to_store.append((email, classification, summary, is_error))

    # 9 & 10. Store results and send alerts (unless --dry-run)
    d1_stored = 0
    alerts_sent = 0
    urgent_emails: list[tuple[EmailMessage, str]] = []

    for email, classification, summary, is_error in records_to_store:
        if not args.dry_run:
            d1.insert_triage_result({
                "message_id": email["message_id"],
                "source": email["source"],
                "from_addr": email["from_addr"],
                "subject": email["subject"],
                "received_at": email["received_at"],
                "classification": classification,
                "summary": summary,
                "alerted": 1 if classification == "URGENT" else 0,
                "classification_error": 1 if is_error else 0,
                "processed_at": now_str,
            })
            d1_stored += 1

        if classification == "URGENT":
            urgent_emails.append((email, summary))

    if not args.dry_run:
        for email, summary in urgent_emails:
            send_urgent_alert(email, summary)
            alerts_sent += 1

    # 11. Print summary
    classification_counts: dict[str, int] = {}
    for _, classification, _, _ in records_to_store:
        classification_counts[classification] = classification_counts.get(classification, 0) + 1

    classified_total = len(to_classify)
    urgent_count = classification_counts.get("URGENT", 0)
    needs_action_count = classification_counts.get("NEEDS_ACTION", 0)
    fyi_count = classification_counts.get("FYI", 0)

    print("Triage complete.")
    print(f"  Gmail: {gmail_fetched} fetched, {gmail_new} new")
    outlook_junk_note = f" ({outlook_junk_count} junk rescue)" if outlook_junk_count else ""
    print(f"  Outlook: {outlook_fetched} fetched, {outlook_new} new{outlook_junk_note}")
    print(f"  Pre-filtered NOISE: {len(noise_emails)}")
    print(
        f"  Classified: {classified_total} "
        f"({urgent_count} URGENT, {needs_action_count} NEEDS_ACTION, {fyi_count} FYI)"
    )

    if args.dry_run:
        print("  D1 stored: 0 (dry run)")
        print("  Alerts sent: 0 (dry run)")
    else:
        print(f"  D1 stored: {d1_stored}")
        print(f"  Alerts sent: {alerts_sent}")

    if urgent_emails:
        print()
        print("URGENT emails:")
        for email, _ in urgent_emails:
            print(f"  - [{email['source']}] From: {email['from_addr']} | Subject: {email['subject']}")

    if gmail_error and not gmail_emails:
        print(f"\nNOTE: Gmail fetch failed ({gmail_error}), only Outlook results processed.")
    if outlook_error and not outlook_emails:
        print(f"\nNOTE: Outlook fetch failed ({outlook_error}), only Gmail results processed.")


if __name__ == "__main__":
    main()
