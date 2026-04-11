import base64
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import parsedate_to_datetime

from email_types import EmailMessage

_GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange refresh token for access token. Raises on failure."""
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }).encode("utf-8")

    req = urllib.request.Request(
        _TOKEN_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Token refresh failed: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )

    if status != 200:
        raise RuntimeError(
            f"Token refresh failed: HTTP {status} — {raw.decode('utf-8', errors='replace')}"
        )

    data = json.loads(raw)
    if "access_token" not in data:
        raise RuntimeError(
            f"Token refresh response missing access_token: {raw.decode('utf-8', errors='replace')}"
        )

    return data["access_token"]


def fetch_unread_emails(access_token: str, max_results: int = 50, since_days: int = 7) -> list[EmailMessage]:
    """Fetch unread emails from Gmail inbox received within the last since_days days. Raises on API error."""
    query = f"is:unread in:inbox newer_than:{since_days}d"
    list_url = (
        f"{_GMAIL_API_BASE}/messages"
        f"?q={urllib.parse.quote(query)}&maxResults={max_results}"
    )
    list_data = _gmail_get(list_url, access_token)

    message_stubs = list_data.get("messages", [])
    if not message_stubs:
        return []

    results: list[EmailMessage] = []
    for stub in message_stubs:
        msg_id = stub["id"]
        msg_url = f"{_GMAIL_API_BASE}/messages/{msg_id}?format=full"
        msg_data = _gmail_get(msg_url, access_token)
        results.append(_parse_message(msg_data))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gmail_get(url: str, access_token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    try:
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Gmail API error: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )

    if status != 200:
        raise RuntimeError(
            f"Gmail API error: HTTP {status} — {raw.decode('utf-8', errors='replace')}"
        )

    return json.loads(raw)


def _decode_header_value(raw: str) -> str:
    parts = decode_header(raw)
    decoded_parts = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded_parts.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(part)
    return "".join(decoded_parts)


def _parse_date(date_str: str | None) -> str:
    if date_str:
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _extract_body(payload: dict) -> str:
    mime_type = payload.get("mimeType", "")

    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        return ""

    if mime_type == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
            return _strip_html(html)
        return ""

    # multipart: recurse through parts, prefer text/plain
    parts = payload.get("parts", [])
    plain_text = ""
    html_fallback = ""

    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                plain_text = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        elif part_mime == "text/html" and not plain_text:
            data = part.get("body", {}).get("data", "")
            if data:
                html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                html_fallback = _strip_html(html)
        elif part_mime.startswith("multipart/"):
            nested = _extract_body(part)
            if nested and not plain_text:
                plain_text = nested

    return plain_text or html_fallback


def _parse_message(msg: dict) -> EmailMessage:
    payload = msg.get("payload", {})
    header_list = payload.get("headers", [])

    headers: dict = {}
    for h in header_list:
        headers[h["name"].lower()] = h["value"]

    from_addr = headers.get("from", "")
    subject_raw = headers.get("subject", "")
    subject = _decode_header_value(subject_raw)
    received_at = _parse_date(headers.get("date"))

    body_raw = _extract_body(payload)
    body_preview = body_raw.strip()[:500]

    return EmailMessage(
        message_id=msg["id"],
        source="gmail",
        from_addr=from_addr,
        subject=subject,
        received_at=received_at,
        body_preview=body_preview,
        is_junk_rescue=False,
        headers=headers,
    )
