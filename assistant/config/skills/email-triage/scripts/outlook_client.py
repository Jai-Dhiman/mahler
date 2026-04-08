import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from email_types import EmailMessage


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()

_TOKEN_ENDPOINT = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
_GRAPH_BASE = "https://graph.microsoft.com/v1.0/me"


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange a refresh token for a new access token. Raises on auth failure."""
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "scope": "https://graph.microsoft.com/Mail.ReadWrite offline_access",
    }).encode("utf-8")

    req = urllib.request.Request(
        _TOKEN_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Outlook token refresh failed: {exc.code} {body_text}") from exc

    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError(f"No access_token in Outlook token response: {data}")
    return access_token


def _graph_get(url: str, access_token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    try:
        with _OPENER.open(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Graph API request failed: {exc.code} {body_text}") from exc


def _mark_read(message_id: str, access_token: str) -> None:
    url = f"{_GRAPH_BASE}/messages/{message_id}"
    data = json.dumps({"isRead": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="PATCH",
    )
    try:
        with _OPENER.open(req):
            pass
    except urllib.error.HTTPError:
        pass  # marking read is best-effort


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _parse_received_at(date_str: Optional[str]) -> str:
    if not date_str:
        return datetime.now(tz=timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.now(tz=timezone.utc).isoformat()


def _fetch_from_folder(
    folder_path: str,
    access_token: str,
    is_junk: bool,
    max_results: int = 50,
) -> list[EmailMessage]:
    """Fetch unread messages from a Graph API mail folder."""
    params = urllib.parse.urlencode({
        "$filter": "isRead eq false",
        "$select": "id,subject,from,receivedDateTime,body,internetMessageId,internetMessageHeaders",
        "$top": str(max_results),
    })
    url = f"{_GRAPH_BASE}/{folder_path}/messages?{params}"

    try:
        data = _graph_get(url, access_token)
    except RuntimeError:
        return []

    messages = data.get("value", [])
    results: list[EmailMessage] = []

    for msg in messages:
        try:
            msg_id = msg.get("internetMessageId", "").strip("<>") or f"outlook:{msg['id']}"

            from_obj = msg.get("from", {}).get("emailAddress", {})
            from_name = from_obj.get("name", "")
            from_email = from_obj.get("address", "")
            from_addr = f"{from_name} <{from_email}>" if from_name else from_email

            subject = msg.get("subject", "")
            received_at = _parse_received_at(msg.get("receivedDateTime"))

            body_obj = msg.get("body", {})
            raw_body = body_obj.get("content", "")
            if body_obj.get("contentType", "text") == "html":
                raw_body = _strip_html(raw_body)
            body_preview = raw_body.strip()[:500]

            raw_headers = msg.get("internetMessageHeaders", [])
            headers = {h["name"].lower(): h["value"] for h in raw_headers}

            result: EmailMessage = {
                "message_id": msg_id,
                "source": "outlook",
                "from_addr": from_addr,
                "subject": subject,
                "received_at": received_at,
                "body_preview": body_preview,
                "is_junk_rescue": is_junk,
                "headers": headers,
            }
            results.append(result)
            _mark_read(msg["id"], access_token)

        except Exception:
            continue

    return results


def fetch_unread_emails(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    max_results: int = 50,
) -> list[EmailMessage]:
    """Fetch unread emails from Outlook INBOX and Junk via Microsoft Graph API.
    Marks fetched messages as read. Raises on auth failure."""
    access_token = refresh_access_token(client_id, client_secret, refresh_token)

    results: list[EmailMessage] = []
    results.extend(_fetch_from_folder("mailFolders/inbox", access_token, is_junk=False, max_results=max_results))
    results.extend(_fetch_from_folder("mailFolders/junkemail", access_token, is_junk=True, max_results=max_results))
    return results
