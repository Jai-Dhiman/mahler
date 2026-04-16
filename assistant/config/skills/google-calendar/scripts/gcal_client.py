import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


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
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Token refresh failed: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )
    data = json.loads(raw)
    if "access_token" not in data:
        raise RuntimeError(f"Token refresh response missing access_token: {data}")
    return data["access_token"]


def list_events(
    access_token: str,
    time_min: str,
    time_max: str,
    max_results: int = 50,
) -> list[dict]:
    """Fetch calendar events in [time_min, time_max]. Returns normalized event dicts. Raises on API error."""
    params = urllib.parse.urlencode({
        "timeMin": time_min,
        "timeMax": time_max,
        "maxResults": max_results,
        "singleEvents": "true",
        "orderBy": "startTime",
    })
    url = f"{_CALENDAR_API_BASE}/calendars/primary/events?{params}"
    data = _calendar_get(url, access_token)
    return [_normalize_event(item) for item in data.get("items", [])]


def _normalize_event(item: dict) -> dict:
    start_obj = item.get("start", {})
    end_obj = item.get("end", {})
    return {
        "id": item.get("id", ""),
        "summary": item.get("summary", "(no title)"),
        "start": start_obj.get("dateTime") or start_obj.get("date", ""),
        "end": end_obj.get("dateTime") or end_obj.get("date", ""),
        "attendees": [a["email"] for a in item.get("attendees", []) if "email" in a],
        "description": item.get("description", ""),
    }


def _calendar_get(url: str, access_token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    try:
        with _OPENER.open(req) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"Calendar API error: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )
    return json.loads(raw)
