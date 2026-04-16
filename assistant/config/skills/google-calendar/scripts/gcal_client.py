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
        raise RuntimeError(f"Token refresh response missing access_token: {data}")
    return data["access_token"]
