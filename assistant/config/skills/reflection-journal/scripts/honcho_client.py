import json
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
_SESSION_ID = "reflection-journal"


def _ensure_session(api_key: str, base_url: str, app_name: str, user_id: str) -> None:
    url = f"{base_url.rstrip('/')}/v1/apps/{app_name}/users/{user_id}/sessions"
    body = json.dumps({"session_id": _SESSION_ID, "metadata": {}}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _OPENER.open(req):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise RuntimeError(
                f"Honcho session creation failed: HTTP {exc.code}"
            ) from exc


def conclude(
    text: str,
    api_key: str,
    base_url: str,
    app_name: str,
    user_id: str,
) -> None:
    """Deposit a durable fact into Honcho. Raises RuntimeError on HTTP failure."""
    _ensure_session(api_key, base_url, app_name, user_id)
    url = (
        f"{base_url.rstrip('/')}/v1/apps/{app_name}/users/{user_id}"
        f"/sessions/{_SESSION_ID}/metamessages"
    )
    body = json.dumps({
        "content": text,
        "metamessage_type": "honcho_conclude",
        "is_user": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _OPENER.open(req):
            pass
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Honcho conclude failed: HTTP {exc.code}") from exc
