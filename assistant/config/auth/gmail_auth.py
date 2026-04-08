"""
One-time local script to perform OAuth2 authorization and print the refresh token.

Usage:
    python3 config/auth/gmail_auth.py

Reads GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET from .env or environment.
Uses PKCE (required by Google for Desktop app OAuth clients).

Flow:
  1. Script opens an authorization URL in your browser.
  2. Sign in and grant access.
  3. Browser redirects to http://localhost:1 — shows "connection refused". Expected.
  4. Copy the full URL from the address bar and paste it back here.

After obtaining the refresh token, store it as a Fly.io secret:
    flyctl secrets set GMAIL_REFRESH_TOKEN=<token>
"""

import base64
import hashlib
import json
import os
import secrets
import ssl
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path


def _load_dotenv() -> None:
    env_file = Path(__file__).parent.parent.parent / ".env"
    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()

_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_REDIRECT_URI = "http://localhost:1"
_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256 method)."""
    code_verifier = secrets.token_urlsafe(64)  # 86 url-safe chars
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def _extract_code(code_or_url: str) -> str:
    code_or_url = code_or_url.strip()
    if code_or_url.startswith("http"):
        parsed = urllib.parse.urlparse(code_or_url)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" not in params:
            raise RuntimeError("No 'code' parameter in URL. Copy the full address bar URL.")
        return params["code"][0]
    return code_or_url


def main() -> None:
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")

    if not client_id:
        raise RuntimeError("GMAIL_CLIENT_ID environment variable is not set")
    if not client_secret:
        raise RuntimeError("GMAIL_CLIENT_SECRET environment variable is not set")

    code_verifier, code_challenge = _generate_pkce()

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": _REDIRECT_URI,
        "response_type": "code",
        "scope": _SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{_AUTH_ENDPOINT}?{params}"

    print("Opening browser for Gmail authorization...")
    print(f"\n  {auth_url}\n")
    webbrowser.open(auth_url)

    print("After signing in and granting access, your browser will show")
    print("'This site can't be reached' — that's expected.")
    print("Copy the FULL URL from the address bar and paste it below.\n")

    raw_input = input("Paste the redirect URL: ").strip()
    if not raw_input:
        raise RuntimeError("No input provided.")

    code = _extract_code(raw_input)

    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": _REDIRECT_URI,
        "code_verifier": code_verifier,
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
            f"Token exchange failed: HTTP {exc.code} — {raw.decode('utf-8', errors='replace')}"
        )

    data = json.loads(raw)
    refresh_token = data.get("refresh_token")

    if not refresh_token:
        raise RuntimeError(
            f"No refresh_token in response: {raw.decode('utf-8', errors='replace')}\n"
            "Ensure 'access_type=offline' and 'prompt=consent' were set."
        )

    print("\nRefresh token obtained successfully.")
    print(f"\nRefresh token:\n  {refresh_token}")
    print(f"\nRun this to store it:\n  flyctl secrets set GMAIL_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
