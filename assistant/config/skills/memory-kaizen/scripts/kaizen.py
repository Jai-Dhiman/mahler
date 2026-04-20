import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SHARED_DIR = Path.home() / ".hermes" / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

import honcho_client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_REQUIRED_ENV = ["OPENROUTER_API_KEY", "HONCHO_API_KEY"]
_SESSION_ID = "memory-kaizen"
_MIN_CONCLUSIONS = 5

_KAIZEN_PROMPT = """\
You are Mahler reviewing memory conclusions about Jai from the past 30 days.

Conclusions (oldest first):
{conclusions_text}

Identify 2-4 high-signal patterns that appear across multiple entries. Each pattern must be supported by at least 2 different conclusions above.

Write each as one plain-English sentence starting with "Jai ". Return each prefixed with "PATTERN: ".
If fewer than 2 meaningful multi-entry patterns exist, return "NO_PATTERNS".\
"""


def _supplement_env_from_hermes() -> None:
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


def _load_env() -> dict:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _build_https_opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_llm(prompt: str, api_key: str, model: str = _DEFAULT_MODEL, max_tokens: int = 400) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc


def run(env: dict) -> str:
    conclusions = honcho_client.list_conclusions(since_days=30)
    if len(conclusions) < _MIN_CONCLUSIONS:
        msg = f"Insufficient data for memory kaizen ({len(conclusions)} conclusions, need {_MIN_CONCLUSIONS})."
        print(msg)
        return msg
    conclusions_text = "\n".join(
        f"{i + 1}. {getattr(c, 'content', str(c))}"
        for i, c in enumerate(conclusions)
    )
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    raw = _call_llm(
        _KAIZEN_PROMPT.format(conclusions_text=conclusions_text),
        env["OPENROUTER_API_KEY"],
        model,
    )
    if raw.strip() == "NO_PATTERNS":
        msg = "Memory kaizen: no multi-entry patterns found."
        print(msg)
        return msg
    patterns = [
        line[len("PATTERN: "):].strip()
        for line in raw.splitlines()
        if line.startswith("PATTERN: ")
    ]
    for pattern in patterns:
        honcho_client.conclude(pattern, session_id=_SESSION_ID)
    summary = f"Memory kaizen: {len(patterns)} patterns written to Honcho."
    print(summary)
    return summary


def main(argv: list | None = None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(description="Memory kaizen — weekly Honcho conclusion distillation")
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    env = _load_env()
    run(env)


if __name__ == "__main__":
    main()
