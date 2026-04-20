import argparse
import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from d1_client import D1Client
import honcho_client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "x-ai/grok-4.1-fast"
_HONCHO_BASE_URL = "https://api.honcho.dev"
_HONCHO_APP_NAME = "mahler"
_HONCHO_USER_ID = "jai"
_REQUIRED_ENV = [
    "CF_ACCOUNT_ID",
    "CF_D1_DATABASE_ID",
    "CF_API_TOKEN",
    "OPENROUTER_API_KEY",
    "HONCHO_API_KEY",
]

_QUESTIONS = """\
Reflection time. Reply to all three in one message:

1. How did last week go overall?
2. What drained your energy or felt hard this week?
3. What are you avoiding or putting off?
"""

_SYNTHESIS_PROMPT = """\
You are a personal chief-of-staff assistant processing a weekly reflection.

Raw reflection:
{raw_text}

Extract 2-3 durable facts about this person's current state, values, or patterns. \
Write each as a plain-English sentence useful as context in future conversations.

Return each fact on its own line, prefixed with "FACT: ". Return at most 3 facts.\
"""


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_llm(prompt: str, api_key: str, model: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response shape: {data}"
        ) from exc


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
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _prompt() -> None:
    print(_QUESTIONS)


def _record(answer_text: str, env: dict) -> None:
    week_of = datetime.now().strftime("%G-W%V")
    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    d1.insert_reflection(week_of, answer_text)

    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    prompt = _SYNTHESIS_PROMPT.format(raw_text=answer_text)
    raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)

    facts = [
        line[len("FACT: "):].strip()
        for line in raw.splitlines()
        if line.startswith("FACT: ")
    ]
    for fact in facts:
        honcho_client.conclude(
            fact,
            env["HONCHO_API_KEY"],
            _HONCHO_BASE_URL,
            _HONCHO_APP_NAME,
            _HONCHO_USER_ID,
        )
    print("Reflection recorded.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mahler reflection journal")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", action="store_true")
    group.add_argument("--record", metavar="ANSWER_TEXT")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _supplement_env_from_hermes()
    args = _parse_args(argv)
    if args.prompt:
        _prompt()
    else:
        env = _load_env()
        _record(args.record, env)


if __name__ == "__main__":
    main()
