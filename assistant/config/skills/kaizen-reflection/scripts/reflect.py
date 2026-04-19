import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from d1_client import D1Client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "x-ai/grok-4.1-fast"
_REQUIRED_ENV = ["CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN", "OPENROUTER_API_KEY"]


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
        if exc.code in (401, 403):
            raise RuntimeError(
                f"OpenRouter auth failure: HTTP {exc.code} — check OPENROUTER_API_KEY"
            ) from exc
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc


_PROPOSAL_PROMPT = """\
You are analyzing email triage patterns for a personal chief-of-staff assistant.

Current priority map:
{priority_map}

Detected patterns (senders appearing frequently at the same classification tier):
{patterns}

For each pattern, propose a reclassification only if the current tier appears wrong.
A sender appearing many times as NEEDS_ACTION with no escalation signal likely belongs at FYI.

Return a JSON array (no other text). Each element must have exactly these fields:
{{"sender": "<email address>", "current_tier": "<tier>", "proposed_tier": "<tier>", "evidence": "<one sentence>"}}

If no reclassifications are warranted, return an empty JSON array: []\
"""


def _load_env() -> dict:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _run(since_days: int, env: dict) -> None:
    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    patterns = d1.get_triage_patterns(since_days=since_days, min_count=3)
    if not patterns:
        print("No proposals this week.")
        return

    priority_map = d1.get_priority_map()
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)

    patterns_text = "\n".join(
        f"- {p['from_addr']}: {p['occurrence_count']} times as {p['classification']}"
        for p in patterns
    )
    prompt = _PROPOSAL_PROMPT.format(priority_map=priority_map, patterns=patterns_text)
    raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)

    try:
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            raise ValueError("LLM did not return a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"LLM returned unparseable proposals: {exc}\nRaw: {raw}") from exc

    if not proposals:
        print("No proposals this week.")
        return

    print(json.dumps(proposals))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mahler kaizen reflection")
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--since-days", type=int, default=7, metavar="N")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _supplement_env_from_hermes()
    args = _parse_args(argv)
    env = _load_env()
    _run(since_days=args.since_days, env=env)


if __name__ == "__main__":
    main()
