import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
_SHARED_DIR = Path.home() / ".hermes" / "shared"

for _p in [str(_SCRIPTS_DIR), str(_SHARED_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_client import D1Client
import honcho_client

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_REQUIRED_ENV = [
    "CF_ACCOUNT_ID",
    "CF_D1_DATABASE_ID",
    "CF_API_TOKEN",
    "OPENROUTER_API_KEY",
    "HONCHO_API_KEY",
]
_SESSION_ID = "project-synthesis"

_SYNTHESIS_PROMPT = """\
You are Mahler, a personal chief-of-staff. Analyze this week's development activity.

Project log entries (last 7 days, format: [project] date — TYPE: summary):
{log_text}

Write one paragraph (2-4 sentences) covering:
- Which project(s) received the most attention
- Overall trajectory (making progress / stuck / shipping features)
- Any recurring friction pattern visible across sessions

Write in third person starting with "Jai". Be specific about project names. Return only the paragraph.\
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


def _call_llm(prompt: str, api_key: str, model: str = _DEFAULT_MODEL, max_tokens: int = 200) -> str:
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


def _format_log(rows: list) -> str:
    return "\n".join(
        f"[{r['project']}] {r['created_at']} — {r['entry_type']}: {r['summary']}"
        for r in rows
    )


def run(env: dict) -> str:
    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    rows = d1.get_recent_project_log(days=7)
    if not rows:
        msg = "No project activity this week."
        print(msg)
        return msg
    log_text = _format_log(rows)
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    synthesis = _call_llm(
        _SYNTHESIS_PROMPT.format(log_text=log_text),
        env["OPENROUTER_API_KEY"],
        model,
    )
    honcho_client.conclude(synthesis, session_id=_SESSION_ID)
    summary = f"Project synthesis: {len(rows)} entries synthesized."
    print(summary)
    return summary


def main(argv: list | None = None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(description="Project synthesis — weekly coding session summary")
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args(argv)
    env = _load_env()
    run(env)


if __name__ == "__main__":
    main()
