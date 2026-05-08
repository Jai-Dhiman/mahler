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
_LOCAL_SHARED = _SCRIPTS_DIR.parent.parent.parent / "shared"
for _p in (str(_SCRIPTS_DIR), str(_SHARED_DIR), str(_LOCAL_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inputs  # noqa: E402
import validator  # noqa: E402

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_REQUIRED_ENV = [
    "CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN",
    "OPENROUTER_API_KEY", "HONCHO_API_KEY",
]

_PROMPT_TEMPLATE = """\
You are Mahler, a personal chief-of-staff. Synthesize today's daily brief.

RECENT (last 24h):
{recent}

CONTEXT (last 14d):
{context}

DO NOT REPEAT these from past briefs:
{past}

Output STRICT JSON only, no prose:
{{
  "connections": [
    {{"summary": "<one non-obvious link, 1-2 sentences>",
      "citations": [{{"source": "<src>", "id": "<id from RECENT or CONTEXT>"}}, ...]}},
    ... exactly 3 ...
  ],
  "pattern": "<one weekly theme, 1-2 sentences>",
  "question": "<one question to sit with today, 1 sentence>"
}}

Each connection MUST cite at least 2 distinct ids drawn verbatim from RECENT or CONTEXT.
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


def _build_d1(env: dict):
    from d1_base import D1Client
    return D1Client(env["CF_ACCOUNT_ID"], env["CF_D1_DATABASE_ID"], env["CF_API_TOKEN"])


def _build_honcho():
    import honcho_client
    return honcho_client


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _call_llm(prompt: str, api_key: str, model: str = _DEFAULT_MODEL, max_tokens: int = 1500) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        _OPENROUTER_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter error: HTTP {exc.code}") from exc
    content = data["choices"][0]["message"]["content"]
    if content is None:
        raise RuntimeError("OpenRouter returned null content")
    return content


def _format_items(items: list) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- [{it.id}] {it.content}" for it in items)


def _format_past(past: list) -> str:
    if not past:
        return "(none)"
    return "\n".join(
        f"- pattern: {p['pattern']} | question: {p['question']}"
        for p in past
    )


def main_with_args(argv: list | None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(description="Daily synthesis brief")
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    env = _load_env()
    d1 = _build_d1(env)
    honcho = _build_honcho()

    bundle = inputs.load_all(d1, honcho, recent_days=1, context_days=14)

    # Empty brief probes only the thin_context gate (validator checks thin_context first)
    pre_ok, pre_reason = validator.validate({"connections": [], "pattern": "", "question": ""}, bundle)
    if not pre_ok and pre_reason == "thin_context":
        print(f"Synthesis brief skipped: {pre_reason}")
        return

    prompt = _PROMPT_TEMPLATE.format(
        recent=_format_items(bundle.recent_items),
        context=_format_items(bundle.context_items),
        past=_format_past(bundle.past_briefs),
    )
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    try:
        raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model=model)
    except Exception as exc:
        print(f"Synthesis brief skipped: llm_error — {exc}")
        return
    try:
        brief = json.loads(raw)
    except (ValueError, TypeError):
        print("Synthesis brief skipped: malformed")
        return

    ok, reason = validator.validate(brief, bundle)
    if not ok:
        print(f"Synthesis brief skipped: {reason}")
        return

    if args.dry_run:
        print(json.dumps(brief, indent=2))
        return

    from datetime import datetime, timezone
    posted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    connections_json = json.dumps(brief["connections"])
    d1.query(
        "INSERT INTO synthesis_brief (posted_at, connections_json, pattern, question) "
        "VALUES (?, ?, ?, ?)",
        [posted_at, connections_json, brief["pattern"], brief["question"]],
    )
    kv_payload = json.dumps({
        "posted_at": posted_at,
        "connections": brief["connections"],
        "pattern": brief["pattern"],
        "question": brief["question"],
    })
    d1.query(
        "INSERT INTO mahler_kv (key, value, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        ["synthesis_brief:latest", kv_payload],
    )
    print("Synthesis brief written.")


def main() -> None:
    main_with_args(None)


if __name__ == "__main__":
    main()
