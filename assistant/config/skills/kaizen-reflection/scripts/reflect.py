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
import honcho_client

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

Detected patterns (senders with frequency and reply rate over the past week):
{patterns}

For each pattern, propose a reclassification only if the current tier appears wrong.
A sender appearing many times as NEEDS_ACTION with no escalation signal likely belongs at FYI.
A sender with >50% reply rate is confirmed important — do not downgrade.
A sender with 0 replies over 5 or more occurrences is a strong candidate to downgrade.

Return a JSON array (no other text). Each element must have exactly these fields:
{{"sender": "<email address>", "current_tier": "<tier>", "proposed_tier": "<tier>", "evidence": "<one sentence>"}}

If no reclassifications are warranted, return an empty JSON array: []\
"""

_PROJECT_ANALYSIS_PROMPT = """\
You are analyzing project activity logs for a personal chief-of-staff assistant.

Recent project activity (last 7 days):
{project_log_text}

Identify 1-3 meaningful patterns across these entries. Focus on:
- Projects with many blockers and no wins (possible focus or morale issue)
- Recurring themes in blockers across different projects
- Sustained absence of progress in an area that was recently active

For each meaningful pattern, write one concise fact in plain English.
Return each fact on its own line, prefixed with "FACT: ".
If no meaningful patterns exist, return "NO_PATTERNS".\
"""

_REFLECTION_ANALYSIS_PROMPT = """\
You are analyzing weekly reflection journal entries for a personal chief-of-staff assistant.

Recent reflections (last 4 weeks):
{reflections_text}

Identify recurring themes that appear in at least 2 reflections. Focus on:
- Recurring sources of energy drain
- Persistent avoidance patterns
- Consistent sources of satisfaction or momentum

For each recurring theme, write one concise fact in plain English.
Return each fact on its own line, prefixed with "FACT: ".
If no recurring themes exist across multiple reflections, return "NO_PATTERNS".\
"""

_HONCHO_BASE_URL = "https://api.honcho.dev"
_HONCHO_APP_NAME = "mahler"
_HONCHO_USER_ID = "jai"


def _run_project_analysis(d1: D1Client, env: dict) -> None:
    honcho_api_key = os.environ.get("HONCHO_API_KEY")
    if not honcho_api_key:
        sys.stderr.write(
            "WARNING: HONCHO_API_KEY not set — skipping project analysis\n"
        )
        return
    rows = d1.get_recent_project_log(since_days=7)
    if not rows:
        return
    project_log_text = "\n".join(
        "[{project}] {created_at} — {entry_type}: {summary}".format(**r)
        for r in rows
    )
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    prompt = _PROJECT_ANALYSIS_PROMPT.format(project_log_text=project_log_text)
    raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)
    facts = [
        line[len("FACT: "):].strip()
        for line in raw.splitlines()
        if line.startswith("FACT: ")
    ]
    for fact in facts:
        honcho_client.conclude(
            fact,
            honcho_api_key,
            _HONCHO_BASE_URL,
            _HONCHO_APP_NAME,
            _HONCHO_USER_ID,
        )


def _run_reflection_analysis(d1: D1Client, env: dict) -> None:
    honcho_api_key = os.environ.get("HONCHO_API_KEY")
    if not honcho_api_key:
        sys.stderr.write(
            "WARNING: HONCHO_API_KEY not set — skipping reflection analysis\n"
        )
        return
    rows = d1.get_recent_reflections(since_weeks=4)
    if not rows:
        return
    reflections_text = "\n\n".join(
        "[{week_of}]: {raw_text}".format(**r) for r in rows
    )
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    prompt = _REFLECTION_ANALYSIS_PROMPT.format(reflections_text=reflections_text)
    raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)
    facts = [
        line[len("FACT: "):].strip()
        for line in raw.splitlines()
        if line.startswith("FACT: ")
    ]
    for fact in facts:
        honcho_client.conclude(
            fact,
            honcho_api_key,
            _HONCHO_BASE_URL,
            _HONCHO_APP_NAME,
            _HONCHO_USER_ID,
        )


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
    patterns = d1.get_triage_patterns_with_reply_rate(since_days=since_days, min_count=3)
    if not patterns:
        print("No proposals this week.")
    else:
        priority_map = d1.get_priority_map()
        model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
        patterns_text = "\n".join(
            "- {addr}: {count} times as {cls}, {replies} replies ({rate}%)".format(
                addr=p["from_addr"],
                count=p["occurrence_count"],
                cls=p["classification"],
                replies=p.get("reply_count", 0),
                rate=(
                    p.get("reply_count", 0) * 100 // p["occurrence_count"]
                    if p["occurrence_count"] > 0
                    else 0
                ),
            )
            for p in patterns
        )
        prompt = _PROPOSAL_PROMPT.format(
            priority_map=priority_map, patterns=patterns_text
        )
        raw = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)
        try:
            proposals = json.loads(raw)
            if not isinstance(proposals, list):
                raise ValueError("LLM did not return a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"LLM returned unparseable proposals: {exc}\nRaw: {raw}"
            ) from exc
        if proposals:
            print(json.dumps(proposals))
        else:
            print("No proposals this week.")

    try:
        _run_project_analysis(d1, env)
    except Exception as exc:
        sys.stderr.write(f"WARNING: project analysis failed: {exc}\n")

    try:
        _run_reflection_analysis(d1, env)
    except Exception as exc:
        sys.stderr.write(f"WARNING: reflection analysis failed: {exc}\n")


def _apply(proposal_json: str, env: dict) -> None:
    try:
        proposal = json.loads(proposal_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid proposal JSON: {exc}") from exc

    required_keys = {"sender", "current_tier", "proposed_tier", "evidence"}
    missing = required_keys - proposal.keys()
    if missing:
        raise RuntimeError(f"Proposal missing required keys: {missing}")

    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )
    priority_map = d1.get_priority_map()
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)

    sender = proposal["sender"]
    current_tier = proposal["current_tier"]
    proposed_tier = proposal["proposed_tier"]
    evidence = proposal["evidence"]
    prompt = (
        "You are editing an email classification priority map in markdown format.\n\n"
        f"Current priority map:\n{priority_map}\n\n"
        f'Apply this change: move the sender "{sender}" from {current_tier} to {proposed_tier}.\n'
        f"Evidence: {evidence}\n\n"
        f"If {sender} appears as an example under {current_tier}, move that line to {proposed_tier}.\n"
        f"If it does not appear explicitly, add it as a new example under {proposed_tier}.\n"
        "Return the complete updated priority map as a markdown document. No other text."
    )
    updated_map = _call_llm(prompt, env["OPENROUTER_API_KEY"], model)
    if not updated_map.strip() or "##" not in updated_map:
        raise RuntimeError(
            "LLM returned implausible priority map (missing ## headings) — aborting write"
        )
    d1.set_priority_map(updated_map)
    print(f"Priority map updated. Moved {proposal['sender']} to {proposal['proposed_tier']}.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mahler kaizen reflection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--apply", metavar="PROPOSAL_JSON")
    parser.add_argument("--since-days", type=int, default=7, metavar="N")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _supplement_env_from_hermes()
    args = _parse_args(argv)
    env = _load_env()
    if args.run:
        _run(since_days=args.since_days, env=env)
    else:
        _apply(proposal_json=args.apply, env=env)


if __name__ == "__main__":
    main()
