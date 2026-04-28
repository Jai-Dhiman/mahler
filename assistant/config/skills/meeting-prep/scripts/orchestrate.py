"""Meeting prep orchestrator. Invoked by cron every 15 min."""
from __future__ import annotations
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
_SHARED_DIR = Path.home() / ".hermes" / "shared"
for _p in [str(_SCRIPTS_DIR), str(_SHARED_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SKILLS = Path.home() / ".hermes" / "skills"
_DEFAULT_MODEL = "openai/gpt-5-nano"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_TAVILY_URL = "https://api.tavily.com/search"
_GCAL_SKIP = "orchestra,rehearsal,bohemian,jinks,encampment"
_SOCIAL_TITLES = {"1:1", "sync", "standup", "stand-up", "catch up", "catchup", "chat"}


def _load_hermes_env() -> None:
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


def _run(cmd: list[str], runner, timeout: int = 60) -> subprocess.CompletedProcess:
    return runner(cmd, capture_output=True, text=True, timeout=timeout)


# --- Gcal ---

class GcalEvent:
    def __init__(self, event_id: str, start: str, title: str, attendees: list[str], description: str):
        self.event_id = event_id
        self.start = start
        self.title = title
        self.attendees = attendees
        self.description = description


def _parse_gcal_output(output: str) -> GcalEvent | None:
    if "No meetings in window." in output:
        return None
    lines = output.strip().splitlines()
    if not lines:
        return None
    # Format: "{id}  {start}  {summary}"
    parts = lines[0].split("  ", 2)
    if len(parts) < 3:
        return None
    event_id, start, title = parts[0].strip(), parts[1].strip(), parts[2].strip()
    attendees: list[str] = []
    description = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("Attendees:"):
            raw = stripped[len("Attendees:"):].strip()
            attendees = [e.strip() for e in raw.split(",") if e.strip()]
        elif stripped.startswith("Description:"):
            description = stripped[len("Description:"):].strip()
        elif stripped and not description and not stripped.startswith("Attendees:"):
            description = stripped
    return GcalEvent(event_id, start, title, attendees, description)


def fetch_upcoming_event(runner, min_minutes: int = 45, max_minutes: int = 75) -> GcalEvent | None:
    result = _run(
        ["python3", str(_SKILLS / "google-calendar" / "scripts" / "gcal.py"),
         "upcoming", "--min-minutes", str(min_minutes), "--max-minutes", str(max_minutes),
         "--skip-keywords", _GCAL_SKIP],
        runner,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcal.py upcoming failed: {result.stderr.strip()}")
    return _parse_gcal_output(result.stdout)


# --- Dedup ---

def check_dedup(event_id: str, runner) -> bool:
    """Returns True if not yet briefed (proceed), False if already briefed (stop)."""
    result = _run(
        ["python3", str(_SKILLS / "meeting-prep" / "scripts" / "dedup.py"),
         "check", "--event-id", event_id],
        runner,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(f"dedup.py check failed: {result.stderr.strip()}")


def log_dedup(event_id: str, title: str, start: str, runner) -> None:
    result = _run(
        ["python3", str(_SKILLS / "meeting-prep" / "scripts" / "dedup.py"),
         "log", "--event-id", event_id, "--summary", title, "--start-time", start],
        runner,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dedup.py log failed: {result.stderr.strip()}")


# --- Context gathering ---

def fetch_email_context(attendees: list[str], runner) -> str | None:
    owner = os.environ.get("MAHLER_OWNER_EMAIL", "").lower()
    external = [e for e in attendees if e.lower() != owner]
    if not external:
        return None
    result = _run(
        ["python3", str(_SKILLS / "meeting-prep" / "scripts" / "email_context.py"),
         "email-context", "--attendees", ",".join(external)],
        runner,
    )
    if result.returncode != 0:
        return f"(email context error: {result.stderr.strip()[:200]})"
    text = result.stdout.strip()
    if not text or "No recent flagged emails" in text:
        return None
    return text


def fetch_open_tasks(due_date: str, runner) -> list[str]:
    result = _run(
        ["python3", str(_SKILLS / "notion-tasks" / "scripts" / "tasks.py"),
         "list", "--status", "Not started", "--due-before", due_date],
        runner,
    )
    if result.returncode != 0:
        return []
    titles = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "] " in stripped:
            titles.append(stripped[stripped.index("] ") + 2:])
    return titles


def _should_skip_wiki(title: str, description: str) -> bool:
    lower = title.lower().strip()
    if lower in _SOCIAL_TITLES and not description:
        return True
    if not description and re.match(r"^(meeting with|call with|1:1 with|catch up with)\s+\w+\s*\w*$", lower):
        return True
    return False


def _extract_wiki_keywords(title: str, description: str) -> list[str]:
    stop = {
        "meeting", "call", "with", "intro", "catch", "up", "sync", "the",
        "and", "for", "a", "an", "to", "of", "in", "on",
        "interview", "between",
        "location", "size", "vertical", "website", "title", "salary", "equity",
        "people", "role", "about", "new", "york",
    }
    # Prefer description keywords — title often has person names rather than topics
    source = (description + " " + title) if description else title
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9]+\b", source)
    seen: set[str] = set()
    keywords = []
    for w in words:
        lw = w.lower()
        if lw not in stop and lw not in seen and len(lw) > 3:
            seen.add(lw)
            keywords.append(w)
        if len(keywords) == 2:
            break
    return keywords


def fetch_wiki_context(title: str, description: str, runner) -> str | None:
    if _should_skip_wiki(title, description):
        return None
    for kw in _extract_wiki_keywords(title, description):
        search_result = _run(
            ["python3", str(_SKILLS / "notion-wiki" / "scripts" / "wiki.py"),
             "search", "--query", kw],
            runner,
        )
        if search_result.returncode != 0 or "No results." in search_result.stdout:
            continue
        first_line = search_result.stdout.strip().splitlines()[0]
        m = re.match(r"^\[([^\]]+)\]", first_line)
        if not m:
            continue
        page_id = m.group(1)
        read_result = _run(
            ["python3", str(_SKILLS / "notion-wiki" / "scripts" / "wiki.py"),
             "read", "--id", page_id],
            runner,
        )
        if read_result.returncode != 0:
            continue
        body_lines = [
            l for l in read_result.stdout.strip().splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("type:") and not l.startswith("url:")
        ]
        body = "\n".join(body_lines)
        if body:
            return body[:400]
    return None


def fetch_crm_context(attendees: list[str], runner) -> str | None:
    owner = os.environ.get("MAHLER_OWNER_EMAIL", "").lower()
    parts = []
    for email in attendees:
        if email.lower() == owner:
            continue
        local = email.split("@")[0]
        name = " ".join(p.capitalize() for p in re.split(r"[._\-]", local) if p)
        result = _run(
            ["python3", str(_SKILLS / "relationship-manager" / "scripts" / "contacts.py"),
             "summarize", "--name", name],
            runner,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(result.stdout.strip())
    return "\n\n".join(parts) if parts else None


def fetch_honcho_context(attendees: list[str], research_target: str | None) -> str | None:
    """Query Honcho semantic memory for attendees and the company/topic."""
    try:
        import honcho_client
    except ImportError:
        return None
    owner = os.environ.get("MAHLER_OWNER_EMAIL", "").lower()
    queries = []
    for email in attendees:
        if email.lower() == owner:
            continue
        local = email.split("@")[0]
        name = " ".join(p.capitalize() for p in re.split(r"[._\-]", local) if p)
        queries.append(name)
    if research_target:
        queries.append(research_target)
    if not queries:
        return None
    parts = []
    seen: set[str] = set()
    for query in queries[:3]:
        try:
            results = honcho_client.query_conclusions(query)
            for r in results[:2]:
                content = getattr(r, "content", str(r)).strip()
                if content and content not in seen:
                    seen.add(content)
                    parts.append(content)
        except Exception:
            pass
    return "\n".join(parts) if parts else None


# --- Web research ---

def _extract_research_target(description: str) -> str | None:
    """Extract an explicit research target (website domain or company name) from the description."""
    if description:
        m = re.search(r"Website:\s*(https?://)?([^\s|,]+)", description, re.IGNORECASE)
        if m:
            return m.group(2).rstrip("/")
        m = re.search(r"Company:\s*([^\|,\n]+)", description, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _search_tavily(query: str) -> list[dict]:
    api_key = os.environ["TAVILY_API_KEY"]
    body = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": 5,
        "search_depth": "advanced",
    }).encode("utf-8")
    req = urllib.request.Request(
        _TAVILY_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with _OPENER.open(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Tavily error: HTTP {exc.code}") from exc
    return data.get("results", [])


def fetch_web_research(title: str, description: str) -> str | None:
    """Search Tavily for the company/topic and return formatted results."""
    research_target = _extract_research_target(description)
    if research_target:
        query = f"{research_target} company overview product"
    else:
        stop = {"meeting", "call", "with", "intro", "catch", "up", "sync", "interview", "between"}
        words = [w for w in title.split() if w.lower() not in stop]
        query = " ".join(words[:4]) if words else title
    try:
        results = _search_tavily(query)
    except Exception:
        return None
    parts = []
    for r in results[:3]:
        snippet = r.get("content", "")[:300].strip()
        if snippet:
            parts.append(f"{r.get('title', '')}: {snippet}")
    return "\n\n".join(parts) if parts else None


# --- Synthesis ---

def synthesize_brief(
    title: str,
    start: str,
    attendees: list[str],
    description: str,
    emails: str | None,
    tasks: list[str],
    wiki: str | None,
    crm: str | None,
    web_research: str | None,
    honcho: str | None,
    llm_caller,
) -> str:
    prompt = _build_synthesis_prompt(
        title, start, attendees, description, emails, tasks, wiki, crm, web_research, honcho
    )
    raw = llm_caller(prompt)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    bullets = []
    for line in lines[:4]:
        if not line.startswith("•"):
            line = "• " + line.lstrip("-•* ")
        bullets.append(line)
    return "\n".join(bullets) if bullets else "• No specific context available."


def _build_synthesis_prompt(
    title, start, attendees, description, emails, tasks, wiki, crm, web_research, honcho
) -> str:
    sections = [f"Meeting: {title}", f"Start: {start}"]
    if attendees:
        sections.append(f"Attendees: {', '.join(attendees)}")
    if description:
        sections.append(f"Description: {description}")
    if web_research:
        sections.append(f"Web research:\n{web_research}")
    if honcho:
        sections.append(f"Memory about these people/topics:\n{honcho}")
    if crm:
        sections.append(f"CRM context:\n{crm}")
    if emails:
        sections.append(f"Recent emails from attendees:\n{emails}")
    if tasks:
        sections.append("Open tasks:\n" + "\n".join(f"- {t}" for t in tasks[:10]))
    if wiki:
        sections.append(f"Relevant wiki context:\n{wiki}")
    return (
        "You are a chief of staff preparing a pre-meeting brief for Jai Dhiman. "
        "Using the context provided below, write exactly 3-4 bullet points covering "
        "what Jai needs to know and how to prepare. "
        "Each bullet must be ACTIONABLE and SPECIFIC — ground it in the provided context, "
        "especially the description, web research, and memory. "
        "Do NOT rephrase the description. Do NOT make up facts not in the context. "
        "NEVER include personal background, political, immigration, biographical, or "
        "legal information about any individual — only company/product/business context. "
        "Start each bullet with '•'. Output only the bullets, no preamble.\n\n"
        + "\n\n".join(sections)
    )


# --- Post brief ---

def post_brief_to_discord(
    title: str,
    start: str,
    synthesis: str,
    attendees: list[str],
    emails: str | None,
    tasks: list[str],
    wiki: str | None,
    runner,
) -> None:
    cmd = [
        "python3", str(_SKILLS / "meeting-prep" / "scripts" / "post_brief.py"),
        "--title", title,
        "--start", start,
        "--synthesis", synthesis,
    ]
    if attendees:
        cmd += ["--attendees", ", ".join(attendees)]
    if emails:
        cmd += ["--emails", emails]
    if tasks:
        cmd += ["--tasks", "\n".join(tasks[:10])]
    if wiki:
        cmd += ["--wiki", wiki[:500]]
    result = _run(cmd, runner)
    if result.returncode != 0:
        raise RuntimeError(f"post_brief.py failed: {result.stderr.strip()}")
    if "Brief sent." not in result.stdout:
        raise RuntimeError(f"post_brief.py unexpected output: {result.stdout.strip()}")


# --- HTTP ---

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


def _call_openrouter(prompt: str) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
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
    return data["choices"][0]["message"]["content"]


# --- Entry point ---

def run_prep(*, runner, llm_caller, test: bool = False) -> str:
    min_minutes = 0 if test else 45
    max_minutes = 1440 if test else 75
    event = fetch_upcoming_event(runner, min_minutes=min_minutes, max_minutes=max_minutes)
    if event is None:
        return "NO_WORK"

    if not test and not check_dedup(event.event_id, runner):
        return "NO_WORK"

    due_date = event.start[:10] if event.start else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    research_target = _extract_research_target(event.description)

    emails = fetch_email_context(event.attendees, runner)
    tasks = fetch_open_tasks(due_date, runner)
    wiki = fetch_wiki_context(event.title, event.description, runner)
    crm = fetch_crm_context(event.attendees, runner)
    web_research = fetch_web_research(event.title, event.description)
    honcho = fetch_honcho_context(event.attendees, research_target)

    synthesis = synthesize_brief(
        title=event.title,
        start=event.start,
        attendees=event.attendees,
        description=event.description,
        emails=emails,
        tasks=tasks,
        wiki=wiki,
        crm=crm,
        web_research=web_research,
        honcho=honcho,
        llm_caller=llm_caller,
    )

    post_brief_to_discord(
        title=event.title,
        start=event.start,
        synthesis=synthesis,
        attendees=event.attendees,
        emails=emails,
        tasks=tasks,
        wiki=wiki,
        runner=runner,
    )

    log_dedup(event.event_id, event.title, event.start, runner)
    return f"Brief sent: {event.title}"


def cli_main() -> int:
    import argparse
    _load_hermes_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Look 24h ahead, skip dedup — for manually triggering a test brief")
    args = parser.parse_args()
    try:
        result = run_prep(runner=subprocess.run, llm_caller=_call_openrouter, test=args.test)
        print(result)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
