import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_BLOCKER_KEYWORDS = [
    "stuck",
    "blocked",
    "can't figure out",
    "cannot figure out",
    "issue is",
    "problem is",
    "broken",
    "failing",
    "doesn't work",
    "not working",
    "frustrat",
]


def _load_mahler_env() -> None:
    env_path = Path.home() / ".mahler.env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _get_d1_client():
    _load_mahler_env()
    email_triage_scripts = str(
        Path(__file__).parent.parent
        / "config" / "skills" / "email-triage" / "scripts"
    )
    if email_triage_scripts not in sys.path:
        sys.path.insert(0, email_triage_scripts)
    from d1_client import D1Client
    return D1Client(
        account_id=os.environ["CF_ACCOUNT_ID"],
        database_id=os.environ["CF_D1_DATABASE_ID"],
        api_token=os.environ["CF_API_TOKEN"],
    )


def _derive_project_name(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            name = url.rstrip("/").replace(".git", "").rsplit("/", 1)[-1]
            if name:
                return name
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return os.path.basename(cwd.rstrip("/"))


def _derive_git_ref(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def _derive_branch(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def _scan_for_keywords(transcript: dict) -> bool:
    messages = transcript.get("messages", transcript.get("transcript", []))
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if any(kw in content.lower() for kw in _BLOCKER_KEYWORDS):
            return True
    return False


def _call_openrouter(transcript: dict, api_key: str, model: str) -> str:
    import ssl
    import urllib.request

    messages = transcript.get("messages", transcript.get("transcript", []))
    user_texts = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        user_texts.append(content)

    excerpt = "\n".join(f"User: {t}" for t in user_texts[-10:])

    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    body = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are extracting a concise blocker summary from a development session. "
                    "Return exactly 1-2 sentences describing the main technical blocker or problem "
                    "the developer is stuck on. Be specific. "
                    "If no clear blocker exists, return an empty string."
                ),
            },
            {"role": "user", "content": excerpt},
        ],
        "max_tokens": 150,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    with opener.open(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()


def log_session_heartbeat(cwd: str) -> None:
    project = _derive_project_name(cwd)
    git_ref = _derive_git_ref(cwd)
    branch = _derive_branch(cwd)
    client = _get_d1_client()
    client.ensure_tables()
    client.insert_session_heartbeat(project, git_ref, branch)


def log_win(project: str, summary: str, git_ref: str) -> None:
    client = _get_d1_client()
    client.insert_project_log(project, "win", summary, git_ref)


def log_blocker_if_triggered(transcript: dict, cwd: str) -> None:
    if not _scan_for_keywords(transcript):
        return
    _load_mahler_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = "x-ai/grok-4.1-fast"
    summary = _call_openrouter(transcript, api_key, model)
    if not summary:
        return
    project = _derive_project_name(cwd)
    git_ref = _derive_git_ref(cwd)
    client = _get_d1_client()
    client.insert_project_log(project, "blocker", summary, git_ref)


_INSERT_LOCAL_CAPTURE = (
    "INSERT OR IGNORE INTO local_capture "
    "(source, project, content, content_hash) VALUES (?, ?, ?, ?)"
)

_CREATE_LOCAL_CAPTURE_DDL = """
CREATE TABLE IF NOT EXISTS local_capture (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL CHECK(source IN ('memory','git')),
  project TEXT,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  captured_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _ensure_local_capture(d1) -> None:
    d1.query(_CREATE_LOCAL_CAPTURE_DDL, [])
    d1.query(
        "CREATE INDEX IF NOT EXISTS idx_local_capture_recent ON local_capture(captured_at)",
        [],
    )


def _sync_memory_dir(d1, memory_dir: Path) -> None:
    if not memory_dir.is_dir():
        return
    for md_file in sorted(memory_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError:
            continue
        content_hash = hashlib.sha256(
            f"memory:{md_file.name}:{content}".encode("utf-8")
        ).hexdigest()
        body = f"# {md_file.name}\n{content}"
        d1.query(_INSERT_LOCAL_CAPTURE, ["memory", md_file.name, body, content_hash])


def _sync_git_recent(d1, repos_root: Path) -> None:
    if not repos_root.is_dir():
        return
    for repo_dir in sorted(repos_root.iterdir()):
        if not (repo_dir / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "log", "--since=24.hours.ago",
                 "--pretty=format:%h %s"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            content = f"[{repo_dir.name}] {line}"
            content_hash = hashlib.sha256(
                f"git:{repo_dir.name}:{line}".encode("utf-8")
            ).hexdigest()
            d1.query(_INSERT_LOCAL_CAPTURE, ["git", repo_dir.name, content, content_hash])


def sync_local_to_d1(memory_dir: Path, repos_root: Path) -> None:
    d1 = _get_d1_client()
    _ensure_local_capture(d1)
    try:
        _sync_memory_dir(d1, memory_dir)
    except Exception as exc:
        print(f"sync_local_to_d1 memory error: {exc}", file=sys.stderr)
    try:
        _sync_git_recent(d1, repos_root)
    except Exception as exc:
        print(f"sync_local_to_d1 git error: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log project activity to D1")
    subparsers = parser.add_subparsers(dest="mode")

    subparsers.add_parser("blocker")
    subparsers.add_parser("stop")

    win_parser = subparsers.add_parser("win")
    win_parser.add_argument("--project", required=True)
    win_parser.add_argument("--summary", required=True)
    win_parser.add_argument("--git-ref", default="")

    args = parser.parse_args()

    if args.mode == "stop":
        try:
            data = json.loads(sys.stdin.read())
            cwd = data.get("cwd", os.getcwd())
            log_blocker_if_triggered(data, cwd)
            log_session_heartbeat(cwd)
        except Exception as exc:
            print(f"project_log stop error: {exc}", file=sys.stderr)
        sys.exit(0)

    elif args.mode == "blocker":
        try:
            data = json.loads(sys.stdin.read())
            cwd = data.get("cwd", os.getcwd())
            log_blocker_if_triggered(data, cwd)
        except Exception as exc:
            print(f"project_log blocker error: {exc}", file=sys.stderr)
        sys.exit(0)

    elif args.mode == "win":
        try:
            log_win(args.project, args.summary, args.git_ref)
        except Exception as exc:
            print(f"project_log win error: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
