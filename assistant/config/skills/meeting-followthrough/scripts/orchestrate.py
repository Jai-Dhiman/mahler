"""Meeting follow-through orchestrator. Invoked by cron every 15 min."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path


_DEFAULT_MODEL = "openai/gpt-5-nano"


def _load_hermes_env() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    if not hermes_env.exists():
        print(f"WARNING: hermes env file not found: {hermes_env}", file=sys.stderr)
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


def process_meeting(row, *, runner, llm_caller, discord_poster, d1_client) -> str:
    title = row["title"]
    action_lines = "  None"
    crm_line = "CRM updated: No CRM matches"
    summary = (
        f"Post-meeting: {title}\n"
        f"Action items created:\n"
        f"{action_lines}\n"
        f"{crm_line}"
    )
    discord_poster(summary)
    d1_client.mark_done(row["recording_id"])
    return summary


def main(argv, *, d1_client, runner, llm_caller, discord_poster) -> int:
    rows = d1_client.fetch_pending()
    if not rows:
        print("NO_WORK")
        return 0
    return 0
