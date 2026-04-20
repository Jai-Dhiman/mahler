import argparse
import os
from pathlib import Path

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
        raise NotImplementedError("--record not yet implemented")


if __name__ == "__main__":
    main()
