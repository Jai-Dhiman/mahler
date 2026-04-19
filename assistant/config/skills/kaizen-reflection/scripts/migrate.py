import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from d1_client import D1Client

_REQUIRED_ENV = ["CF_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_API_TOKEN"]


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
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {k: os.environ[k] for k in _REQUIRED_ENV}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed priority_map table in D1")
    parser.add_argument("--file", required=True, metavar="PATH", help="Path to priority-map.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _supplement_env_from_hermes()
    args = _parse_args(argv)
    env = _load_env()

    map_path = Path(args.file).expanduser()
    with open(map_path, "r", encoding="utf-8") as f:
        content = f.read()

    d1 = D1Client(
        account_id=env["CF_ACCOUNT_ID"],
        database_id=env["CF_D1_DATABASE_ID"],
        api_token=env["CF_API_TOKEN"],
    )

    d1.ensure_priority_map_table()

    try:
        d1.get_priority_map()
        raise RuntimeError(
            "priority_map table is already seeded — use reflect.py --apply to make changes"
        )
    except RuntimeError as exc:
        if "priority_map table is empty" not in str(exc):
            raise

    d1.set_priority_map(content)
    print(f"Priority map seeded from {map_path} (version 1).")


if __name__ == "__main__":
    main()
