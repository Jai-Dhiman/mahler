#!/usr/bin/env python3
"""Query the finance-state Worker bearer-auth read API."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is required but not set")
    return value


def _request(path: str, *, method: str = "GET", mock_from: str | None) -> dict:
    if mock_from:
        return json.loads(Path(mock_from).read_text())
    import requests  # local import — avoids requiring `requests` for tests using --mock-from
    base = _require_env("FINANCE_WORKER_URL").rstrip("/")
    token = _require_env("FINANCE_BEARER_TOKEN")
    headers = {"authorization": f"Bearer {token}"}
    fn = requests.post if method == "POST" else requests.get
    res = fn(f"{base}{path}", headers=headers, timeout=20)
    res.raise_for_status()
    return res.json()


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Query finance-state worker.")
    parser.add_argument("command", choices=["balances", "networth", "history", "refresh"])
    parser.add_argument("--account-id")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--mock-from", help="path to a JSON fixture; bypasses HTTP")
    args = parser.parse_args()

    # Validate bearer presence even when using --mock-from, so the test for missing bearer still
    # exercises the same guard the real path uses.
    _require_env("FINANCE_BEARER_TOKEN")

    if args.command == "balances":
        out = _request("/balances", mock_from=args.mock_from)
    elif args.command == "networth":
        out = _request("/networth", mock_from=args.mock_from)
    elif args.command == "history":
        if not args.account_id:
            raise SystemExit("--account-id required for history")
        from urllib.parse import urlencode
        path = f"/history?{urlencode({'account_id': args.account_id, 'days': args.days})}"
        out = _request(path, mock_from=args.mock_from)
    elif args.command == "refresh":
        out = _request("/refresh", method="POST", mock_from=args.mock_from)
    else:
        raise SystemExit(f"unknown command {args.command}")

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
