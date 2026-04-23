import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "query.py"


def run(args, env_overrides=None):
    env = {
        "FINANCE_WORKER_URL": "https://finance.test",
        "FINANCE_BEARER_TOKEN": "test-token",
        "PATH": "/usr/bin:/bin",
    }
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_balances_subcommand_prints_json(monkeypatch, tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"snapshots": [{"account_id": "acc_chk", "current_balance": 3200}]}))

    res = run(["balances", "--mock-from", str(fixture)])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["snapshots"][0]["account_id"] == "acc_chk"


def test_missing_bearer_raises(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}")
    res = run(["balances", "--mock-from", str(fixture)], env_overrides={"FINANCE_BEARER_TOKEN": ""})
    assert res.returncode != 0
    assert "FINANCE_BEARER_TOKEN" in res.stderr
