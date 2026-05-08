import json
import subprocess
import sys
import unittest
from pathlib import Path

_ENTRYPOINT = Path(__file__).parent.parent / "entrypoint.sh"

_NEXT_RUN_HELPER = """
import json, sys
from datetime import datetime, timezone, timedelta

def next_run_for(cron_expr, now):
    parts = cron_expr.strip().split()
    minute_field, hour_field, dow_field = parts[0], parts[1], parts[4]
    base = now.replace(second=0, microsecond=0)
    if minute_field.startswith('*/') and hour_field == '*':
        interval = int(minute_field[2:])
        delta = interval - (base.minute % interval)
        return base + timedelta(minutes=delta)
    elif minute_field == '0' and hour_field.isdigit():
        h = int(hour_field)
        candidate = base.replace(hour=h, minute=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        if dow_field != '*':
            first_day = int(dow_field.split('-')[0])
            py_target = (first_day + 6) % 7
            while candidate.weekday() != py_target:
                candidate += timedelta(days=1)
        return candidate
    else:
        return base + timedelta(minutes=1)

cron_expr, iso_now = sys.argv[1], sys.argv[2]
now = datetime.fromisoformat(iso_now)
result = next_run_for(cron_expr, now)
print(json.dumps({"weekday": result.weekday(), "hour": result.hour, "minute": result.minute}))
"""


def _next_run(cron_expr: str, iso_now: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _NEXT_RUN_HELPER, cron_expr, iso_now],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        raise AssertionError(f"next_run_for crashed:\n{proc.stderr}")
    return json.loads(proc.stdout)


class TestEntrypointSynthesisBriefCron(unittest.TestCase):
    def test_registers_synthesis_brief_cron(self):
        text = _ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn("'synthesis-brief'", text)
        self.assertIn("synthesize.py --run", text)
        self.assertIn("'0 13 * * 1-5'", text)

    def test_entrypoint_uses_split_for_dow_range(self):
        text = _ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn("dow_field.split('-')", text,
                      "next_run_for must handle range dow fields like '1-5' via split")

    def test_next_run_for_weekday_range_does_not_crash(self):
        # Friday 10:00 UTC — next Monday at 13:00
        result = _next_run("0 13 * * 1-5", "2026-05-08T10:00:00+00:00")
        self.assertEqual(result["hour"], 13)
        self.assertEqual(result["minute"], 0)
        self.assertEqual(result["weekday"], 0)  # Monday

    def test_next_run_for_same_day_before_run_time(self):
        # Monday 08:00 UTC — same day at 13:00
        result = _next_run("0 13 * * 1-5", "2026-05-11T08:00:00+00:00")
        self.assertEqual(result["weekday"], 0)  # Monday
        self.assertEqual(result["hour"], 13)


if __name__ == "__main__":
    unittest.main()
