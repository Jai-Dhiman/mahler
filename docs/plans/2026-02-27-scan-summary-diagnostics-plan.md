# Scan Summary + Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Always send a scan summary notification after the morning scan with enhanced diagnostics (errors, shadow decisions, timing), regardless of whether trades were placed.

**Architecture:** Two files change: `discord.py` gets a new `send_scan_summary()` method that adapts title/color based on trades placed. `morning_scan.py` gets timing instrumentation, shadow decision counters, error tracking, and detailed skip reasons, then always calls the new summary method.

**Tech Stack:** Python, async/await, Discord embeds, `time.time()` for timing

---

### Task 1: Add `send_scan_summary()` to DiscordClient

**Files:**
- Test: `tests/unit/notifications/test_scan_summary.py`
- Create: `tests/unit/notifications/__init__.py`
- Modify: `src/core/notifications/discord.py:1288-1418`

**Step 1: Create test file with tests for `send_scan_summary()`**

```python
"""Tests for send_scan_summary Discord notification."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def discord_client():
    """Create a DiscordClient with mocked HTTP."""
    from core.notifications.discord import DiscordClient

    client = DiscordClient(
        bot_token="test-token",
        public_key="test-key",
        channel_id="test-channel",
    )
    client._request = AsyncMock(return_value={"id": "msg-123"})
    return client


@pytest.fixture
def base_kwargs():
    """Base keyword arguments for send_scan_summary."""
    return {
        "scan_time": "morning",
        "underlyings_scanned": 5,
        "opportunities_found": 140,
        "opportunities_filtered": 8,
        "skip_reasons": {"no_opportunities": 2},
        "market_context": {
            "vix": 26.9,
            "iv_percentile": {"SPY": 83, "QQQ": 73},
            "regime": "bull_high_vol",
            "combined_multiplier": 0.5,
        },
        "underlying_details": {
            "SPY": {"found": 28, "passed": 2, "reason": ""},
            "QQQ": {"found": 45, "passed": 2, "reason": ""},
            "IWM": {"found": 2, "passed": 2, "reason": ""},
            "TLT": {"found": 0, "passed": 0, "reason": "No opportunities"},
            "GLD": {"found": 65, "passed": 2, "reason": ""},
        },
    }


class TestScanSummaryNoTrades:
    """Tests for scan summary when no trades were placed."""

    async def test_no_trades_sends_yellow_embed(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)

        discord_client._request.assert_called_once()
        call_data = discord_client._request.call_args[0][2]
        embed = call_data["embeds"][0]

        assert "No Viable Trades" in embed["title"]
        assert embed["color"] == 0x5865F2  # Blurple (VIX < 30)

    async def test_no_trades_high_vix_red_embed(self, discord_client, base_kwargs):
        base_kwargs["market_context"]["vix"] = 45.0
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)

        call_data = discord_client._request.call_args[0][2]
        embed = call_data["embeds"][0]
        assert embed["color"] == 0xED4245  # Red

    async def test_description_shows_found_and_filtered(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)

        call_data = discord_client._request.call_args[0][2]
        embed = call_data["embeds"][0]
        assert "140" in embed["description"]
        assert "8" in embed["description"]


class TestScanSummaryWithTrades:
    """Tests for scan summary when trades were placed."""

    async def test_trades_placed_sends_green_embed(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=2)

        call_data = discord_client._request.call_args[0][2]
        embed = call_data["embeds"][0]

        assert "2 Trades Placed" in embed["title"]
        assert embed["color"] == 0x57F287  # Green

    async def test_trades_placed_description_includes_approved(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=1)

        call_data = discord_client._request.call_args[0][2]
        embed = call_data["embeds"][0]
        assert "1" in embed["description"]
        assert "approved" in embed["description"].lower()


class TestScanSummaryDiagnostics:
    """Tests for diagnostic fields in scan summary."""

    async def test_shadow_decisions_field(self, discord_client, base_kwargs):
        shadow_stats = {"approve": 2, "skip": 1}
        await discord_client.send_scan_summary(
            **base_kwargs, trades_placed=0, agent_shadow_stats=shadow_stats
        )

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "Agent Decisions (shadow)" in field_names

        shadow_field = next(f for f in fields if f["name"] == "Agent Decisions (shadow)")
        assert "2" in shadow_field["value"]
        assert "1" in shadow_field["value"]

    async def test_errors_field_shown_when_errors_exist(self, discord_client, base_kwargs):
        errors = {"Claude rate limit": 1, "Order placement": 2}
        await discord_client.send_scan_summary(
            **base_kwargs, trades_placed=0, errors=errors
        )

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "Errors" in field_names

    async def test_no_errors_field_when_empty(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(
            **base_kwargs, trades_placed=0, errors={}
        )

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "Errors" not in field_names

    async def test_timing_field(self, discord_client, base_kwargs):
        timing = {
            "total_seconds": 45,
            "per_underlying": {"SPY": 8, "QQQ": 12},
            "agent_avg_seconds": 15,
        }
        await discord_client.send_scan_summary(
            **base_kwargs, trades_placed=0, scan_timing=timing
        )

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "Timing" in field_names

    async def test_skip_reasons_shown(self, discord_client, base_kwargs):
        base_kwargs["skip_reasons"] = {
            "Equity correlation limit (SPY, QQQ)": 2,
            "no_opportunities": 1,
        }
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        skip_field = next(f for f in fields if f["name"] == "Skip Reasons")
        assert "Equity correlation limit" in skip_field["value"]

    async def test_per_underlying_details(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        underlying_field = next(f for f in fields if f["name"] == "Per-Underlying")
        assert "SPY" in underlying_field["value"]
        assert "28 found" in underlying_field["value"]


class TestScanSummaryDefaults:
    """Tests for default parameter behavior."""

    async def test_all_optional_params_default_to_none(self, discord_client, base_kwargs):
        """Calling without optional params should not error."""
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)
        discord_client._request.assert_called_once()

    async def test_no_shadow_stats_omits_field(self, discord_client, base_kwargs):
        await discord_client.send_scan_summary(**base_kwargs, trades_placed=0)

        call_data = discord_client._request.call_args[0][2]
        fields = call_data["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "Agent Decisions (shadow)" not in field_names
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/unit/notifications/test_scan_summary.py -v`
Expected: FAIL -- `send_scan_summary` does not exist yet

**Step 3: Implement `send_scan_summary()` in discord.py**

Add the method after the existing `send_no_trade_notification()` (after line 1418). The method signature:

```python
async def send_scan_summary(
    self,
    scan_time: str,
    underlyings_scanned: int,
    opportunities_found: int,
    opportunities_filtered: int,
    skip_reasons: dict,
    market_context: dict,
    trades_placed: int,
    underlying_details: dict | None = None,
    agent_shadow_stats: dict | None = None,
    scan_timing: dict | None = None,
    errors: dict | None = None,
) -> str:
    """Send scan summary notification. Always called at end of morning scan.

    Adapts title/color based on whether trades were placed.
    """
    vix = market_context.get("vix", 0)

    if trades_placed > 0:
        color = 0x57F287  # Green
        title = f"Morning Scan Complete - {trades_placed} Trade{'s' if trades_placed != 1 else ''} Placed"
        description = (
            f"Found {opportunities_found} opportunities, "
            f"{opportunities_filtered} passed filters, "
            f"{trades_placed} approved and placed."
        )
    else:
        # Same VIX-based coloring as old no-trade notification
        if vix >= 40:
            color = 0xED4245  # Red
        elif vix >= 30:
            color = 0xF97316  # Orange
        else:
            color = 0x5865F2  # Blurple

        if opportunities_found == 0:
            title = f"No Trades - {scan_time.title()} Scan"
            description = (
                f"Scanned {underlyings_scanned} underlyings but found no opportunities "
                "that passed initial screening criteria."
            )
        elif opportunities_filtered == 0:
            title = f"No Trades - {scan_time.title()} Scan"
            description = (
                f"Found {opportunities_found} opportunities across {underlyings_scanned} underlyings, "
                "but none passed the screening filters."
            )
        else:
            title = f"No Trades - {scan_time.title()} Scan"
            description = (
                f"Found {opportunities_found} opportunities, {opportunities_filtered} passed filters, "
                "but none were approved."
            )

    fields = []

    # Market context
    fields.append({
        "name": "VIX",
        "value": f"{vix:.1f}" if vix else "N/A",
        "inline": True,
    })

    regime = market_context.get("regime", "unknown")
    if regime:
        regime_display = regime.replace("_", " ").title()
        fields.append({
            "name": "Market Regime",
            "value": regime_display,
            "inline": True,
        })

    combined_mult = market_context.get("combined_multiplier", 1.0)
    if combined_mult < 1.0:
        fields.append({
            "name": "Size Multiplier",
            "value": f"{combined_mult:.0%}",
            "inline": True,
        })

    iv_percentiles = market_context.get("iv_percentile", {})
    if iv_percentiles:
        iv_text = " | ".join(f"{sym}: {pct:.0f}%" for sym, pct in iv_percentiles.items())
        fields.append({
            "name": "IV Percentile",
            "value": iv_text,
            "inline": False,
        })

    # Skip reasons
    if skip_reasons:
        reasons_text = "\n".join(
            f"- {reason.replace('_', ' ').title()}: {count}"
            for reason, count in skip_reasons.items()
        )
        fields.append({
            "name": "Skip Reasons",
            "value": reasons_text,
            "inline": False,
        })

    # Agent shadow decisions
    if agent_shadow_stats:
        approves = agent_shadow_stats.get("approve", 0)
        skips = agent_shadow_stats.get("skip", 0)
        fields.append({
            "name": "Agent Decisions (shadow)",
            "value": f"Would approve: {approves} | Would skip: {skips}",
            "inline": False,
        })

    # Errors
    if errors:
        errors_text = "\n".join(f"- {err}: {count}" for err, count in errors.items())
        fields.append({
            "name": "Errors",
            "value": errors_text,
            "inline": False,
        })

    # Timing
    if scan_timing:
        total = scan_timing.get("total_seconds", 0)
        per_underlying = scan_timing.get("per_underlying", {})
        agent_avg = scan_timing.get("agent_avg_seconds")

        timing_parts = [f"Total: {total:.0f}s"]
        if per_underlying:
            per_parts = " | ".join(f"{sym}: {secs:.0f}s" for sym, secs in per_underlying.items())
            timing_parts.append(per_parts)
        if agent_avg is not None:
            timing_parts.append(f"Agent pipeline: avg {agent_avg:.0f}s")

        fields.append({
            "name": "Timing",
            "value": "\n".join(timing_parts),
            "inline": False,
        })

    # Per-underlying details
    if underlying_details:
        underlying_text = []
        for sym, details in underlying_details.items():
            found = details.get("found", 0)
            passed = details.get("passed", 0)
            reason = details.get("reason", "")
            if found == 0:
                underlying_text.append(f"**{sym}**: No opportunities")
            elif reason:
                underlying_text.append(f"**{sym}**: {found} found, {passed} passed - {reason}")
            else:
                underlying_text.append(f"**{sym}**: {found} found, {passed} passed")

        if underlying_text:
            fields.append({
                "name": "Per-Underlying",
                "value": "\n".join(underlying_text[:5]),
                "inline": False,
            })

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {
            "text": "No action required - market conditions or filters prevented trades"
            if trades_placed == 0
            else f"Scan complete. {trades_placed} order(s) placed.",
        },
    }

    content_title = (
        f"**{scan_time.title()} Scan Complete - {trades_placed} Trade{'s' if trades_placed != 1 else ''} Placed**"
        if trades_placed > 0
        else f"**{scan_time.title()} Scan Complete - No Viable Trades**"
    )

    return await self.send_message(
        content=content_title,
        embeds=[embed],
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/unit/notifications/test_scan_summary.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tests/unit/notifications/ src/core/notifications/discord.py
git commit -m "Add send_scan_summary() to DiscordClient

Always-send scan summary that adapts based on trades placed.
Supports agent shadow stats, error counts, and timing diagnostics."
```

---

### Task 2: Add timing, shadow tracking, and error counters to morning_scan.py

**Files:**
- Modify: `src/handlers/morning_scan.py`

**Step 1: Add `import time` and timing start at top of `_run_morning_scan()`**

At the top of `_run_morning_scan()` (after line 342), add:

```python
import time
scan_start_time = time.time()
```

**Step 2: Add per-underlying timing around the scan loop**

Inside the `for symbol in UNDERLYINGS:` loop (line 606), wrap the symbol scan body:

```python
# Before the try block at line 610
symbol_start = time.time()
```

After the except block at line 758, add:

```python
# Track per-underlying timing
underlying_results[symbol]["scan_seconds"] = time.time() - symbol_start
```

**Step 3: Add shadow stats and error tracking dicts**

After `underlying_results` init (line 602), add:

```python
agent_shadow_stats: dict[str, int] = {"approve": 0, "skip": 0}
scan_errors: dict[str, int] = {}
agent_pipeline_times: list[float] = []
```

**Step 4: Track shadow decisions in the opportunity loop**

After line 941 (`print(f"Agent shadow: {shadow_decision}..."`), add:

```python
if shadow_decision == "approve":
    agent_shadow_stats["approve"] += 1
else:
    agent_shadow_stats["skip"] += 1
```

**Step 5: Track agent pipeline timing**

Before the `_run_v2_analysis()` call (find where `v2_result` is assigned), wrap it:

```python
agent_start = time.time()
# ... existing _run_v2_analysis call ...
agent_pipeline_times.append(time.time() - agent_start)
```

**Step 6: Track errors in the opportunity loop's except blocks**

At line 1094 (`except Exception as e:` for order placement), add:

```python
scan_errors["Order placement"] = scan_errors.get("Order placement", 0) + 1
```

At line 1097-1098 (`except ClaudeRateLimitError`), add:

```python
scan_errors["Claude rate limit"] = scan_errors.get("Claude rate limit", 0) + 1
```

At line 1101-1104 (general exception), add:

```python
scan_errors["Pipeline error"] = scan_errors.get("Pipeline error", 0) + 1
```

**Step 7: Use detailed skip reason from position sizer**

At line 817-819, change:

```python
if size_result.contracts == 0:
    print(f"Position size is 0 for {spread.underlying}: {size_result.reason}")
    skip_reasons["position_size_zero"] = skip_reasons.get("position_size_zero", 0) + 1
    continue
```

To:

```python
if size_result.contracts == 0:
    reason_key = size_result.reason or "Position size zero"
    # Clean up "Blocked by " prefix for readability
    if reason_key.startswith("Blocked by "):
        reason_key = reason_key[len("Blocked by "):]
    print(f"Position size is 0 for {spread.underlying}: {reason_key}")
    skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
    continue
```

**Step 8: Commit**

```bash
git add src/handlers/morning_scan.py
git commit -m "Add timing, shadow tracking, and error counters to morning scan

Instruments the scan loop with per-underlying timing, agent pipeline
duration, shadow decision counts, and categorized error tracking.
Uses detailed position sizer reason instead of generic 'position_size_zero'."
```

---

### Task 3: Replace `send_no_trade_notification()` call with `send_scan_summary()`

**Files:**
- Modify: `src/handlers/morning_scan.py:1146-1167`

**Step 1: Replace the conditional no-trade notification with always-send summary**

Replace lines 1146-1167:

```python
    # Send "no trade" notification if no trades were approved
    if recommendations_sent == 0:
        try:
            market_context_for_notification = {
                "vix": current_vix,
                "iv_percentile": iv_percentiles,
                "regime": regime_result.get("regime") if regime_result else None,
                "combined_multiplier": combined_size_multiplier,
            }

            await discord.send_no_trade_notification(
                scan_time="morning",
                underlyings_scanned=screening_stats["total_underlyings_scanned"],
                opportunities_found=screening_stats["opportunities_found"],
                opportunities_filtered=screening_stats["opportunities_passed_filters"],
                skip_reasons=skip_reasons,
                market_context=market_context_for_notification,
                underlying_details=underlying_results,
            )
            print("Sent no-trade notification")
        except Exception as e:
            print(f"Error sending no-trade notification: {e}")
```

With:

```python
    # Always send scan summary
    try:
        market_context_for_notification = {
            "vix": current_vix,
            "iv_percentile": iv_percentiles,
            "regime": regime_result.get("regime") if regime_result else None,
            "combined_multiplier": combined_size_multiplier,
        }

        scan_timing = {
            "total_seconds": time.time() - scan_start_time,
            "per_underlying": {
                sym: details.get("scan_seconds", 0)
                for sym, details in underlying_results.items()
            },
            "agent_avg_seconds": (
                sum(agent_pipeline_times) / len(agent_pipeline_times)
                if agent_pipeline_times
                else None
            ),
        }

        await discord.send_scan_summary(
            scan_time="morning",
            underlyings_scanned=screening_stats["total_underlyings_scanned"],
            opportunities_found=screening_stats["opportunities_found"],
            opportunities_filtered=screening_stats["opportunities_passed_filters"],
            skip_reasons=skip_reasons,
            market_context=market_context_for_notification,
            trades_placed=recommendations_sent,
            underlying_details=underlying_results,
            agent_shadow_stats=agent_shadow_stats if any(agent_shadow_stats.values()) else None,
            scan_timing=scan_timing,
            errors=scan_errors if scan_errors else None,
        )
        print(f"Sent scan summary (trades_placed={recommendations_sent})")
    except Exception as e:
        print(f"Error sending scan summary: {e}")
```

**Step 2: Verify the old `send_no_trade_notification()` is no longer called from morning_scan.py**

Run: `cd /Users/jdhiman/Documents/mahler && grep -n "send_no_trade_notification" src/handlers/morning_scan.py`
Expected: No matches (method still exists in discord.py but is no longer called from morning_scan)

**Step 3: Run full test suite**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/ -v --timeout=30`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/handlers/morning_scan.py
git commit -m "Always send scan summary with full diagnostics

Replace conditional send_no_trade_notification with always-send
send_scan_summary that includes timing, shadow decisions, and errors."
```

---

### Task 4: Verify end-to-end via wrangler logs

**Step 1: Deploy to Cloudflare Workers**

Run: `cd /Users/jdhiman/Documents/mahler && npx wrangler deploy`

**Step 2: Check next morning scan via wrangler tail**

Run: `npx wrangler tail` during the morning scan window (10:00 AM ET)

Verify in logs:
- `scan_start_time` is captured
- Per-underlying timing is logged
- Shadow decisions are counted
- Skip reasons use detailed position sizer output
- Scan summary notification fires at end

**Step 3: Verify Discord notification**

Check Discord channel for the scan summary embed. Should show:
- Title reflects trades placed (or no trades)
- VIX, Market Regime, IV Percentile fields present
- Skip Reasons with detailed constraint names
- Timing field with per-underlying breakdown
- Agent Decisions field (if any opportunities reached the pipeline)
- Errors field (only if errors occurred)
