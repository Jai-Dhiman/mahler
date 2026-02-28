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

        assert "No Trades" in embed["title"]
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
        assert "Equity Correlation Limit" in skip_field["value"]

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
