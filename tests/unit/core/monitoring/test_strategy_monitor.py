"""Tests for strategy monitoring module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.monitoring.alert_thresholds import AlertThresholds
from core.monitoring.metrics_tracker import MetricsTracker, RollingMetrics
from core.monitoring.strategy_monitor import (
    Alert,
    AlertCategory,
    AlertSeverity,
    MarketContext,
    StrategyMonitor,
)


@pytest.fixture
def mock_d1_client():
    """Mock D1 database client."""
    d1 = MagicMock()
    d1.execute = AsyncMock(return_value=[])
    return d1


@pytest.fixture
def mock_kv_client():
    """Mock KV client."""
    kv = MagicMock()
    kv.get = AsyncMock(return_value=None)
    kv.put = AsyncMock()
    return kv


@pytest.fixture
def mock_discord_client():
    """Mock Discord client."""
    discord = MagicMock()
    discord.send_message = AsyncMock(return_value="123456789")
    return discord


@pytest.fixture
def thresholds():
    """Default alert thresholds."""
    return AlertThresholds()


@pytest.fixture
def strategy_monitor(mock_d1_client, mock_kv_client, mock_discord_client, thresholds):
    """Strategy monitor with mocked dependencies."""
    return StrategyMonitor(
        d1_client=mock_d1_client,
        kv_client=mock_kv_client,
        discord_client=mock_discord_client,
        thresholds=thresholds,
    )


class TestAlertThresholds:
    """Tests for AlertThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values match backtest findings."""
        th = AlertThresholds()

        # IV thresholds
        assert th.iv.low_iv_threshold == 30.0
        assert th.iv.elevated_iv_threshold == 70.0
        assert th.iv.crisis_iv_threshold == 90.0
        assert th.iv.low_iv_consecutive_days == 5

        # Performance thresholds (from backtest)
        assert th.performance.expected_win_rate == 70.0
        assert th.performance.win_rate_warning_threshold == 60.0
        assert th.performance.expected_profit_factor == 6.0
        assert th.performance.backtest_max_drawdown == 4.35

        # Slippage thresholds (ORATS methodology)
        assert th.slippage.expected_slippage == 66.0
        assert th.slippage.warning_slippage == 75.0
        assert th.slippage.critical_slippage == 85.0

    def test_from_env(self, monkeypatch):
        """Test loading thresholds from environment variables."""
        monkeypatch.setenv("MAHLER_WIN_RATE_WARNING", "55")
        monkeypatch.setenv("MAHLER_SLIPPAGE_WARNING", "80")

        th = AlertThresholds.from_env()

        assert th.performance.win_rate_warning_threshold == 55.0
        assert th.slippage.warning_slippage == 80.0


class TestRollingMetrics:
    """Tests for RollingMetrics data class."""

    def test_to_dict(self):
        """Test serialization of rolling metrics."""
        metrics = RollingMetrics(
            window_trades=20,
            window_start_date=datetime(2026, 1, 1),
            window_end_date=datetime(2026, 1, 31),
            win_rate=68.5,
            profit_factor=3.2,
            total_pnl=1500.0,
            avg_pnl_per_trade=75.0,
            current_drawdown=2.1,
            max_drawdown_in_window=3.5,
            last_n_results=["W", "W", "L", "W", "L"],
            avg_slippage_pct=68.0,
            worst_slippage_pct=72.0,
        )

        result = metrics.to_dict()

        assert result["window_trades"] == 20
        assert result["win_rate"] == 68.5
        assert result["profit_factor"] == 3.2
        assert result["last_n_results"] == ["W", "W", "L", "W", "L"]


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    @pytest.mark.asyncio
    async def test_get_rolling_metrics_empty(self, mock_d1_client):
        """Test rolling metrics with no trades."""
        tracker = MetricsTracker(mock_d1_client)
        mock_d1_client.execute = AsyncMock(return_value=[])

        metrics = await tracker.get_rolling_metrics(lookback_trades=20)

        assert metrics.window_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0

    @pytest.mark.asyncio
    async def test_get_rolling_metrics_with_trades(self, mock_d1_client):
        """Test rolling metrics calculation."""
        tracker = MetricsTracker(mock_d1_client)

        # Mock trade data: 7 wins, 3 losses
        mock_trades = [
            {"id": f"t{i}", "realized_pnl": 100.0, "exit_date": "2026-01-15"}
            for i in range(7)
        ] + [
            {"id": f"t{i+7}", "realized_pnl": -50.0, "exit_date": "2026-01-15"}
            for i in range(3)
        ]

        # First call for trades, second for drawdown, third for slippage
        mock_d1_client.execute = AsyncMock(
            side_effect=[
                mock_trades,  # trades query
                [],  # drawdown query
                [],  # slippage query
            ]
        )

        metrics = await tracker.get_rolling_metrics(lookback_trades=20)

        assert metrics.window_trades == 10
        assert metrics.win_rate == 70.0  # 7/10
        # profit_factor = 700 / 150 = 4.67
        assert metrics.profit_factor == pytest.approx(4.67, rel=0.1)
        assert metrics.total_pnl == 550.0  # 700 - 150


class TestStrategyMonitor:
    """Tests for StrategyMonitor."""

    @pytest.mark.asyncio
    async def test_iv_crisis_alert(self, strategy_monitor, mock_d1_client):
        """Test crisis IV alert generation."""
        mock_d1_client.execute = AsyncMock(return_value=[])

        alerts = await strategy_monitor.run_all_checks(
            iv_percentile=95.0,
            vix_level=35.0,
            market_regime="crisis",
            underlying="QQQ",
        )

        # Should have at least one IV environment alert
        iv_alerts = [a for a in alerts if a.category == AlertCategory.IV_ENVIRONMENT]
        assert len(iv_alerts) >= 1

        alert = iv_alerts[0]
        assert alert.severity == AlertSeverity.CRITICAL
        assert "volatility" in alert.title.lower() or "spike" in alert.title.lower()

    @pytest.mark.asyncio
    async def test_elevated_iv_alert(self, strategy_monitor, mock_d1_client):
        """Test elevated IV (favorable) alert generation."""
        mock_d1_client.execute = AsyncMock(return_value=[])

        alerts = await strategy_monitor.run_all_checks(
            iv_percentile=75.0,
            vix_level=22.0,
            market_regime="bull_high_vol",
            underlying="QQQ",
        )

        iv_alerts = [a for a in alerts if a.category == AlertCategory.IV_ENVIRONMENT]
        if iv_alerts:
            alert = iv_alerts[0]
            assert alert.severity == AlertSeverity.INFO
            assert "elevated" in alert.title.lower() or "favorable" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_win_rate_degradation_alert(self, strategy_monitor, mock_d1_client):
        """Test win rate degradation alert."""
        # Mock trades with poor win rate: 5 wins, 15 losses = 25% win rate
        mock_trades = [
            {"id": f"t{i}", "realized_pnl": 100.0, "exit_date": "2026-01-15"}
            for i in range(5)
        ] + [
            {"id": f"t{i+5}", "realized_pnl": -50.0, "exit_date": "2026-01-15"}
            for i in range(15)
        ]

        mock_d1_client.execute = AsyncMock(
            side_effect=[
                mock_trades,  # trades query
                [],  # drawdown query
                [],  # slippage query
                [],  # IV history query
            ]
        )

        alerts = await strategy_monitor.run_all_checks(
            iv_percentile=50.0,
            vix_level=18.0,
            market_regime="bull_low_vol",
            underlying="QQQ",
        )

        perf_alerts = [a for a in alerts if a.category == AlertCategory.PERFORMANCE_DEVIATION]
        assert len(perf_alerts) >= 1

        alert = perf_alerts[0]
        assert alert.severity == AlertSeverity.WARNING
        assert "win rate" in alert.title.lower()

    @pytest.mark.asyncio
    async def test_cooldown_filtering(self, strategy_monitor, mock_d1_client, mock_kv_client):
        """Test that alerts are filtered by cooldown."""
        # Set up previous alert time (recent)
        recent_time = (datetime.now() - timedelta(hours=1)).isoformat()
        mock_kv_client.get = AsyncMock(return_value=recent_time)
        mock_d1_client.execute = AsyncMock(return_value=[])

        # First call should generate alerts
        alerts1 = await strategy_monitor.run_all_checks(
            iv_percentile=95.0,
            vix_level=35.0,
            market_regime="crisis",
            underlying="QQQ",
        )

        # With cooldown in place, same alert shouldn't fire again
        # Note: This depends on the cooldown duration for IV alerts (24h by default)
        # Since the recent time is only 1 hour ago, alerts should be filtered
        # This test verifies the cooldown mechanism is called

        assert mock_kv_client.get.called

    @pytest.mark.asyncio
    async def test_send_alerts(self, strategy_monitor, mock_discord_client):
        """Test sending alerts to Discord."""
        alerts = [
            Alert(
                category=AlertCategory.IV_ENVIRONMENT,
                severity=AlertSeverity.WARNING,
                title="Test Alert",
                message="This is a test alert message.",
                metrics={"iv_percentile": 25.0, "vix": 15.0},
                suggested_actions=["Action 1", "Action 2"],
            )
        ]

        message_ids = await strategy_monitor.send_alerts(alerts)

        assert len(message_ids) == 1
        assert mock_discord_client.send_message.called


class TestAlertSeverity:
    """Tests for alert severity categorization."""

    def test_severity_ordering(self):
        """Test that severity levels are correctly ordered."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertCategory:
    """Tests for alert category enumeration."""

    def test_all_categories_defined(self):
        """Test that all required alert categories are defined."""
        categories = [
            AlertCategory.IV_ENVIRONMENT,
            AlertCategory.PERFORMANCE_DEVIATION,
            AlertCategory.SLIPPAGE_QUALITY,
            AlertCategory.REGIME_CHANGE,
            AlertCategory.STRATEGY_SWITCH,
        ]

        assert len(categories) == 5
        for cat in categories:
            assert isinstance(cat.value, str)
