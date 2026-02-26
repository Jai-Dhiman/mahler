"""Configurable thresholds for strategy monitoring alerts.

All thresholds are derived from backtest findings in:
analysis/walkforward_findings_2026-01-30.log

These can be overridden via environment variables or config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class IVEnvironmentThresholds:
    """Thresholds for IV environment alerts.

    IV Percentile measures where current IV sits relative to the past year.
    More reliable than IV Rank because it considers all 252 trading days.
    """

    low_iv_threshold: float = 30.0  # IV percentile below this is "low"
    low_iv_consecutive_days: int = 5  # Days below threshold before alert
    elevated_iv_threshold: float = 70.0  # IV percentile above this is "elevated"
    crisis_iv_threshold: float = 90.0  # IV percentile above this is "crisis" (VIX > 30 equivalent)


@dataclass
class PerformanceThresholds:
    """Thresholds for performance deviation alerts.

    Based on QQQ backtest baseline (2007-2025):
    - Win Rate: 69.9% (optimal config), 67.9% (baseline)
    - Profit Factor: 6.10 (optimal), 3.16 (baseline)
    - Max Drawdown: 4.35% (optimal), 4.10% (baseline)
    """

    # Win rate thresholds (lowered for multi-ticker including IWM)
    expected_win_rate: float = 65.0  # Backtest baseline %
    win_rate_warning_threshold: float = 60.0  # Alert if rolling win rate drops below
    win_rate_lookback_trades: int = 20  # Number of recent trades to evaluate

    # Profit factor thresholds (lowered for multi-ticker aggregate)
    expected_profit_factor: float = 4.0  # Backtest aggregate
    profit_factor_warning_threshold: float = 2.0  # Alert if drops below
    profit_factor_lookback_trades: int = 20

    # Drawdown thresholds
    backtest_max_drawdown: float = 5.0  # Backtest max DD % (raised for multi-ticker)
    drawdown_warning_threshold: float = 5.0  # Alert at this level
    drawdown_critical_threshold: float = 10.0  # Critical alert level


@dataclass
class SlippageThresholds:
    """Thresholds for slippage quality alerts.

    ORATS methodology uses 66% slippage for 2-leg spreads.
    Strategy profitability degrades significantly above 85% slippage.
    """

    expected_slippage: float = 66.0  # ORATS 2-leg assumption %
    warning_slippage: float = 75.0  # Alert if average exceeds
    critical_slippage: float = 85.0  # Strategy breaks at this level
    lookback_trades: int = 10  # Number of recent trades to evaluate


@dataclass
class RegimeThresholds:
    """Thresholds for regime change alerts.

    VIX-based regime classification per circuit_breaker.py.
    """

    vix_elevated: float = 20.0  # Start of elevated regime
    vix_caution: float = 30.0  # Caution regime
    vix_high: float = 40.0  # High volatility regime
    vix_crisis: float = 50.0  # Crisis regime (halt new trades)


@dataclass
class CooldownConfig:
    """Rate limiting configuration for alerts.

    Prevents spam by enforcing minimum time between similar alerts.
    """

    iv_alert_cooldown_hours: float = 24.0  # Once per day for IV alerts
    performance_alert_cooldown_hours: float = 4.0  # More frequent for performance
    slippage_alert_cooldown_hours: float = 1.0  # Per-trade granularity
    regime_alert_cooldown_hours: float = 1.0  # Important, more frequent
    strategy_switch_cooldown_hours: float = 24.0  # Daily recommendation


@dataclass
class AlertThresholds:
    """Aggregated alert thresholds for strategy monitoring.

    All values can be overridden via environment variables with prefix MAHLER_.
    Example: MAHLER_WIN_RATE_WARNING=55 to set win_rate_warning_threshold to 55%.
    """

    iv: IVEnvironmentThresholds = field(default_factory=IVEnvironmentThresholds)
    performance: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    slippage: SlippageThresholds = field(default_factory=SlippageThresholds)
    regime: RegimeThresholds = field(default_factory=RegimeThresholds)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)

    @classmethod
    def from_env(cls) -> "AlertThresholds":
        """Load thresholds from environment variables with defaults."""
        thresholds = cls()

        # IV thresholds
        if val := os.environ.get("MAHLER_LOW_IV_THRESHOLD"):
            thresholds.iv.low_iv_threshold = float(val)
        if val := os.environ.get("MAHLER_LOW_IV_DAYS"):
            thresholds.iv.low_iv_consecutive_days = int(val)
        if val := os.environ.get("MAHLER_ELEVATED_IV_THRESHOLD"):
            thresholds.iv.elevated_iv_threshold = float(val)
        if val := os.environ.get("MAHLER_CRISIS_IV_THRESHOLD"):
            thresholds.iv.crisis_iv_threshold = float(val)

        # Performance thresholds
        if val := os.environ.get("MAHLER_EXPECTED_WIN_RATE"):
            thresholds.performance.expected_win_rate = float(val)
        if val := os.environ.get("MAHLER_WIN_RATE_WARNING"):
            thresholds.performance.win_rate_warning_threshold = float(val)
        if val := os.environ.get("MAHLER_EXPECTED_PROFIT_FACTOR"):
            thresholds.performance.expected_profit_factor = float(val)
        if val := os.environ.get("MAHLER_PROFIT_FACTOR_WARNING"):
            thresholds.performance.profit_factor_warning_threshold = float(val)
        if val := os.environ.get("MAHLER_DRAWDOWN_WARNING"):
            thresholds.performance.drawdown_warning_threshold = float(val)
        if val := os.environ.get("MAHLER_DRAWDOWN_CRITICAL"):
            thresholds.performance.drawdown_critical_threshold = float(val)

        # Slippage thresholds
        if val := os.environ.get("MAHLER_EXPECTED_SLIPPAGE"):
            thresholds.slippage.expected_slippage = float(val)
        if val := os.environ.get("MAHLER_SLIPPAGE_WARNING"):
            thresholds.slippage.warning_slippage = float(val)
        if val := os.environ.get("MAHLER_SLIPPAGE_CRITICAL"):
            thresholds.slippage.critical_slippage = float(val)

        # Cooldown config
        if val := os.environ.get("MAHLER_IV_ALERT_COOLDOWN"):
            thresholds.cooldown.iv_alert_cooldown_hours = float(val)
        if val := os.environ.get("MAHLER_PERF_ALERT_COOLDOWN"):
            thresholds.cooldown.performance_alert_cooldown_hours = float(val)

        return thresholds
