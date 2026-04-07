"""Strategy monitoring with intelligent alerts.

Monitors strategy performance against backtest expectations and triggers
Discord notifications when action may be needed.

Backtest reference: analysis/walkforward_findings_2026-01-30.log
Key expectations:
- Win Rate: 70% (optimal 69.9%)
- Profit Factor: 6.0 (optimal 6.10)
- Max Drawdown: 4.35%
- Slippage: 66% (ORATS 2-leg)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from core.monitoring.alert_thresholds import AlertThresholds
from core.monitoring.metrics_tracker import MetricsTracker, RollingMetrics

if TYPE_CHECKING:
    from core.db.d1 import D1Client
    from core.db.kv import KVClient
    from core.notifications.discord import DiscordClient


class AlertCategory(str, Enum):
    """Categories of strategy monitoring alerts."""

    IV_ENVIRONMENT = "iv_environment"
    PERFORMANCE_DEVIATION = "performance_deviation"
    SLIPPAGE_QUALITY = "slippage_quality"
    REGIME_CHANGE = "regime_change"
    STRATEGY_SWITCH = "strategy_switch"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Attention needed, consider action
    CRITICAL = "critical"  # Immediate action recommended


@dataclass
class Alert:
    """A strategy monitoring alert."""

    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    metrics: dict | None = None
    suggested_actions: list[str] | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketContext:
    """Current market context for alerts."""

    iv_percentile: float
    vix_level: float
    market_regime: str
    underlying: str


class StrategyMonitor:
    """Monitors strategy performance and generates alerts.

    Compares live performance against backtest expectations and triggers
    Discord notifications when deviations exceed configured thresholds.
    """

    def __init__(
        self,
        d1_client: D1Client,
        kv_client: KVClient,
        discord_client: DiscordClient,
        thresholds: AlertThresholds | None = None,
    ):
        self.d1 = d1_client
        self.kv = kv_client
        self.discord = discord_client
        self.thresholds = thresholds or AlertThresholds.from_env()
        self.metrics_tracker = MetricsTracker(d1_client)

    async def run_all_checks(
        self,
        iv_percentile: float,
        vix_level: float,
        market_regime: str,
        underlying: str = "QQQ",
    ) -> list[Alert]:
        """Run all monitoring checks and return any alerts.

        Args:
            iv_percentile: Current IV percentile (0-100)
            vix_level: Current VIX level
            market_regime: Current regime classification
            underlying: Primary underlying being monitored

        Returns:
            List of Alert objects for any triggered conditions
        """
        context = MarketContext(
            iv_percentile=iv_percentile,
            vix_level=vix_level,
            market_regime=market_regime,
            underlying=underlying,
        )

        alerts: list[Alert] = []

        # Get rolling metrics
        metrics = await self.metrics_tracker.get_rolling_metrics(
            lookback_trades=self.thresholds.performance.win_rate_lookback_trades,
            underlying=underlying,
        )

        # Run each check category
        if alert := await self._check_iv_environment(context):
            alerts.append(alert)

        if alert := await self._check_performance_deviation(metrics, context):
            alerts.append(alert)

        if alert := await self._check_slippage_quality(metrics, context):
            alerts.append(alert)

        if alert := await self._check_regime_change(context):
            alerts.append(alert)

        # Strategy switch recommendations (combines multiple signals)
        if alert := await self._check_strategy_switch(metrics, context):
            alerts.append(alert)

        # Filter alerts by cooldown
        filtered_alerts = await self._filter_by_cooldown(alerts)

        return filtered_alerts

    async def _check_iv_environment(self, context: MarketContext) -> Alert | None:
        """Check IV environment and generate alerts.

        Triggers:
        - Low IV (< 30th percentile) for 5+ consecutive days
        - Elevated IV (> 70th percentile)
        - Crisis IV (> 90th percentile, VIX > 30 equivalent)
        """
        th = self.thresholds.iv

        # Crisis IV spike
        if context.iv_percentile >= th.crisis_iv_threshold:
            return Alert(
                category=AlertCategory.IV_ENVIRONMENT,
                severity=AlertSeverity.CRITICAL,
                title="High Volatility Spike",
                message=f"IV at {context.iv_percentile:.0f}th percentile (VIX: {context.vix_level:.1f}). "
                "Review position sizes and consider defensive adjustments.",
                metrics={
                    "iv_percentile": context.iv_percentile,
                    "vix": context.vix_level,
                    "threshold": th.crisis_iv_threshold,
                },
                suggested_actions=[
                    "Review open positions for potential early exit",
                    "Reduce position sizes on new entries",
                    "Consider pausing new entries until volatility stabilizes",
                ],
            )

        # Elevated IV (favorable)
        if context.iv_percentile >= th.elevated_iv_threshold:
            return Alert(
                category=AlertCategory.IV_ENVIRONMENT,
                severity=AlertSeverity.INFO,
                title="Elevated IV Environment",
                message=f"IV at {context.iv_percentile:.0f}th percentile. "
                "Favorable conditions for premium selling.",
                metrics={
                    "iv_percentile": context.iv_percentile,
                    "vix": context.vix_level,
                    "threshold": th.elevated_iv_threshold,
                },
                suggested_actions=[
                    "Maintain current strategy",
                    "Consider standard position sizing",
                ],
            )

        # Low IV check (requires consecutive days)
        iv_history = await self.metrics_tracker.get_iv_history(
            context.underlying,
            days=th.low_iv_consecutive_days + 1,
        )

        if len(iv_history) >= th.low_iv_consecutive_days:
            low_days = sum(
                1 for h in iv_history[:th.low_iv_consecutive_days]
                if h.get("iv_percentile", 100) < th.low_iv_threshold
            )

            if low_days >= th.low_iv_consecutive_days:
                return Alert(
                    category=AlertCategory.IV_ENVIRONMENT,
                    severity=AlertSeverity.WARNING,
                    title="Low IV Environment Detected",
                    message=f"IV below {th.low_iv_threshold:.0f}th percentile for {low_days}+ consecutive days. "
                    "Consider pausing new entries or accepting lower premium.",
                    metrics={
                        "iv_percentile": context.iv_percentile,
                        "consecutive_low_days": low_days,
                        "threshold": th.low_iv_threshold,
                    },
                    suggested_actions=[
                        "Consider pausing new entries",
                        "Tighten delta to 0.05-0.10 for higher probability",
                        "Wait for IV expansion before resuming",
                    ],
                )

        return None

    async def _check_performance_deviation(
        self,
        metrics: RollingMetrics,
        context: MarketContext,
    ) -> Alert | None:
        """Check for performance deviations from backtest expectations.

        Triggers:
        - Win rate drops below 60% (backtest: 70%)
        - Profit factor drops below 2.0 (backtest: 6.0)
        - Drawdown exceeds 5% (backtest max: 4.35%)
        """
        th = self.thresholds.performance

        # Not enough data
        if metrics.window_trades < 10:
            return None

        # Drawdown check (most important)
        if metrics.current_drawdown >= th.drawdown_critical_threshold:
            return Alert(
                category=AlertCategory.PERFORMANCE_DEVIATION,
                severity=AlertSeverity.CRITICAL,
                title="Critical Drawdown Alert",
                message=f"Drawdown at {metrics.current_drawdown:.1f}% exceeds critical threshold. "
                "Consider reducing position sizes or halting new entries.",
                metrics={
                    "current_drawdown": metrics.current_drawdown,
                    "backtest_max_dd": th.backtest_max_drawdown,
                    "critical_threshold": th.drawdown_critical_threshold,
                },
                suggested_actions=[
                    "Reduce position sizes by 50%",
                    "Skip next N entries until recovery",
                    "Review open positions for potential early exit",
                ],
            )

        if metrics.current_drawdown >= th.drawdown_warning_threshold:
            return Alert(
                category=AlertCategory.PERFORMANCE_DEVIATION,
                severity=AlertSeverity.WARNING,
                title="Drawdown Alert",
                message=f"Drawdown at {metrics.current_drawdown:.1f}% exceeds historical max ({th.backtest_max_drawdown:.1f}%). "
                "Monitor closely.",
                metrics={
                    "current_drawdown": metrics.current_drawdown,
                    "backtest_max_dd": th.backtest_max_drawdown,
                    "warning_threshold": th.drawdown_warning_threshold,
                },
                suggested_actions=[
                    "Monitor open positions closely",
                    "Consider reducing position sizes",
                ],
            )

        # Win rate degradation
        if metrics.win_rate < th.win_rate_warning_threshold:
            deviation = th.expected_win_rate - metrics.win_rate
            return Alert(
                category=AlertCategory.PERFORMANCE_DEVIATION,
                severity=AlertSeverity.WARNING,
                title="Win Rate Degradation",
                message=f"Win rate at {metrics.win_rate:.1f}% over last {metrics.window_trades} trades "
                f"(expected: {th.expected_win_rate:.0f}%).",
                metrics={
                    "current_win_rate": metrics.win_rate,
                    "expected_win_rate": th.expected_win_rate,
                    "deviation": deviation,
                    "window_trades": metrics.window_trades,
                    "last_results": metrics.last_n_results,
                },
                suggested_actions=[
                    "Review entry delta selection (current 0.05-0.15)",
                    "Check fill quality on recent losers",
                    "Consider pausing entries until win rate recovers",
                ],
            )

        # Profit factor degradation
        if metrics.profit_factor < th.profit_factor_warning_threshold:
            return Alert(
                category=AlertCategory.PERFORMANCE_DEVIATION,
                severity=AlertSeverity.WARNING,
                title="Profit Factor Degradation",
                message=f"Profit factor at {metrics.profit_factor:.2f} over last {metrics.window_trades} trades "
                f"(expected: {th.expected_profit_factor:.1f}).",
                metrics={
                    "current_pf": metrics.profit_factor,
                    "expected_pf": th.expected_profit_factor,
                    "window_trades": metrics.window_trades,
                    "total_pnl": metrics.total_pnl,
                },
                suggested_actions=[
                    "Check slippage and market conditions",
                    "Review exit timing (profit target/stop loss)",
                    "Consider market conditions may be unfavorable",
                ],
            )

        return None

    async def _check_slippage_quality(
        self,
        metrics: RollingMetrics,
        context: MarketContext,
    ) -> Alert | None:
        """Check fill quality against expectations.

        Triggers:
        - Average slippage exceeds 75% over last 10 trades
        - Any single fill worse than 85%
        """
        th = self.thresholds.slippage

        if metrics.avg_slippage_pct is None:
            return None

        # Critical slippage (strategy breaks)
        if metrics.worst_slippage_pct and metrics.worst_slippage_pct >= th.critical_slippage:
            return Alert(
                category=AlertCategory.SLIPPAGE_QUALITY,
                severity=AlertSeverity.CRITICAL,
                title="Critical Slippage Detected",
                message=f"Poor fill detected: {metrics.worst_slippage_pct:.1f}% slippage "
                f"(strategy breaks at {th.critical_slippage:.0f}%).",
                metrics={
                    "worst_slippage": metrics.worst_slippage_pct,
                    "avg_slippage": metrics.avg_slippage_pct,
                    "expected_slippage": th.expected_slippage,
                    "critical_threshold": th.critical_slippage,
                },
                suggested_actions=[
                    "Review execution timing",
                    "Consider wider bid-ask spread filters",
                    "Check liquidity conditions at entry time",
                ],
            )

        # Warning slippage
        if metrics.avg_slippage_pct >= th.warning_slippage:
            return Alert(
                category=AlertCategory.SLIPPAGE_QUALITY,
                severity=AlertSeverity.WARNING,
                title="Fill Quality Degrading",
                message=f"Average slippage at {metrics.avg_slippage_pct:.1f}% over last {th.lookback_trades} trades "
                f"(expected: {th.expected_slippage:.0f}%).",
                metrics={
                    "avg_slippage": metrics.avg_slippage_pct,
                    "expected_slippage": th.expected_slippage,
                    "warning_threshold": th.warning_slippage,
                },
                suggested_actions=[
                    "Review execution or widen entry criteria",
                    "Consider limiting entries to higher liquidity periods",
                    "Monitor individual fill quality",
                ],
            )

        return None

    async def _check_regime_change(self, context: MarketContext) -> Alert | None:
        """Check for market regime changes.

        Triggers:
        - Transition to Bear/Crisis regime
        - Transition from Crisis back to normal
        - Circuit breaker triggered
        """
        th = self.thresholds.regime

        # Get previous regime from KV
        prev_regime = await self.kv.get("strategy:previous_regime")

        # Store current regime for next check
        await self.kv.put("strategy:previous_regime", context.market_regime)

        # No change or first run
        if prev_regime is None or prev_regime == context.market_regime:
            return None

        # Regime changed
        is_crisis_entry = context.vix_level >= th.vix_crisis
        is_crisis_exit = (
            prev_regime and "crisis" in prev_regime.lower() and
            context.vix_level < th.vix_caution
        )

        if is_crisis_entry:
            return Alert(
                category=AlertCategory.REGIME_CHANGE,
                severity=AlertSeverity.CRITICAL,
                title="Crisis Regime Entered",
                message=f"Market regime changed to Crisis (VIX: {context.vix_level:.1f}). "
                "Position sizing reduced, new entries halted.",
                metrics={
                    "previous_regime": prev_regime,
                    "current_regime": context.market_regime,
                    "vix": context.vix_level,
                },
                suggested_actions=[
                    "No new entries until regime normalizes",
                    "Monitor existing positions for early exit",
                    "Wait for VIX to drop below 30 before resuming",
                ],
            )

        if is_crisis_exit:
            return Alert(
                category=AlertCategory.REGIME_CHANGE,
                severity=AlertSeverity.INFO,
                title="Market Regime Normalized",
                message=f"Market regime changed from Crisis to {context.market_regime} "
                f"(VIX: {context.vix_level:.1f}). Full position sizing resumed.",
                metrics={
                    "previous_regime": prev_regime,
                    "current_regime": context.market_regime,
                    "vix": context.vix_level,
                },
                suggested_actions=[
                    "Resume normal trading strategy",
                    "Maintain vigilance for regime reversal",
                ],
            )

        # Non-crisis regime change
        if context.vix_level >= th.vix_high:
            severity = AlertSeverity.WARNING
        elif context.vix_level >= th.vix_caution:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        return Alert(
            category=AlertCategory.REGIME_CHANGE,
            severity=severity,
            title="Market Regime Changed",
            message=f"Regime changed: {prev_regime} -> {context.market_regime} "
            f"(VIX: {context.vix_level:.1f}).",
            metrics={
                "previous_regime": prev_regime,
                "current_regime": context.market_regime,
                "vix": context.vix_level,
            },
        )

    async def _check_strategy_switch(
        self,
        metrics: RollingMetrics,
        context: MarketContext,
    ) -> Alert | None:
        """Generate strategy switch recommendations.

        Combines multiple signals to provide actionable recommendations.
        """
        th_iv = self.thresholds.iv
        th_perf = self.thresholds.performance

        # Scenario: Sustained low IV + poor win rate
        low_iv = context.iv_percentile < th_iv.low_iv_threshold
        poor_win_rate = (
            metrics.window_trades >= 10 and
            metrics.win_rate < th_perf.win_rate_warning_threshold
        )

        if low_iv and poor_win_rate:
            return Alert(
                category=AlertCategory.STRATEGY_SWITCH,
                severity=AlertSeverity.WARNING,
                title="Unfavorable Environment",
                message="Current environment unfavorable for premium selling. "
                f"Low IV ({context.iv_percentile:.0f}th pctl) combined with poor win rate ({metrics.win_rate:.1f}%).",
                metrics={
                    "iv_percentile": context.iv_percentile,
                    "win_rate": metrics.win_rate,
                    "vix": context.vix_level,
                },
                suggested_actions=[
                    "Pause trading until conditions improve",
                    "Tighten delta to 0.05-0.10 for higher probability",
                    "Wait for IV expansion above 50th percentile",
                ],
            )

        # Scenario: High IV + high win rate (favorable)
        high_iv = context.iv_percentile >= th_iv.elevated_iv_threshold
        good_win_rate = (
            metrics.window_trades >= 10 and
            metrics.win_rate >= th_perf.expected_win_rate
        )

        if high_iv and good_win_rate:
            return Alert(
                category=AlertCategory.STRATEGY_SWITCH,
                severity=AlertSeverity.INFO,
                title="Favorable Environment Confirmed",
                message=f"Elevated IV ({context.iv_percentile:.0f}th pctl) with strong win rate ({metrics.win_rate:.1f}%). "
                "Consider maintaining current strategy.",
                metrics={
                    "iv_percentile": context.iv_percentile,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                },
                suggested_actions=[
                    "Maintain current strategy",
                    "Consider slightly increasing position frequency if risk limits allow",
                ],
            )

        # Scenario: Approaching drawdown limit
        if metrics.current_drawdown >= th_perf.drawdown_warning_threshold * 0.8:
            return Alert(
                category=AlertCategory.STRATEGY_SWITCH,
                severity=AlertSeverity.WARNING,
                title="Risk Management Alert",
                message=f"Approaching max drawdown (current: {metrics.current_drawdown:.1f}%, "
                f"limit: {th_perf.drawdown_critical_threshold:.1f}%).",
                metrics={
                    "current_drawdown": metrics.current_drawdown,
                    "warning_threshold": th_perf.drawdown_warning_threshold,
                    "critical_threshold": th_perf.drawdown_critical_threshold,
                },
                suggested_actions=[
                    "Reduce position sizes by 50%",
                    "Skip next N entries until recovery",
                    "Review open positions for potential early exit",
                ],
            )

        return None

    async def _filter_by_cooldown(self, alerts: list[Alert]) -> list[Alert]:
        """Filter alerts by cooldown period to prevent spam."""
        filtered: list[Alert] = []
        now = datetime.now()

        for alert in alerts:
            # Get cooldown for this category
            cooldown_hours = self._get_cooldown_hours(alert.category)

            # Check last alert time
            key = f"alert:last:{alert.category.value}"
            last_alert_str = await self.kv.get(key)

            if last_alert_str:
                last_alert = datetime.fromisoformat(last_alert_str)
                if now - last_alert < timedelta(hours=cooldown_hours):
                    continue  # Skip, still in cooldown

            # Not in cooldown, include alert and update timestamp
            filtered.append(alert)
            await self.kv.put(key, now.isoformat())

        return filtered

    def _get_cooldown_hours(self, category: AlertCategory) -> float:
        """Get cooldown period for an alert category."""
        cooldowns = {
            AlertCategory.IV_ENVIRONMENT: self.thresholds.cooldown.iv_alert_cooldown_hours,
            AlertCategory.PERFORMANCE_DEVIATION: self.thresholds.cooldown.performance_alert_cooldown_hours,
            AlertCategory.SLIPPAGE_QUALITY: self.thresholds.cooldown.slippage_alert_cooldown_hours,
            AlertCategory.REGIME_CHANGE: self.thresholds.cooldown.regime_alert_cooldown_hours,
            AlertCategory.STRATEGY_SWITCH: self.thresholds.cooldown.strategy_switch_cooldown_hours,
        }
        return cooldowns.get(category, 4.0)

    async def send_alerts(self, alerts: list[Alert]) -> list[str]:
        """Send alerts to Discord.

        Args:
            alerts: List of Alert objects to send

        Returns:
            List of message IDs for sent alerts
        """
        message_ids: list[str] = []

        for alert in alerts:
            message_id = await self._send_alert_to_discord(alert)
            if message_id:
                message_ids.append(message_id)

        return message_ids

    async def _send_alert_to_discord(self, alert: Alert) -> str | None:
        """Send a single alert to Discord with rich formatting."""
        # Color based on severity
        colors = {
            AlertSeverity.INFO: 0x5865F2,  # Blurple
            AlertSeverity.WARNING: 0xF97316,  # Orange
            AlertSeverity.CRITICAL: 0xED4245,  # Red
        }
        color = colors.get(alert.severity, 0x5865F2)

        # Build fields from metrics
        fields = []
        if alert.metrics:
            for key, value in alert.metrics.items():
                if isinstance(value, float):
                    formatted = f"{value:.2f}" if abs(value) < 100 else f"{value:.1f}"
                elif isinstance(value, list):
                    formatted = " ".join(str(v) for v in value[:10])
                else:
                    formatted = str(value)

                # Clean up key name for display
                display_key = key.replace("_", " ").title()
                fields.append({
                    "name": display_key,
                    "value": formatted,
                    "inline": True,
                })

        # Build embed
        embed = {
            "title": f"Strategy Alert: {alert.title}",
            "description": alert.message,
            "color": color,
            "fields": fields[:9],  # Discord limit is 25, but keep it readable
            "footer": {
                "text": f"Category: {alert.category.value} | Severity: {alert.severity.value}",
            },
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
        }

        # Add suggested actions if present
        if alert.suggested_actions:
            actions_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(alert.suggested_actions))
            embed["fields"].append({
                "name": "Suggested Actions",
                "value": actions_text,
                "inline": False,
            })

        # Add backtest reference
        embed["fields"].append({
            "name": "Reference",
            "value": "Backtest: `analysis/walkforward_findings_2026-01-30.log`",
            "inline": False,
        })

        # Determine content prefix based on severity
        prefix = {
            AlertSeverity.INFO: "**Strategy Monitor**",
            AlertSeverity.WARNING: "**Strategy Alert**",
            AlertSeverity.CRITICAL: "**CRITICAL ALERT**",
        }
        content = f"{prefix.get(alert.severity, 'Alert')}: {alert.title}"

        return await self.discord.send_message(
            content=content,
            embeds=[embed],
        )
