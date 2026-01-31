"""Strategy monitoring and alerting system.

This module provides intelligent monitoring of strategy performance against
backtest expectations, with Discord notifications when action may be needed.

Backtest reference: analysis/walkforward_findings_2026-01-30.log
"""

from core.monitoring.alert_thresholds import AlertThresholds
from core.monitoring.metrics_tracker import MetricsTracker, RollingMetrics
from core.monitoring.strategy_monitor import StrategyMonitor, AlertCategory, Alert

__all__ = [
    "AlertThresholds",
    "MetricsTracker",
    "RollingMetrics",
    "StrategyMonitor",
    "AlertCategory",
    "Alert",
]
