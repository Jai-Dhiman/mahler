"""Backtesting infrastructure with ORATS-compliant slippage and commissions.

This module provides:
- BacktestConfig: Configuration with proper slippage by number of legs
- TradeCosts: Commission and slippage calculations per ORATS methodology
- ProfitTargetBacktest: Comparison testing for different profit targets
- BacktestStatistics: Statistical validation (bootstrap CI, Monte Carlo)
- WalkForwardValidator: Enhanced walk-forward with parameter stability
"""

from core.backtesting.config import BacktestConfig, MarketRegime
from core.backtesting.costs import TradeCosts
from core.backtesting.execution import BacktestExecutor, BacktestTrade, SimulatedFill
from core.backtesting.profit_targets import ProfitTargetBacktest, ProfitTargetResult
from core.backtesting.statistics import BacktestStatistics
from core.backtesting.walk_forward import WalkForwardConfig, WalkForwardValidator

__all__ = [
    "BacktestConfig",
    "MarketRegime",
    "TradeCosts",
    "BacktestExecutor",
    "BacktestTrade",
    "SimulatedFill",
    "ProfitTargetBacktest",
    "ProfitTargetResult",
    "BacktestStatistics",
    "WalkForwardConfig",
    "WalkForwardValidator",
]
