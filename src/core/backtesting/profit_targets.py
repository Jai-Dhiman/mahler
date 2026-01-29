"""Profit target comparison testing.

Option Alpha research shows:
- 50% target: Sharpe 0.77
- 75% target: Sharpe 0.83, +9% RoR

This module enables systematic comparison of different profit targets
with proper slippage and commission modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.backtesting.execution import BacktestTrade


@dataclass
class ProfitTargetResult:
    """Results for a single profit target level."""

    target_pct: float  # e.g., 0.50 for 50%
    total_trades: int
    win_count: int
    loss_count: int

    # Performance metrics
    sharpe_ratio: float
    win_rate: float
    avg_pnl: float
    avg_pnl_pct: float
    total_pnl: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_days_in_trade: float

    # Cost impact
    total_slippage: float
    total_commission: float
    cost_impact_pct: float  # Costs as % of gross P/L


@dataclass
class TargetComparisonResult:
    """Complete comparison across all profit targets."""

    results: dict[float, ProfitTargetResult]
    best_sharpe_target: float
    best_ror_target: float
    recommendation: str

    def get_ranking(self, metric: str = "sharpe_ratio") -> list[tuple[float, float]]:
        """Get targets ranked by specified metric.

        Args:
            metric: Metric to rank by (sharpe_ratio, win_rate, profit_factor)

        Returns:
            List of (target_pct, metric_value) sorted descending
        """
        rankings = []
        for target, result in self.results.items():
            value = getattr(result, metric, 0.0)
            rankings.append((target, value))
        return sorted(rankings, key=lambda x: x[1], reverse=True)


@dataclass
class SimulatedTradeOutcome:
    """Outcome of a single trade under specific parameters."""

    trade_id: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    days_in_trade: int
    slippage_cost: float
    commission_cost: float


class ProfitTargetBacktest:
    """Compare profit targets with proper slippage/commissions.

    Simulates the same set of trades with different profit targets
    to determine optimal exit strategy.
    """

    # Profit targets to test (based on Option Alpha research)
    TARGETS_TO_TEST: list[float] = [0.50, 0.65, 0.75, 0.90]

    def __init__(
        self,
        stop_loss_pct: float = 1.25,
        dte_exit: int = 21,
        risk_free_rate: float = 0.05,
    ):
        """Initialize profit target backtester.

        Args:
            stop_loss_pct: Stop loss as multiple of credit (1.25 = 125%)
            dte_exit: Days to expiration for time-based exit
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.stop_loss_pct = stop_loss_pct
        self.dte_exit = dte_exit
        self.risk_free_rate = risk_free_rate

    def simulate_trade_with_target(
        self,
        trade: BacktestTrade,
        price_path: list[tuple[date, float]],
        target_pct: float,
    ) -> SimulatedTradeOutcome:
        """Simulate a single trade with specific profit target.

        Args:
            trade: Original BacktestTrade with entry details
            price_path: List of (date, spread_debit) for trade duration
            target_pct: Profit target as decimal (0.50 = 50%)

        Returns:
            SimulatedTradeOutcome with result
        """
        entry_credit = trade.entry_credit

        for sim_date, current_debit in price_path:
            # Calculate current P/L percentage
            pnl_pct = (entry_credit - current_debit) / entry_credit

            # Check profit target
            if pnl_pct >= target_pct:
                actual_pnl = target_pct * entry_credit * 100 * trade.contracts
                return SimulatedTradeOutcome(
                    trade_id=trade.trade_id,
                    pnl=actual_pnl - trade.total_costs,
                    pnl_pct=target_pct,
                    exit_reason="profit_target",
                    days_in_trade=(sim_date - trade.entry_date).days,
                    slippage_cost=trade.entry_slippage + trade.exit_slippage,
                    commission_cost=trade.entry_commission + trade.exit_commission,
                )

            # Check stop loss
            if pnl_pct <= -self.stop_loss_pct:
                actual_pnl = -self.stop_loss_pct * entry_credit * 100 * trade.contracts
                return SimulatedTradeOutcome(
                    trade_id=trade.trade_id,
                    pnl=actual_pnl - trade.total_costs,
                    pnl_pct=-self.stop_loss_pct,
                    exit_reason="stop_loss",
                    days_in_trade=(sim_date - trade.entry_date).days,
                    slippage_cost=trade.entry_slippage + trade.exit_slippage,
                    commission_cost=trade.entry_commission + trade.exit_commission,
                )

            # Check time exit
            dte = (trade.expiration - sim_date).days
            if dte <= self.dte_exit:
                actual_pnl = pnl_pct * entry_credit * 100 * trade.contracts
                return SimulatedTradeOutcome(
                    trade_id=trade.trade_id,
                    pnl=actual_pnl - trade.total_costs,
                    pnl_pct=pnl_pct,
                    exit_reason="time_exit",
                    days_in_trade=(sim_date - trade.entry_date).days,
                    slippage_cost=trade.entry_slippage + trade.exit_slippage,
                    commission_cost=trade.entry_commission + trade.exit_commission,
                )

        # If we reach here, trade expired
        final_date, final_debit = price_path[-1]
        pnl_pct = (entry_credit - final_debit) / entry_credit
        actual_pnl = pnl_pct * entry_credit * 100 * trade.contracts

        return SimulatedTradeOutcome(
            trade_id=trade.trade_id,
            pnl=actual_pnl - trade.total_costs,
            pnl_pct=pnl_pct,
            exit_reason="expiration",
            days_in_trade=(final_date - trade.entry_date).days,
            slippage_cost=trade.entry_slippage + trade.exit_slippage,
            commission_cost=trade.entry_commission + trade.exit_commission,
        )

    def calculate_metrics(
        self,
        outcomes: list[SimulatedTradeOutcome],
        target_pct: float,
    ) -> ProfitTargetResult:
        """Calculate performance metrics for a set of outcomes.

        Args:
            outcomes: List of trade outcomes
            target_pct: The profit target used

        Returns:
            ProfitTargetResult with all metrics
        """
        if not outcomes:
            return ProfitTargetResult(
                target_pct=target_pct,
                total_trades=0,
                win_count=0,
                loss_count=0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_pnl=0.0,
                avg_pnl_pct=0.0,
                total_pnl=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                avg_days_in_trade=0.0,
                total_slippage=0.0,
                total_commission=0.0,
                cost_impact_pct=0.0,
            )

        pnls = [o.pnl for o in outcomes]
        pnl_pcts = [o.pnl_pct for o in outcomes]

        # Basic counts
        wins = [o for o in outcomes if o.pnl > 0]
        losses = [o for o in outcomes if o.pnl <= 0]

        # Calculate metrics
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        avg_pnl_pct = np.mean(pnl_pcts)
        win_rate = len(wins) / len(outcomes) if outcomes else 0.0

        # Profit factor
        gross_profit = sum(o.pnl for o in wins) if wins else 0
        gross_loss = abs(sum(o.pnl for o in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe ratio (annualized)
        returns_array = np.array(pnl_pcts)
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            # Assume ~8 trades per year for credit spreads
            trades_per_year = 8
            excess_return = np.mean(returns_array) - self.risk_free_rate / trades_per_year
            sharpe = excess_return / np.std(returns_array) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        # Drawdown calculation
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Max drawdown as percentage (relative to peak)
        max_dd_pct = 0.0
        if len(running_max) > 0 and np.max(running_max) > 0:
            dd_pcts = drawdowns / np.where(running_max > 0, running_max, 1)
            max_dd_pct = np.max(dd_pcts) * 100

        # Average days in trade
        avg_days = np.mean([o.days_in_trade for o in outcomes])

        # Costs
        total_slippage = sum(o.slippage_cost for o in outcomes)
        total_commission = sum(o.commission_cost for o in outcomes)
        total_costs = total_slippage + total_commission
        gross_pnl = total_pnl + total_costs
        cost_impact = (total_costs / gross_pnl * 100) if gross_pnl != 0 else 0.0

        return ProfitTargetResult(
            target_pct=target_pct,
            total_trades=len(outcomes),
            win_count=len(wins),
            loss_count=len(losses),
            sharpe_ratio=float(sharpe),
            win_rate=win_rate,
            avg_pnl=float(avg_pnl),
            avg_pnl_pct=float(avg_pnl_pct),
            total_pnl=float(total_pnl),
            profit_factor=float(profit_factor) if profit_factor != float("inf") else 999.99,
            max_drawdown=float(max_drawdown),
            max_drawdown_pct=float(max_dd_pct),
            avg_days_in_trade=float(avg_days),
            total_slippage=float(total_slippage),
            total_commission=float(total_commission),
            cost_impact_pct=float(cost_impact),
        )

    def run_comparison(
        self,
        trades: list[BacktestTrade],
        price_paths: dict[str, list[tuple[date, float]]],
        targets: list[float] | None = None,
    ) -> TargetComparisonResult:
        """Run backtest for each profit target and compare results.

        Args:
            trades: List of BacktestTrade entries
            price_paths: Dict mapping trade_id to price path [(date, debit), ...]
            targets: Optional custom targets (defaults to TARGETS_TO_TEST)

        Returns:
            TargetComparisonResult with Sharpe comparison
        """
        targets = targets or self.TARGETS_TO_TEST
        results: dict[float, ProfitTargetResult] = {}

        for target in targets:
            outcomes = []
            for trade in trades:
                if trade.trade_id not in price_paths:
                    continue
                outcome = self.simulate_trade_with_target(
                    trade, price_paths[trade.trade_id], target
                )
                outcomes.append(outcome)

            results[target] = self.calculate_metrics(outcomes, target)

        # Find best targets
        best_sharpe_target = max(results.keys(), key=lambda t: results[t].sharpe_ratio)
        best_ror_target = max(results.keys(), key=lambda t: results[t].total_pnl)

        # Generate recommendation
        best_result = results[best_sharpe_target]
        recommendation = (
            f"Recommended profit target: {best_sharpe_target:.0%}. "
            f"Sharpe ratio: {best_result.sharpe_ratio:.2f}, "
            f"Win rate: {best_result.win_rate:.1%}, "
            f"Avg days in trade: {best_result.avg_days_in_trade:.1f}"
        )

        return TargetComparisonResult(
            results=results,
            best_sharpe_target=best_sharpe_target,
            best_ror_target=best_ror_target,
            recommendation=recommendation,
        )

    def run_simple_comparison(
        self,
        trades: list[BacktestTrade],
        targets: list[float] | None = None,
    ) -> TargetComparisonResult:
        """Run simplified comparison using trade outcomes directly.

        Uses the actual trade outcomes rather than simulating price paths.
        Less accurate but faster and requires less data.

        Args:
            trades: List of completed BacktestTrade entries
            targets: Optional custom targets

        Returns:
            TargetComparisonResult
        """
        targets = targets or self.TARGETS_TO_TEST
        results: dict[float, ProfitTargetResult] = {}

        for target in targets:
            outcomes = []
            for trade in trades:
                if trade.net_pnl is None or trade.exit_date is None:
                    continue

                # Determine what would have happened with this target
                pnl_pct = (trade.entry_credit - (trade.exit_debit or 0)) / trade.entry_credit

                if pnl_pct >= target:
                    # Would have hit profit target
                    adjusted_pnl_pct = target
                    exit_reason = "profit_target"
                elif pnl_pct <= -self.stop_loss_pct:
                    adjusted_pnl_pct = -self.stop_loss_pct
                    exit_reason = "stop_loss"
                else:
                    adjusted_pnl_pct = pnl_pct
                    exit_reason = trade.exit_reason or "time_exit"

                adjusted_pnl = (
                    adjusted_pnl_pct * trade.entry_credit * 100 * trade.contracts
                    - trade.total_costs
                )

                outcomes.append(
                    SimulatedTradeOutcome(
                        trade_id=trade.trade_id,
                        pnl=adjusted_pnl,
                        pnl_pct=adjusted_pnl_pct,
                        exit_reason=exit_reason,
                        days_in_trade=trade.days_in_trade or 0,
                        slippage_cost=trade.entry_slippage + trade.exit_slippage,
                        commission_cost=trade.entry_commission + trade.exit_commission,
                    )
                )

            results[target] = self.calculate_metrics(outcomes, target)

        # Find best targets
        best_sharpe_target = max(results.keys(), key=lambda t: results[t].sharpe_ratio)
        best_ror_target = max(results.keys(), key=lambda t: results[t].total_pnl)

        best_result = results[best_sharpe_target]
        recommendation = (
            f"Recommended profit target: {best_sharpe_target:.0%}. "
            f"Sharpe ratio: {best_result.sharpe_ratio:.2f}, "
            f"Win rate: {best_result.win_rate:.1%}"
        )

        return TargetComparisonResult(
            results=results,
            best_sharpe_target=best_sharpe_target,
            best_ror_target=best_ror_target,
            recommendation=recommendation,
        )
