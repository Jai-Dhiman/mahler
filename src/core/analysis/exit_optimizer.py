"""Exit parameter optimizer using scipy optimization.

Uses differential_evolution to find optimal exit parameters
(profit target, stop loss, time exit) that maximize Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.db.kv import KVClient
    from core.inference.exit_inference import PrecomputedExitProvider


@dataclass
class OptimizedExitParams:
    """Optimized exit parameters."""

    profit_target: float  # 0.05 - 0.50
    stop_loss: float  # 0.02 - 0.30
    time_exit_dte: int  # 7 - 45
    sharpe_ratio: float
    n_trades: int
    optimized_at: str


@dataclass
class TradeOutcome:
    """Trade outcome for optimization."""

    entry_credit: float
    exit_debit: float
    entry_dte: int
    exit_dte: int
    profit_pct: float  # (entry - exit) / entry


class ExitParameterOptimizer:
    """Optimizes exit parameters using historical trade data.

    Uses scipy.optimize.differential_evolution to find optimal
    profit_target, stop_loss, and time_exit parameters that
    maximize the Sharpe ratio of closed trades.
    """

    # Parameter bounds
    PROFIT_TARGET_BOUNDS = (0.05, 0.50)  # 5% to 50%
    STOP_LOSS_BOUNDS = (0.02, 0.30)  # 2% to 30%
    TIME_EXIT_BOUNDS = (7, 45)  # 7 to 45 DTE

    # Optimization settings
    MAX_ITERATIONS = 100
    POPULATION_SIZE = 15
    MIN_TRADES = 50  # Minimum trades required for optimization

    def __init__(self, trades: list[TradeOutcome]):
        """Initialize optimizer with trade history.

        Args:
            trades: List of historical trade outcomes
        """
        self.trades = trades
        self.n_trades = len(trades)

    def _simulate_exits(
        self,
        profit_target: float,
        stop_loss: float,
        time_exit_dte: int,
    ) -> list[float]:
        """Simulate exits using given parameters.

        For each historical trade, determine what the outcome would
        have been using the specified exit parameters.

        Args:
            profit_target: Profit target as decimal (e.g., 0.50 for 50%)
            stop_loss: Stop loss as decimal (e.g., 0.20 for 20%)
            time_exit_dte: Days to expiration for time exit

        Returns:
            List of simulated returns (P/L percentages)
        """
        returns = []

        for trade in self.trades:
            # Simplified simulation based on actual outcome
            # In reality, this would use intraday data to simulate
            # different exit points. For now, we use the actual outcome
            # adjusted for the parameters.

            # If trade hit profit target (actual profit >= target)
            if trade.profit_pct >= profit_target:
                # Would have exited at profit target
                returns.append(profit_target)

            # If trade hit stop loss (actual loss >= stop loss)
            elif trade.profit_pct <= -stop_loss:
                # Would have exited at stop loss
                returns.append(-stop_loss)

            # If trade exited due to time (exit DTE <= time_exit_dte)
            elif trade.exit_dte <= time_exit_dte:
                # Would have exited at whatever the P/L was at that DTE
                returns.append(trade.profit_pct)

            else:
                # Trade didn't hit any exit condition (edge case)
                returns.append(trade.profit_pct)

        return returns

    def _calculate_sharpe(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio from returns.

        Args:
            returns: List of return percentages

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return -1000.0  # Penalty for insufficient data

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return < 1e-8:
            return -1000.0  # Penalty for zero volatility

        # Annualize assuming ~252 trading days per year
        # and average trade duration of ~30 days
        trades_per_year = 252 / 30
        sharpe = mean_return / std_return * np.sqrt(trades_per_year)

        return sharpe

    def _objective(self, params: np.ndarray) -> float:
        """Objective function for optimization.

        Args:
            params: Array of [profit_target, stop_loss, time_exit_dte]

        Returns:
            Negative Sharpe ratio (we minimize, so negative = maximize)
        """
        profit_target, stop_loss, time_exit_dte = params

        # Simulate exits with these parameters
        returns = self._simulate_exits(
            profit_target=profit_target,
            stop_loss=stop_loss,
            time_exit_dte=int(round(time_exit_dte)),
        )

        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe(returns)

        # Return negative because we minimize
        return -sharpe

    def optimize(self) -> OptimizedExitParams | None:
        """Run optimization to find best exit parameters.

        Returns:
            OptimizedExitParams if successful, None if insufficient data
        """
        if self.n_trades < self.MIN_TRADES:
            print(
                f"Insufficient trades for optimization: {self.n_trades} < {self.MIN_TRADES}"
            )
            return None

        # Define bounds for differential evolution
        bounds = [
            self.PROFIT_TARGET_BOUNDS,
            self.STOP_LOSS_BOUNDS,
            self.TIME_EXIT_BOUNDS,
        ]

        # Lazy import scipy (only used in dev mode, not in production workers)
        from scipy.optimize import differential_evolution

        # Run optimization
        result = differential_evolution(
            self._objective,
            bounds=bounds,
            maxiter=self.MAX_ITERATIONS,
            popsize=self.POPULATION_SIZE,
            polish=True,  # Polish with L-BFGS-B for better convergence
            seed=42,  # Reproducibility
        )

        if not result.success:
            print(f"Optimization did not converge: {result.message}")
            # Still return result, it may be useful

        profit_target, stop_loss, time_exit_dte = result.x
        sharpe = -result.fun  # Convert back to positive

        return OptimizedExitParams(
            profit_target=round(profit_target, 3),
            stop_loss=round(stop_loss, 3),
            time_exit_dte=int(round(time_exit_dte)),
            sharpe_ratio=round(sharpe, 3),
            n_trades=self.n_trades,
            optimized_at=datetime.now().isoformat(),
        )

    @staticmethod
    def from_db_trades(closed_trades: list) -> ExitParameterOptimizer:
        """Create optimizer from database Trade objects.

        Args:
            closed_trades: List of closed Trade objects from D1

        Returns:
            ExitParameterOptimizer instance
        """
        from core.analysis.greeks import days_to_expiry

        outcomes = []
        for trade in closed_trades:
            if trade.entry_credit <= 0 or trade.exit_debit is None:
                continue

            # Calculate entry DTE (estimate from opened_at and expiration)
            if trade.opened_at:
                entry_dte = days_to_expiry(trade.expiration)
                # Adjust for how long trade was open
                if trade.closed_at:
                    days_open = (trade.closed_at - trade.opened_at).days
                    entry_dte = entry_dte + days_open
            else:
                entry_dte = 45  # Default estimate

            # Get exit DTE from stored value or calculate
            exit_dte = getattr(trade, "dte_at_exit", None)
            if exit_dte is None:
                exit_dte = days_to_expiry(trade.expiration)

            profit_pct = (trade.entry_credit - trade.exit_debit) / trade.entry_credit

            outcomes.append(
                TradeOutcome(
                    entry_credit=trade.entry_credit,
                    exit_debit=trade.exit_debit,
                    entry_dte=entry_dte,
                    exit_dte=exit_dte,
                    profit_pct=profit_pct,
                )
            )

        return ExitParameterOptimizer(outcomes)


async def create_exit_provider(
    env: Any,
    kv: KVClient | None = None,
) -> PrecomputedExitProvider:
    """Factory to create exit provider from pre-computed parameters.

    In production (MODELS_BUCKET binding exists): Loads optimized exit params from R2
    In development: Returns provider with None params (uses defaults)

    Args:
        env: Cloudflare environment with bindings
        kv: Optional KV client for caching

    Returns:
        PrecomputedExitProvider instance
    """
    from core.inference.exit_inference import PrecomputedExitProvider

    # Check if we have the models bucket binding (production mode)
    models_bucket = getattr(env, "MODELS_BUCKET", None)

    if models_bucket is None:
        # Development mode: return provider with defaults
        return PrecomputedExitProvider(None)

    # Production mode: load from R2
    from core.inference.model_loader import ModelLoader

    loader = ModelLoader(models_bucket, kv)
    params = await loader.get_exit_params()

    # params may be None if no model was trained yet
    return PrecomputedExitProvider(params)
