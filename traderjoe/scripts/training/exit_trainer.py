"""Train exit optimizer and export parameters.

Uses scipy differential_evolution to optimize exit parameters
(profit target, stop loss, time exit DTE), then exports parameters
for numpy-only inference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Configuration
MIN_TRADES = 50
MAX_ITERATIONS = 100
POPULATION_SIZE = 15

# Parameter bounds
PROFIT_TARGET_BOUNDS = (0.05, 0.50)  # 5% to 50%
STOP_LOSS_BOUNDS = (0.02, 0.30)  # 2% to 30%
TIME_EXIT_BOUNDS = (7, 45)  # 7 to 45 DTE


@dataclass
class TradeOutcome:
    """Trade outcome for optimization."""

    entry_credit: float
    exit_debit: float
    entry_dte: int
    exit_dte: int
    profit_pct: float


def train_exit_model(trades: list[dict]) -> dict | None:
    """Train exit optimizer and export parameters.

    Args:
        trades: List of trade dictionaries from D1 with keys:
            - entry_credit: float
            - exit_debit: float
            - opened_at: str (ISO datetime)
            - closed_at: str (ISO datetime)
            - expiration: str (date)
            - dte_at_exit: int (optional)

    Returns:
        Dictionary with model parameters for JSON serialization,
        or None if insufficient data
    """
    # Convert to TradeOutcome objects
    outcomes = _convert_to_outcomes(trades)

    if len(outcomes) < MIN_TRADES:
        print(f"Insufficient trades for exit optimization: {len(outcomes)} < {MIN_TRADES}")
        return None

    print(f"Training exit optimizer with {len(outcomes)} trades")

    # Define bounds
    bounds = [
        PROFIT_TARGET_BOUNDS,
        STOP_LOSS_BOUNDS,
        TIME_EXIT_BOUNDS,
    ]

    # Run optimization
    result = differential_evolution(
        lambda params: _objective(params, outcomes),
        bounds=bounds,
        maxiter=MAX_ITERATIONS,
        popsize=POPULATION_SIZE,
        polish=True,
        seed=42,
    )

    profit_target, stop_loss, time_exit_dte = result.x
    sharpe = -result.fun

    params = {
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(outcomes),
        "profit_target": round(float(profit_target), 3),
        "stop_loss": round(float(stop_loss), 3),
        "time_exit_dte": int(round(time_exit_dte)),
        "sharpe_ratio": round(float(sharpe), 3),
    }

    print(f"Optimized exit params: profit_target={params['profit_target']:.1%}, "
          f"stop_loss={params['stop_loss']:.1%}, time_exit_dte={params['time_exit_dte']}, "
          f"Sharpe={params['sharpe_ratio']:.3f}")

    return params


def _convert_to_outcomes(trades: list[dict]) -> list[TradeOutcome]:
    """Convert trade dictionaries to TradeOutcome objects."""
    outcomes = []

    for trade in trades:
        entry_credit = trade.get("entry_credit")
        exit_debit = trade.get("exit_debit")

        if not entry_credit or entry_credit <= 0:
            continue
        if exit_debit is None:
            continue

        # Calculate profit percentage
        profit_pct = (entry_credit - exit_debit) / entry_credit

        # Estimate entry DTE (default 45)
        entry_dte = 45  # Reasonable default for options

        # Get exit DTE
        exit_dte = trade.get("dte_at_exit")
        if exit_dte is None:
            exit_dte = 0  # Assume expired if not specified

        outcomes.append(TradeOutcome(
            entry_credit=entry_credit,
            exit_debit=exit_debit,
            entry_dte=entry_dte,
            exit_dte=exit_dte,
            profit_pct=profit_pct,
        ))

    return outcomes


def _objective(params: np.ndarray, outcomes: list[TradeOutcome]) -> float:
    """Objective function for optimization.

    Args:
        params: Array of [profit_target, stop_loss, time_exit_dte]
        outcomes: List of trade outcomes

    Returns:
        Negative Sharpe ratio (we minimize)
    """
    profit_target, stop_loss, time_exit_dte = params
    time_exit_dte = int(round(time_exit_dte))

    # Simulate exits
    returns = _simulate_exits(outcomes, profit_target, stop_loss, time_exit_dte)

    # Calculate Sharpe
    sharpe = _calculate_sharpe(returns)

    return -sharpe


def _simulate_exits(
    outcomes: list[TradeOutcome],
    profit_target: float,
    stop_loss: float,
    time_exit_dte: int,
) -> list[float]:
    """Simulate exits using given parameters."""
    returns = []

    for trade in outcomes:
        # If trade hit profit target
        if trade.profit_pct >= profit_target:
            returns.append(profit_target)

        # If trade hit stop loss
        elif trade.profit_pct <= -stop_loss:
            returns.append(-stop_loss)

        # If trade exited due to time
        elif trade.exit_dte <= time_exit_dte:
            returns.append(trade.profit_pct)

        else:
            returns.append(trade.profit_pct)

    return returns


def _calculate_sharpe(returns: list[float]) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) < 2:
        return -1000.0

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)

    if std_return < 1e-8:
        return -1000.0

    # Annualize assuming ~252 trading days per year
    # and average trade duration of ~30 days
    trades_per_year = 252 / 30
    sharpe = mean_return / std_return * np.sqrt(trades_per_year)

    return sharpe


if __name__ == "__main__":
    print("Exit trainer module loaded successfully")
    print("Run train_models.py to train exit model")
