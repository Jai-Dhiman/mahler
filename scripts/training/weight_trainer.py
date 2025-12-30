"""Train weight optimizer and export parameters.

Uses scipy differential_evolution to optimize scoring weights per regime,
then exports parameters for numpy-only inference.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Configuration
MIN_TRADES_PER_REGIME = 25
MIN_TOTAL_TRADES = 100
MAX_ITERATIONS = 50
POPULATION_SIZE = 10


def train_weight_models(trades: list[dict]) -> dict | None:
    """Train weight optimizer for all regimes and export parameters.

    Args:
        trades: List of trade dictionaries from D1 with keys:
            - profit_loss: float
            - iv_rank: float (0-100)
            - short_delta: float
            - credit: float
            - regime: str

    Returns:
        Dictionary with model parameters for JSON serialization,
        or None if insufficient data

    Raises:
        ValueError: If trade data format is invalid
    """
    if len(trades) < MIN_TOTAL_TRADES:
        print(f"Insufficient trades for weight optimization: {len(trades)} < {MIN_TOTAL_TRADES}")
        return None

    # Group trades by regime
    trades_by_regime = _group_by_regime(trades)

    # Optimize weights for each regime
    weights_by_regime = {}
    sharpe_by_regime = {}
    total_trades = 0

    regimes = ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]

    for regime in regimes:
        regime_trades = trades_by_regime.get(regime, [])

        if len(regime_trades) < MIN_TRADES_PER_REGIME:
            print(f"Skipping {regime}: {len(regime_trades)} trades < {MIN_TRADES_PER_REGIME}")
            continue

        result = _optimize_weights_for_regime(regime_trades)
        if result:
            weights, sharpe = result
            weights_by_regime[regime] = {
                "iv": weights[0],
                "delta": weights[1],
                "credit": weights[2],
                "ev": weights[3],
            }
            sharpe_by_regime[regime] = sharpe
            total_trades += len(regime_trades)
            print(f"Optimized {regime}: Sharpe={sharpe:.3f}, n={len(regime_trades)}")

    if not weights_by_regime:
        print("No regimes had sufficient data for optimization")
        return None

    params = {
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "n_samples": total_trades,
        "weights_by_regime": weights_by_regime,
        "sharpe_by_regime": sharpe_by_regime,
    }

    return params


def _group_by_regime(trades: list[dict]) -> dict[str, list[dict]]:
    """Group trades by regime and calculate score components."""
    trades_by_regime: dict[str, list[dict]] = {}

    for trade in trades:
        regime = trade.get("regime") or "bull_low_vol"  # Default

        if regime not in trades_by_regime:
            trades_by_regime[regime] = []

        # Calculate normalized score components
        iv_rank = trade.get("iv_rank") or 50
        short_delta = abs(trade.get("short_delta") or 0.25)
        credit = trade.get("credit") or 0

        trade_dict = {
            "profit_loss": trade.get("profit_loss") or 0,
            "iv_score": iv_rank / 100,  # 0-1
            "delta_score": 1 - abs(short_delta - 0.25) * 4,  # Peak at 0.25
            "credit_score": min(credit * 2, 1.0) if credit else 0.5,
            "ev_score": 0.5,  # Can't reconstruct EV, use neutral
        }

        trades_by_regime[regime].append(trade_dict)

    return trades_by_regime


def _optimize_weights_for_regime(
    trades: list[dict],
) -> tuple[np.ndarray, float] | None:
    """Optimize weights for a single regime.

    Args:
        trades: List of trades for this regime

    Returns:
        Tuple of (optimized_weights, sharpe_ratio) or None if failed
    """
    signals, outcomes = _extract_signals_and_outcomes(trades)

    # Weight bounds: each weight 0.10 to 0.50
    bounds = [(0.10, 0.50)] * 4

    # Constraint: weights sum to 1.0
    constraint = NonlinearConstraint(lambda w: sum(w), 1.0, 1.0)

    result = differential_evolution(
        lambda w: _backtest_sharpe(w, signals, outcomes),
        bounds=bounds,
        maxiter=MAX_ITERATIONS,
        popsize=POPULATION_SIZE,
        constraints=constraint,
        seed=42,
        polish=False,
    )

    # Normalize weights to sum to 1.0
    optimized_weights = result.x / result.x.sum()
    sharpe = float(-result.fun)

    return optimized_weights, sharpe


def _extract_signals_and_outcomes(
    trades: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract signal matrix and outcome vector from trades."""
    signals = []
    outcomes = []

    for trade in trades:
        signal = [
            trade.get("iv_score", 0.5),
            trade.get("delta_score", 0.5),
            trade.get("credit_score", 0.5),
            trade.get("ev_score", 0.5),
        ]
        signals.append(signal)
        outcomes.append(trade.get("profit_loss", 0.0))

    return np.array(signals), np.array(outcomes)


def _backtest_sharpe(
    weights: np.ndarray,
    signals: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Calculate negative Sharpe ratio for minimization."""
    # Calculate scores for all trades
    scores = np.dot(signals, weights)

    # Select top 70% by score
    threshold = np.percentile(scores, 30)
    selected_mask = scores >= threshold

    if selected_mask.sum() < 2:
        return 1000.0  # Penalty

    selected_outcomes = outcomes[selected_mask]

    # Calculate Sharpe ratio
    mean_return = np.mean(selected_outcomes)
    std_return = np.std(selected_outcomes)

    if std_return < 1e-8:
        return 0.0 if mean_return > 0 else 1000.0

    sharpe = mean_return / std_return

    return -sharpe


if __name__ == "__main__":
    print("Weight trainer module loaded successfully")
    print("Run train_models.py to train weight models")
