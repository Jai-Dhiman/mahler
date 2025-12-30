"""Weight optimization for regime-conditional scoring.

Uses scipy.optimize to find optimal scoring weights per regime
based on historical trade outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from core.analysis.screener import ScoringWeights

if TYPE_CHECKING:
    from core.db.d1 import D1Client
    from core.db.kv import KVClient
    from core.inference.weight_inference import PrecomputedWeightProvider


@dataclass
class OptimizedWeights:
    """Result of weight optimization."""

    regime: str
    weights: ScoringWeights
    sharpe_ratio: float
    n_trades: int
    optimized_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime": self.regime,
            "weights": self.weights.to_dict(),
            "sharpe_ratio": self.sharpe_ratio,
            "n_trades": self.n_trades,
            "optimized_at": self.optimized_at,
        }


class WeightOptimizer:
    """Optimizes scoring weights by regime using historical trades.

    Uses scipy.optimize.differential_evolution to find weights
    that maximize Sharpe ratio for each regime.
    """

    # Configuration
    MIN_TRADES_PER_REGIME = 25  # Minimum trades to optimize
    MIN_TOTAL_TRADES = 100  # Minimum total trades required
    MAX_ITERATIONS = 50  # Limit for CPU time
    POPULATION_SIZE = 10

    def __init__(self, trades_by_regime: dict[str, list[dict]]):
        """Initialize with trades grouped by regime.

        Args:
            trades_by_regime: Dict of regime -> list of trade dicts
                Each trade dict should have:
                - profit_loss: float (P/L amount)
                - iv_score: float (normalized 0-1)
                - delta_score: float (normalized 0-1)
                - credit_score: float (normalized 0-1)
                - ev_score: float (normalized 0-1)
        """
        self.trades_by_regime = trades_by_regime

    def _extract_signals_and_outcomes(
        self,
        trades: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract signal matrix and outcome vector from trades.

        Args:
            trades: List of trade dicts

        Returns:
            Tuple of (signals matrix [n_trades x 4], outcomes vector [n_trades])
        """
        signals = []
        outcomes = []

        for trade in trades:
            # Extract score components (default to 0.5 if missing)
            signal = [
                trade.get("iv_score", 0.5),
                trade.get("delta_score", 0.5),
                trade.get("credit_score", 0.5),
                trade.get("ev_score", 0.5),
            ]
            signals.append(signal)
            outcomes.append(trade.get("profit_loss", 0.0))

        return np.array(signals), np.array(outcomes)

    def backtest_sharpe(
        self,
        weights: np.ndarray,
        signals: np.ndarray,
        outcomes: np.ndarray,
    ) -> float:
        """Calculate Sharpe ratio for given weights.

        Simulates scoring with weights and measures performance
        of selected trades.

        Args:
            weights: Array of [iv, delta, credit, ev] weights
            signals: Matrix of signal values [n_trades x 4]
            outcomes: Array of trade outcomes (P/L)

        Returns:
            Negative Sharpe ratio (for minimization)
        """
        # Calculate scores for all trades
        scores = np.dot(signals, weights)

        # Select top 70% by score (simulates filtering)
        threshold = np.percentile(scores, 30)
        selected_mask = scores >= threshold

        if selected_mask.sum() < 2:
            return 1000.0  # Penalty for too few selected

        selected_outcomes = outcomes[selected_mask]

        # Calculate Sharpe ratio
        mean_return = np.mean(selected_outcomes)
        std_return = np.std(selected_outcomes)

        if std_return < 1e-8:
            return 0.0 if mean_return > 0 else 1000.0

        sharpe = mean_return / std_return

        # Return negative for minimization
        return -sharpe

    def optimize_weights(self, regime: str) -> OptimizedWeights | None:
        """Optimize weights for a specific regime.

        Args:
            regime: Regime name (e.g., "bull_low_vol")

        Returns:
            OptimizedWeights if successful, None if insufficient data
        """
        trades = self.trades_by_regime.get(regime, [])

        if len(trades) < self.MIN_TRADES_PER_REGIME:
            print(f"Insufficient trades for {regime}: {len(trades)} (need {self.MIN_TRADES_PER_REGIME})")
            return None

        signals, outcomes = self._extract_signals_and_outcomes(trades)

        # Weight bounds: each weight 0.10 to 0.50
        bounds = [(0.10, 0.50)] * 4

        # Constraint: weights sum to 1.0
        def constraint_func(w):
            return sum(w) - 1.0

        # Lazy import scipy (only used in dev mode, not in production workers)
        from scipy.optimize import NonlinearConstraint, differential_evolution

        constraint = NonlinearConstraint(lambda w: sum(w), 1.0, 1.0)

        result = differential_evolution(
            lambda w: self.backtest_sharpe(w, signals, outcomes),
            bounds=bounds,
            maxiter=self.MAX_ITERATIONS,
            popsize=self.POPULATION_SIZE,
            constraints=constraint,
            seed=42,
            polish=False,  # Skip local optimization for speed
        )

        # Normalize weights to sum to 1.0 (in case of small numerical errors)
        optimized_weights = result.x / result.x.sum()

        return OptimizedWeights(
            regime=regime,
            weights=ScoringWeights(
                iv_weight=float(optimized_weights[0]),
                delta_weight=float(optimized_weights[1]),
                credit_weight=float(optimized_weights[2]),
                ev_weight=float(optimized_weights[3]),
            ),
            sharpe_ratio=float(-result.fun),  # Negate back to positive
            n_trades=len(trades),
            optimized_at=datetime.now().isoformat(),
        )

    def optimize_all_regimes(self) -> dict[str, OptimizedWeights]:
        """Optimize weights for all regimes with sufficient data.

        Returns:
            Dict of regime -> OptimizedWeights
        """
        results = {}
        regimes = ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]

        for regime in regimes:
            result = self.optimize_weights(regime)
            if result:
                results[regime] = result
                print(f"Optimized {regime}: Sharpe={result.sharpe_ratio:.3f}, n={result.n_trades}")

        return results

    def get_total_trades(self) -> int:
        """Get total number of trades across all regimes."""
        return sum(len(trades) for trades in self.trades_by_regime.values())

    @staticmethod
    async def from_db(db: D1Client) -> WeightOptimizer:
        """Create optimizer from database trades.

        Fetches closed trades with regime context and groups by regime.
        Calculates score components from stored trade data.

        Args:
            db: D1 database client

        Returns:
            WeightOptimizer initialized with trade data
        """
        # Query closed trades with regime at entry time
        result = await db.execute(
            """
            SELECT
                t.id,
                t.profit_loss,
                t.entry_credit,
                r.underlying,
                r.iv_rank,
                r.delta as short_delta,
                r.credit,
                mr.regime
            FROM trades t
            JOIN recommendations r ON t.recommendation_id = r.id
            LEFT JOIN market_regimes mr ON
                mr.symbol = r.underlying AND
                DATE(mr.detected_at) = DATE(t.opened_at)
            WHERE t.status = 'closed'
            ORDER BY t.closed_at DESC
            LIMIT 500
            """
        )

        # Group by regime and calculate score components
        trades_by_regime: dict[str, list[dict]] = {}

        for row in result.get("results", []):
            regime = row.get("regime") or "bull_low_vol"  # Default if no regime stored

            if regime not in trades_by_regime:
                trades_by_regime[regime] = []

            # Calculate normalized score components
            # These approximate what the screener would have calculated
            iv_rank = row.get("iv_rank") or 50
            short_delta = abs(row.get("short_delta") or 0.25)
            credit = row.get("credit") or 0

            trade_dict = {
                "profit_loss": row.get("profit_loss") or 0,
                "iv_score": iv_rank / 100,  # 0-1
                "delta_score": 1 - abs(short_delta - 0.25) * 4,  # Peak at 0.25
                "credit_score": min(credit * 2, 1.0) if credit else 0.5,  # Rough estimate
                "ev_score": 0.5,  # Can't reconstruct EV, use neutral
            }

            trades_by_regime[regime].append(trade_dict)

        return WeightOptimizer(trades_by_regime)


async def create_weight_provider(
    env: Any,
    kv: KVClient | None = None,
) -> PrecomputedWeightProvider:
    """Factory to create weight provider from pre-computed parameters.

    In production (MODELS_BUCKET binding exists): Loads optimized weights from R2
    In development: Returns provider with None params (uses defaults)

    Args:
        env: Cloudflare environment with bindings
        kv: Optional KV client for caching

    Returns:
        PrecomputedWeightProvider instance
    """
    from core.inference.weight_inference import PrecomputedWeightProvider

    # Check if we have the models bucket binding (production mode)
    models_bucket = getattr(env, "MODELS_BUCKET", None)

    if models_bucket is None:
        # Development mode: return provider with defaults
        return PrecomputedWeightProvider(None)

    # Production mode: load from R2
    from core.inference.model_loader import ModelLoader

    loader = ModelLoader(models_bucket, kv)
    params = await loader.get_weight_params()

    # params may be None if no model was trained yet
    return PrecomputedWeightProvider(params)
