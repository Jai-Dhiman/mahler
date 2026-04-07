"""Numpy-only weight retrieval using pre-computed parameters.

This module provides optimized scoring weights by regime without
scipy dependencies, using pre-trained parameters loaded from R2.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.inference.model_loader import WeightModelParams


@dataclass
class ScoringWeights:
    """Weights for scoring spread opportunities.

    All weights must sum to 1.0.
    """

    iv_weight: float = 0.25
    delta_weight: float = 0.25
    credit_weight: float = 0.25
    ev_weight: float = 0.25


class PrecomputedWeightProvider:
    """Provides pre-computed optimized weights by regime.

    Loads weights from pre-trained parameters and provides
    simple lookup by regime name.
    """

    # Default weights if no model available
    DEFAULT_WEIGHTS = ScoringWeights(
        iv_weight=0.25,
        delta_weight=0.25,
        credit_weight=0.25,
        ev_weight=0.25,
    )

    def __init__(self, params: WeightModelParams | None):
        """Initialize with pre-computed parameters.

        Args:
            params: Pre-trained model parameters from R2, or None for defaults
        """
        self.params = params

    def get_weights(self, regime: str) -> ScoringWeights:
        """Get optimized weights for a regime.

        Args:
            regime: Regime name (e.g., "bull_low_vol")

        Returns:
            ScoringWeights for the regime, or defaults if not available
        """
        if self.params is None:
            return self.DEFAULT_WEIGHTS

        weights_dict = self.params.weights_by_regime.get(regime)
        if weights_dict is None:
            return self.DEFAULT_WEIGHTS

        return ScoringWeights(
            iv_weight=weights_dict["iv"],
            delta_weight=weights_dict["delta"],
            credit_weight=weights_dict["credit"],
            ev_weight=weights_dict["ev"],
        )

    def get_all_weights(self) -> dict[str, ScoringWeights]:
        """Get optimized weights for all regimes.

        Returns:
            Dictionary mapping regime name to ScoringWeights,
            or empty dict if no parameters available
        """
        if self.params is None:
            return {}

        return {
            regime: ScoringWeights(
                iv_weight=w["iv"],
                delta_weight=w["delta"],
                credit_weight=w["credit"],
                ev_weight=w["ev"],
            )
            for regime, w in self.params.weights_by_regime.items()
        }

    def get_sharpe_ratio(self, regime: str) -> float | None:
        """Get the Sharpe ratio achieved during optimization.

        Args:
            regime: Regime name

        Returns:
            Sharpe ratio or None if not available
        """
        if self.params is None:
            return None

        return self.params.sharpe_by_regime.get(regime)

    def has_weights_for_regime(self, regime: str) -> bool:
        """Check if optimized weights exist for a regime.

        Args:
            regime: Regime name

        Returns:
            True if optimized weights exist, False otherwise
        """
        if self.params is None:
            return False

        return regime in self.params.weights_by_regime
