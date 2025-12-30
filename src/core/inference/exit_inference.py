"""Numpy-only exit parameter retrieval using pre-computed parameters.

This module provides optimized exit parameters without scipy dependencies,
using pre-trained parameters loaded from R2.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.inference.model_loader import ExitModelParams


@dataclass
class ExitParams:
    """Exit parameters for position management."""

    profit_target: float  # 0.0 - 1.0 (e.g., 0.50 = 50% of credit received)
    stop_loss: float  # 0.0 - 1.0 (e.g., 0.20 = 200% of credit, i.e., 20% of max loss)
    time_exit_dte: int  # Days to expiration for time-based exit


class PrecomputedExitProvider:
    """Provides pre-computed optimized exit parameters.

    Loads parameters from pre-trained model and provides
    simple access to exit thresholds.
    """

    # Default exit params (from PRD)
    DEFAULT_PARAMS = ExitParams(
        profit_target=0.50,  # Exit at 50% of max profit
        stop_loss=0.20,  # Stop at 200% of credit received
        time_exit_dte=21,  # Exit at 21 DTE
    )

    def __init__(self, params: ExitModelParams | None):
        """Initialize with pre-computed parameters.

        Args:
            params: Pre-trained model parameters from R2, or None for defaults
        """
        self.params = params

    def get_exit_params(self) -> ExitParams:
        """Get optimized exit parameters.

        Returns:
            ExitParams with profit target, stop loss, and time exit DTE
        """
        if self.params is None:
            return self.DEFAULT_PARAMS

        return ExitParams(
            profit_target=self.params.profit_target,
            stop_loss=self.params.stop_loss,
            time_exit_dte=self.params.time_exit_dte,
        )

    def get_sharpe_ratio(self) -> float | None:
        """Get the Sharpe ratio achieved during optimization.

        Returns:
            Sharpe ratio or None if not available
        """
        if self.params is None:
            return None

        return self.params.sharpe_ratio

    def has_optimized_params(self) -> bool:
        """Check if optimized parameters exist.

        Returns:
            True if optimized parameters were loaded, False otherwise
        """
        return self.params is not None
