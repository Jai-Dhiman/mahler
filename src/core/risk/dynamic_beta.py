"""Dynamic beta calculation using EWMA and rolling windows.

Replaces static betas with dynamically calculated values that
respond to changing market correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.types import ASSET_BETAS


@dataclass
class DynamicBetaResult:
    """Result of dynamic beta calculation."""

    symbol: str
    beta_ewma: float
    beta_rolling_20: float | None
    beta_rolling_60: float | None
    beta_blended: float
    correlation_spy: float | None
    data_days: int
    calculated_at: str
    is_fallback: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "beta_ewma": self.beta_ewma,
            "beta_rolling_20": self.beta_rolling_20,
            "beta_rolling_60": self.beta_rolling_60,
            "beta_blended": self.beta_blended,
            "correlation_spy": self.correlation_spy,
            "data_days": self.data_days,
            "calculated_at": self.calculated_at,
            "is_fallback": self.is_fallback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DynamicBetaResult:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            beta_ewma=data["beta_ewma"],
            beta_rolling_20=data.get("beta_rolling_20"),
            beta_rolling_60=data.get("beta_rolling_60"),
            beta_blended=data["beta_blended"],
            correlation_spy=data.get("correlation_spy"),
            data_days=data["data_days"],
            calculated_at=data["calculated_at"],
            is_fallback=data.get("is_fallback", False),
        )


class DynamicBetaCalculator:
    """Calculates dynamic betas relative to SPY.

    Uses EWMA and multi-window rolling betas to capture
    changing market relationships.
    """

    # Configuration
    EWMA_HALFLIFE = 20  # EWMA halflife in days
    MIN_DATA_DAYS = 60  # Minimum days for dynamic calculation
    BLEND_WEIGHTS = {
        "ewma": 0.5,
        "rolling_20": 0.3,
        "rolling_60": 0.2,
    }

    def __init__(self, fallback_betas: dict[str, float] | None = None):
        """Initialize with optional fallback betas.

        Args:
            fallback_betas: Static betas to use if dynamic calc fails
        """
        self.fallback_betas = fallback_betas or ASSET_BETAS.copy()

    def _calculate_returns(self, bars: list[dict]) -> np.ndarray:
        """Calculate log returns from OHLCV bars.

        Args:
            bars: List of bar dicts with 'close' key

        Returns:
            Array of log returns
        """
        closes = np.array([bar["close"] for bar in bars])
        returns = np.diff(np.log(closes))
        return returns

    def ewma_beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        halflife: int = 20,
    ) -> float:
        """Calculate EWMA beta.

        Uses exponentially weighted covariance and variance.

        Args:
            asset_returns: Array of asset log returns
            market_returns: Array of market (SPY) log returns
            halflife: EWMA halflife in periods

        Returns:
            EWMA beta
        """
        if len(asset_returns) != len(market_returns):
            raise ValueError("Return arrays must have same length")

        if len(asset_returns) < 2:
            raise ValueError("Need at least 2 returns for beta calculation")

        # Calculate decay factor (alpha)
        alpha = 1 - np.exp(-np.log(2) / halflife)

        n = len(asset_returns)
        weights = np.array([(1 - alpha) ** i for i in range(n - 1, -1, -1)])
        weights = weights / weights.sum()

        # Weighted means
        asset_mean = np.sum(weights * asset_returns)
        market_mean = np.sum(weights * market_returns)

        # Weighted covariance and variance
        asset_dev = asset_returns - asset_mean
        market_dev = market_returns - market_mean

        covariance = np.sum(weights * asset_dev * market_dev)
        variance = np.sum(weights * market_dev * market_dev)

        if variance == 0:
            return 1.0

        return covariance / variance

    def rolling_beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        window: int,
    ) -> float | None:
        """Calculate rolling beta for a specific window.

        Args:
            asset_returns: Array of asset log returns
            market_returns: Array of market log returns
            window: Rolling window size

        Returns:
            Rolling beta or None if insufficient data
        """
        if len(asset_returns) < window:
            return None

        # Use most recent 'window' returns
        asset_recent = asset_returns[-window:]
        market_recent = market_returns[-window:]

        covariance = np.cov(asset_recent, market_recent)[0, 1]
        variance = np.var(market_recent, ddof=1)

        if variance == 0:
            return 1.0

        return covariance / variance

    def rolling_beta_multiwindow(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        windows: list[int] | None = None,
    ) -> dict[int, float | None]:
        """Calculate rolling betas for multiple windows.

        Args:
            asset_returns: Array of asset log returns
            market_returns: Array of market log returns
            windows: List of window sizes (default [20, 60])

        Returns:
            Dict of window -> beta value (or None if insufficient data)
        """
        if windows is None:
            windows = [20, 60]

        return {w: self.rolling_beta(asset_returns, market_returns, w) for w in windows}

    def calculate_correlation(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> float | None:
        """Calculate correlation with market.

        Args:
            asset_returns: Array of asset log returns
            market_returns: Array of market log returns

        Returns:
            Correlation coefficient or None if insufficient data
        """
        if len(asset_returns) < 20:
            return None

        corr_matrix = np.corrcoef(asset_returns, market_returns)
        return float(corr_matrix[0, 1])

    def blended_beta(
        self,
        beta_ewma: float,
        rolling_betas: dict[int, float | None],
        static_beta: float,
    ) -> float:
        """Blend betas using configured weights.

        Falls back to static beta for missing components.

        Args:
            beta_ewma: EWMA beta
            rolling_betas: Dict of window -> beta
            static_beta: Static fallback beta

        Returns:
            Blended beta value
        """
        weights = self.BLEND_WEIGHTS.copy()

        # Get component values, using static fallback for missing
        beta_20 = rolling_betas.get(20) if rolling_betas.get(20) is not None else static_beta
        beta_60 = rolling_betas.get(60) if rolling_betas.get(60) is not None else static_beta

        blended = (
            weights["ewma"] * beta_ewma + weights["rolling_20"] * beta_20 + weights["rolling_60"] * beta_60
        )

        return blended

    def calculate_for_symbol(
        self,
        symbol: str,
        bars: list[dict],
        spy_bars: list[dict],
    ) -> DynamicBetaResult:
        """Calculate dynamic beta for a symbol.

        Args:
            symbol: Asset symbol (e.g., "QQQ")
            bars: Historical OHLCV bars for asset
            spy_bars: Historical OHLCV bars for SPY

        Returns:
            DynamicBetaResult with calculated or fallback beta
        """
        static_beta = self.fallback_betas.get(symbol, 1.0)
        now = datetime.now().isoformat()

        # Check for sufficient data
        if len(bars) < self.MIN_DATA_DAYS or len(spy_bars) < self.MIN_DATA_DAYS:
            return DynamicBetaResult(
                symbol=symbol,
                beta_ewma=static_beta,
                beta_rolling_20=None,
                beta_rolling_60=None,
                beta_blended=static_beta,
                correlation_spy=None,
                data_days=min(len(bars), len(spy_bars)),
                calculated_at=now,
                is_fallback=True,
            )

        # Calculate returns
        asset_returns = self._calculate_returns(bars)
        market_returns = self._calculate_returns(spy_bars)

        # Align returns (use shorter length)
        min_len = min(len(asset_returns), len(market_returns))
        asset_returns = asset_returns[-min_len:]
        market_returns = market_returns[-min_len:]

        # Calculate betas
        beta_ewma = self.ewma_beta(asset_returns, market_returns, self.EWMA_HALFLIFE)
        rolling_betas = self.rolling_beta_multiwindow(asset_returns, market_returns)
        correlation = self.calculate_correlation(asset_returns, market_returns)

        # Blend
        beta_blended = self.blended_beta(beta_ewma, rolling_betas, static_beta)

        return DynamicBetaResult(
            symbol=symbol,
            beta_ewma=beta_ewma,
            beta_rolling_20=rolling_betas.get(20),
            beta_rolling_60=rolling_betas.get(60),
            beta_blended=beta_blended,
            correlation_spy=correlation,
            data_days=min_len + 1,  # +1 because returns are one less than bars
            calculated_at=now,
            is_fallback=False,
        )
