"""Market regime detection using GaussianMixture clustering.

Detects market regimes based on 7 features:
1. realized_vol_20: 20-day annualized realized volatility
2. momentum_20: 20-day cumulative return
3. trend: (SMA20 - SMA50) / SMA50
4. iv: Current implied volatility
5. iv_rv_spread: IV minus realized volatility
6. volume_ratio: Current volume / 20-day average
7. range_pct: (high - low) / close

Maps to 4 regimes with position sizing multipliers:
- BULL_LOW_VOL: 1.0 (full size)
- BULL_HIGH_VOL: 0.5 (reduce exposure)
- BEAR_LOW_VOL: 0.5 (cautious)
- BEAR_HIGH_VOL: 0.25 (defensive)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.db.kv import KVClient
    from core.inference.regime_inference import PrecomputedRegimeDetector


class MarketRegime(str, Enum):
    """Market regime categories for position sizing."""

    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"


@dataclass
class RegimeResult:
    """Result of regime detection."""

    regime: MarketRegime
    probability: float  # Confidence in detected regime (0-1)
    probabilities: dict[str, float]  # All regime probabilities
    position_multiplier: float
    features: dict[str, float]  # Feature values used for current observation
    detected_at: str  # ISO timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary for caching/storage."""
        return {
            "regime": self.regime.value,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "position_multiplier": self.position_multiplier,
            "features": self.features,
            "detected_at": self.detected_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegimeResult:
        """Create from dictionary."""
        return cls(
            regime=MarketRegime(data["regime"]),
            probability=data["probability"],
            probabilities=data["probabilities"],
            position_multiplier=data["position_multiplier"],
            features=data["features"],
            detected_at=data["detected_at"],
        )


class InsufficientDataError(Exception):
    """Raised when insufficient historical data for regime detection."""

    pass


class MarketRegimeDetector:
    """Detects market regime using GaussianMixture clustering.

    Uses 7 market features to classify into 4 regimes:
    - BULL_LOW_VOL: Trending up, low volatility (full size)
    - BULL_HIGH_VOL: Trending up, high volatility (reduce exposure)
    - BEAR_LOW_VOL: Trending down, low volatility (cautious)
    - BEAR_HIGH_VOL: Trending down, high volatility (defensive)
    """

    # Position multipliers per regime
    REGIME_MULTIPLIERS = {
        MarketRegime.BULL_LOW_VOL: 1.0,
        MarketRegime.BULL_HIGH_VOL: 0.75,
        MarketRegime.BEAR_LOW_VOL: 0.5,
        MarketRegime.BEAR_HIGH_VOL: 0.40,
    }

    # Feature names for output
    FEATURE_NAMES = [
        "realized_vol_20",
        "momentum_20",
        "trend",
        "iv",
        "iv_rv_spread",
        "volume_ratio",
        "range_pct",
    ]

    def __init__(
        self,
        lookback_days: int = 60,
        n_regimes: int = 4,
    ):
        """Initialize regime detector.

        Args:
            lookback_days: Minimum days of data required
            n_regimes: Number of regimes to detect (default 4)
        """
        # Lazy import sklearn (only used in dev mode, not in production workers)
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        self.lookback_days = lookback_days
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=42,
            n_init=3,
            max_iter=100,
        )

    def compute_features(
        self,
        bars: list[dict],
        current_iv: float,
    ) -> np.ndarray:
        """Compute 7 features for regime detection.

        Args:
            bars: Historical OHLCV data (oldest first), each with keys:
                  timestamp, open, high, low, close, volume
            current_iv: Current implied volatility (decimal, e.g., 0.20 for 20%)

        Returns:
            numpy array of shape (n_samples, 7) with features

        Raises:
            InsufficientDataError: If len(bars) < lookback_days
        """
        if len(bars) < self.lookback_days:
            raise InsufficientDataError(
                f"Need at least {self.lookback_days} bars, got {len(bars)}"
            )

        # Extract price/volume arrays
        closes = np.array([bar["close"] for bar in bars])
        highs = np.array([bar["high"] for bar in bars])
        lows = np.array([bar["low"] for bar in bars])
        volumes = np.array([bar["volume"] for bar in bars], dtype=float)

        # Calculate log returns
        returns = np.diff(np.log(closes))

        # We need enough data for 50-day SMA, so start at index 50
        min_idx = 50
        n_samples = len(closes) - min_idx

        if n_samples < 1:
            raise InsufficientDataError(
                f"Need at least 51 bars for SMA50, got {len(bars)}"
            )

        features = np.zeros((n_samples, 7))

        for i in range(n_samples):
            idx = min_idx + i  # Current bar index

            # Feature 1: realized_vol_20 (20-day annualized volatility)
            if idx >= 20:
                vol_returns = returns[idx - 20 : idx]
                realized_vol = np.std(vol_returns) * np.sqrt(252)
            else:
                realized_vol = np.std(returns[:idx]) * np.sqrt(252) if idx > 0 else 0.15

            # Feature 2: momentum_20 (20-day cumulative return)
            if idx >= 21:
                momentum = (closes[idx] / closes[idx - 21]) - 1
            else:
                momentum = (closes[idx] / closes[0]) - 1 if closes[0] > 0 else 0

            # Feature 3: trend (SMA20 - SMA50) / SMA50
            sma20 = np.mean(closes[idx - 20 : idx]) if idx >= 20 else np.mean(closes[:idx])
            sma50 = np.mean(closes[idx - 50 : idx]) if idx >= 50 else np.mean(closes[:idx])
            trend = (sma20 - sma50) / sma50 if sma50 > 0 else 0

            # Feature 4: iv (current implied volatility)
            # Use current_iv for the last observation, estimate for historical
            if i == n_samples - 1:
                iv = current_iv
            else:
                # Estimate historical IV as realized vol * 1.1 (IV typically > RV)
                iv = realized_vol * 1.1

            # Feature 5: iv_rv_spread
            iv_rv_spread = iv - realized_vol

            # Feature 6: volume_ratio (current / 20-day average)
            if idx >= 20:
                avg_volume = np.mean(volumes[idx - 20 : idx])
            else:
                avg_volume = np.mean(volumes[:idx]) if idx > 0 else volumes[idx]
            volume_ratio = volumes[idx] / avg_volume if avg_volume > 0 else 1.0

            # Feature 7: range_pct (daily range as % of close)
            range_pct = (highs[idx] - lows[idx]) / closes[idx] if closes[idx] > 0 else 0

            features[i] = [
                realized_vol,
                momentum,
                trend,
                iv,
                iv_rv_spread,
                volume_ratio,
                range_pct,
            ]

        return features

    def fit_and_predict(
        self,
        features: np.ndarray,
    ) -> tuple[MarketRegime, np.ndarray, dict[int, MarketRegime]]:
        """Fit GMM and predict current regime.

        Args:
            features: Feature array from compute_features()

        Returns:
            Tuple of (current regime, probability array, cluster to regime mapping)
        """
        # Scale features
        scaled = self.scaler.fit_transform(features)

        # Fit GMM
        self.gmm.fit(scaled)

        # Get cluster labels and probabilities
        labels = self.gmm.predict(scaled)
        probs = self.gmm.predict_proba(scaled)

        # Characterize clusters by average volatility and trend
        cluster_chars = self._characterize_clusters(features, labels)

        # Map clusters to regime names
        regime_map = self._map_clusters_to_regimes(cluster_chars)

        # Get current regime (last observation)
        current_cluster = labels[-1]
        current_probs = probs[-1]

        return regime_map[current_cluster], current_probs, regime_map

    def _characterize_clusters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[int, dict]:
        """Characterize each cluster by mean volatility and trend."""
        chars = {}
        for cluster_id in range(self.n_regimes):
            mask = labels == cluster_id
            if np.sum(mask) == 0:
                # Empty cluster - use neutral characteristics
                chars[cluster_id] = {
                    "avg_vol": 0.15,
                    "avg_trend": 0.0,
                    "count": 0,
                }
                continue

            cluster_features = features[mask]

            chars[cluster_id] = {
                "avg_vol": float(np.mean(cluster_features[:, 0])),  # realized_vol_20
                "avg_trend": float(np.mean(cluster_features[:, 2])),  # trend
                "count": int(np.sum(mask)),
            }
        return chars

    def _map_clusters_to_regimes(
        self,
        cluster_chars: dict[int, dict],
    ) -> dict[int, MarketRegime]:
        """Map cluster IDs to regime names based on characteristics."""
        # Sort clusters by volatility (low to high)
        sorted_by_vol = sorted(
            cluster_chars.items(),
            key=lambda x: x[1]["avg_vol"],
        )

        # Low vol clusters (bottom 2), high vol clusters (top 2)
        n_low = self.n_regimes // 2
        low_vol_clusters = {c[0] for c in sorted_by_vol[:n_low]}

        regime_map = {}
        for cluster_id, chars in cluster_chars.items():
            is_low_vol = cluster_id in low_vol_clusters
            is_bullish = chars["avg_trend"] > 0

            if is_bullish and is_low_vol:
                regime_map[cluster_id] = MarketRegime.BULL_LOW_VOL
            elif is_bullish and not is_low_vol:
                regime_map[cluster_id] = MarketRegime.BULL_HIGH_VOL
            elif not is_bullish and is_low_vol:
                regime_map[cluster_id] = MarketRegime.BEAR_LOW_VOL
            else:
                regime_map[cluster_id] = MarketRegime.BEAR_HIGH_VOL

        return regime_map

    def get_position_multiplier(
        self,
        regime: MarketRegime,
        current_vix: float | None = None,
    ) -> float:
        """Get position sizing multiplier for regime.

        Args:
            regime: Detected market regime
            current_vix: Current VIX level (optional)

        Returns:
            Position size multiplier (0.1 to 1.0)
        """
        # Base multiplier from regime
        multiplier = self.REGIME_MULTIPLIERS[regime]

        # VIX override (hard safety limit)
        if current_vix is not None and current_vix > 40.0:
            multiplier = min(multiplier, 0.1)

        return multiplier

    def detect_regime(
        self,
        bars: list[dict],
        current_iv: float,
        current_vix: float | None = None,
    ) -> RegimeResult:
        """Full regime detection pipeline.

        Args:
            bars: Historical OHLCV data from broker
            current_iv: Current implied volatility (decimal)
            current_vix: Current VIX level (optional)

        Returns:
            RegimeResult with regime, probabilities, and multiplier

        Raises:
            InsufficientDataError: If insufficient historical data
        """
        # Compute features
        features = self.compute_features(bars, current_iv)

        # Fit and predict
        regime, probs, regime_map = self.fit_and_predict(features)

        # Get multiplier
        multiplier = self.get_position_multiplier(regime, current_vix)

        # Build probability dict using regime names
        prob_dict = {}
        for cluster_id, cluster_regime in regime_map.items():
            regime_name = cluster_regime.value
            if regime_name not in prob_dict:
                prob_dict[regime_name] = 0.0
            prob_dict[regime_name] += probs[cluster_id]

        # Extract current feature values
        current_features = features[-1]
        feature_dict = {
            name: float(current_features[i])
            for i, name in enumerate(self.FEATURE_NAMES)
        }

        return RegimeResult(
            regime=regime,
            probability=float(probs[list(regime_map.keys())[list(regime_map.values()).index(regime)]]),
            probabilities=prob_dict,
            position_multiplier=multiplier,
            features=feature_dict,
            detected_at=datetime.now().isoformat(),
        )


async def create_regime_detector(
    env: Any,
    kv: KVClient | None = None,
) -> MarketRegimeDetector | PrecomputedRegimeDetector:
    """Factory to create appropriate regime detector.

    In production (MAHLER_BUCKET binding exists): Uses pre-computed parameters
    In development: Uses full sklearn implementation

    Args:
        env: Cloudflare environment with bindings
        kv: Optional KV client for caching

    Returns:
        Regime detector instance (MarketRegimeDetector or PrecomputedRegimeDetector)
    """
    # Check if we have the models bucket binding (production mode)
    models_bucket = getattr(env, "MAHLER_BUCKET", None)

    if models_bucket is None:
        # Development mode: use full sklearn implementation
        return MarketRegimeDetector()

    # Production mode: use pre-computed parameters
    from core.inference.model_loader import ModelLoader
    from core.inference.regime_inference import PrecomputedRegimeDetector

    loader = ModelLoader(models_bucket, kv)
    params = await loader.get_regime_params()

    if params is None:
        # Fallback to sklearn if no model available
        print("Warning: No pre-computed regime model found, using sklearn")
        return MarketRegimeDetector()

    return PrecomputedRegimeDetector(params)
