"""Numpy-only regime detection using pre-computed GMM parameters.

This module provides regime detection without sklearn/scipy dependencies,
using pre-trained model parameters loaded from R2.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from core.analysis.regime import InsufficientDataError, MarketRegime, RegimeResult
from core.inference.model_loader import RegimeModelParams


class PrecomputedRegimeDetector:
    """Regime detector using pre-computed GMM parameters.

    Performs numpy-only inference without sklearn/scipy.
    Compatible with Cloudflare Workers Python runtime.
    """

    # Position multipliers per regime (same as MarketRegimeDetector)
    REGIME_MULTIPLIERS = {
        MarketRegime.BULL_LOW_VOL: 1.0,
        MarketRegime.BULL_HIGH_VOL: 0.5,
        MarketRegime.BEAR_LOW_VOL: 0.5,
        MarketRegime.BEAR_HIGH_VOL: 0.25,
    }

    FEATURE_NAMES = [
        "realized_vol_20",
        "momentum_20",
        "trend",
        "iv",
        "iv_rv_spread",
        "volume_ratio",
        "range_pct",
    ]

    def __init__(self, params: RegimeModelParams):
        """Initialize with pre-computed parameters.

        Args:
            params: Pre-trained model parameters from R2
        """
        self.params = params

        # Convert lists to numpy arrays for efficient computation
        self.scaler_mean = np.array(params.scaler_mean)
        self.scaler_scale = np.array(params.scaler_scale)
        self.gmm_means = np.array(params.gmm_means)
        self.gmm_covariances = np.array(params.gmm_covariances)
        self.gmm_weights = np.array(params.gmm_weights)

        # Precompute inverse covariances and log determinants for efficiency
        self._inv_covariances = []
        self._log_dets = []
        for cov in self.gmm_covariances:
            self._inv_covariances.append(np.linalg.inv(cov))
            _, logdet = np.linalg.slogdet(cov)
            self._log_dets.append(logdet)

    def compute_features(
        self,
        bars: list[dict],
        current_iv: float,
    ) -> np.ndarray:
        """Compute 7 features for regime detection.

        This is the same logic as MarketRegimeDetector.compute_features().

        Args:
            bars: Historical OHLCV data (oldest first), each with keys:
                  timestamp, open, high, low, close, volume
            current_iv: Current implied volatility (decimal, e.g., 0.20 for 20%)

        Returns:
            numpy array of shape (n_samples, 7) with features

        Raises:
            InsufficientDataError: If insufficient historical data
        """
        lookback_days = 60
        if len(bars) < lookback_days:
            raise InsufficientDataError(
                f"Need at least {lookback_days} bars, got {len(bars)}"
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
            # Use current_iv for the last observation, estimate historical as RV * 1.1
            iv = current_iv if i == n_samples - 1 else realized_vol * 1.1

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

    def _scale(self, features: np.ndarray) -> np.ndarray:
        """Apply StandardScaler transform using pre-computed parameters.

        Args:
            features: Raw feature vector (1D array of 7 values)

        Returns:
            Scaled feature vector
        """
        return (features - self.scaler_mean) / self.scaler_scale

    def _gmm_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability for each GMM component.

        Implements multivariate Gaussian log PDF:
        log P(x|k) = log(w_k) - 0.5 * [d*log(2pi) + log|Sigma_k| + (x-mu_k)^T Sigma_k^-1 (x-mu_k)]

        Args:
            x: Scaled feature vector (1D array of 7 values)

        Returns:
            Array of log probabilities for each component
        """
        n_components = len(self.gmm_weights)
        d = x.shape[0]
        log_probs = np.zeros(n_components)

        for k in range(n_components):
            diff = x - self.gmm_means[k]
            mahal = diff @ self._inv_covariances[k] @ diff
            log_probs[k] = (
                np.log(self.gmm_weights[k])
                - 0.5 * (d * np.log(2 * np.pi) + self._log_dets[k] + mahal)
            )

        return log_probs

    def predict(self, features: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict cluster and probabilities for single observation.

        Args:
            features: Raw feature vector (1D array of 7 values)

        Returns:
            Tuple of (cluster_id, probability_array)
        """
        scaled = self._scale(features)
        log_probs = self._gmm_log_prob(scaled)

        # Convert to probabilities via log-sum-exp trick (numerically stable softmax)
        max_log = np.max(log_probs)
        exp_probs = np.exp(log_probs - max_log)
        probs = exp_probs / np.sum(exp_probs)

        cluster_id = int(np.argmax(probs))
        return cluster_id, probs

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

        # Use only the last observation for prediction
        current_features = features[-1]
        cluster_id, probs = self.predict(current_features)

        # Map cluster to regime using stored mapping
        regime_str = self.params.regime_mapping[str(cluster_id)]
        regime = MarketRegime(regime_str)

        # Get multiplier
        multiplier = self.get_position_multiplier(regime, current_vix)

        # Build probability dict using regime mapping
        prob_dict: dict[str, float] = {}
        for k_str, regime_name in self.params.regime_mapping.items():
            k = int(k_str)
            if regime_name not in prob_dict:
                prob_dict[regime_name] = 0.0
            prob_dict[regime_name] += float(probs[k])

        # Extract current feature values
        feature_dict = {
            name: float(current_features[i]) for i, name in enumerate(self.FEATURE_NAMES)
        }

        return RegimeResult(
            regime=regime,
            probability=float(probs[cluster_id]),
            probabilities=prob_dict,
            position_multiplier=multiplier,
            features=feature_dict,
            detected_at=datetime.now().isoformat(),
        )
