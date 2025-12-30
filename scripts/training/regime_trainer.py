"""Train regime detection model and export parameters.

Uses sklearn GaussianMixture and StandardScaler to train the regime
detection model, then exports parameters for numpy-only inference.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.analysis.regime import MarketRegime, MarketRegimeDetector


def train_regime_model(bars: list[dict], current_iv: float = 0.20) -> dict:
    """Train GMM + Scaler and export parameters.

    Args:
        bars: Historical OHLCV bars (at least 60 days, ideally 90+)
        current_iv: Current implied volatility estimate for feature computation

    Returns:
        Dictionary with model parameters for JSON serialization

    Raises:
        ValueError: If insufficient data for training
    """
    if len(bars) < 60:
        raise ValueError(f"Need at least 60 bars for training, got {len(bars)}")

    # Use existing detector to compute features (ensures consistency)
    detector = MarketRegimeDetector(lookback_days=60, n_regimes=4)

    # Compute features
    features = detector.compute_features(bars, current_iv)
    print(f"Computed {len(features)} feature samples from {len(bars)} bars")

    # Fit scaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=42,
        n_init=3,
        max_iter=100,
    )
    gmm.fit(scaled_features)

    # Get cluster labels to determine regime mapping
    labels = gmm.predict(scaled_features)

    # Characterize clusters
    cluster_chars = _characterize_clusters(features, labels)
    regime_map = _map_clusters_to_regimes(cluster_chars)

    # Export parameters
    params = {
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(features),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "gmm_means": gmm.means_.tolist(),
        "gmm_covariances": gmm.covariances_.tolist(),
        "gmm_weights": gmm.weights_.tolist(),
        "regime_mapping": {str(k): v.value for k, v in regime_map.items()},
    }

    # Print summary
    print("Regime mapping:")
    for cluster_id, regime in regime_map.items():
        char = cluster_chars[cluster_id]
        print(
            f"  Cluster {cluster_id} -> {regime.value}: "
            f"avg_vol={char['avg_vol']:.3f}, avg_trend={char['avg_trend']:.4f}, "
            f"count={char['count']}"
        )

    return params


def _characterize_clusters(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[int, dict]:
    """Characterize each cluster by mean volatility and trend."""
    chars = {}
    n_regimes = 4

    for cluster_id in range(n_regimes):
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
    cluster_chars: dict[int, dict],
) -> dict[int, MarketRegime]:
    """Map cluster IDs to regime names based on characteristics."""
    # Sort clusters by volatility (low to high)
    sorted_by_vol = sorted(
        cluster_chars.items(),
        key=lambda x: x[1]["avg_vol"],
    )

    # Low vol clusters (bottom 2), high vol clusters (top 2)
    n_low = 2
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


if __name__ == "__main__":

    # Test with sample data
    print("Regime trainer module loaded successfully")
    print("Run train_models.py to train the regime model")
