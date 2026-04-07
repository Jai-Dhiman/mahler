"""Tests for numpy-only regime inference."""

import numpy as np
import pytest

from core.analysis.regime import MarketRegime, RegimeResult
from core.inference.model_loader import RegimeModelParams
from core.inference.regime_inference import PrecomputedRegimeDetector


@pytest.fixture
def sample_params():
    """Create sample model parameters for testing."""
    # These are realistic-ish parameters for testing
    np.random.seed(42)

    n_features = 7
    n_components = 4

    # Generate some reasonable means and covariances
    means = [
        [0.12, 0.02, 0.01, 0.15, 0.03, 1.0, 0.012],  # bull_low_vol
        [0.25, 0.03, 0.02, 0.30, 0.05, 1.2, 0.020],  # bull_high_vol
        [0.14, -0.02, -0.01, 0.18, 0.04, 0.9, 0.014],  # bear_low_vol
        [0.30, -0.04, -0.02, 0.35, 0.05, 1.3, 0.025],  # bear_high_vol
    ]

    # Generate positive definite covariances
    covariances = []
    for _ in range(n_components):
        # Create a random matrix and multiply by its transpose
        A = np.random.randn(n_features, n_features) * 0.01
        cov = A @ A.T + np.eye(n_features) * 0.001
        covariances.append(cov.tolist())

    return RegimeModelParams(
        version="1.0.0",
        trained_at="2025-01-06T10:00:00",
        n_samples=1000,
        scaler_mean=[0.15, 0.01, 0.005, 0.20, 0.04, 1.0, 0.015],
        scaler_scale=[0.08, 0.03, 0.02, 0.10, 0.03, 0.3, 0.008],
        gmm_means=means,
        gmm_covariances=covariances,
        gmm_weights=[0.25, 0.25, 0.25, 0.25],
        regime_mapping={
            "0": "bull_low_vol",
            "1": "bull_high_vol",
            "2": "bear_low_vol",
            "3": "bear_high_vol",
        },
    )


@pytest.fixture
def sample_bars():
    """Create sample OHLCV bars for testing."""
    np.random.seed(42)
    n_bars = 80

    # Generate random walk for close prices
    returns = np.random.randn(n_bars) * 0.01
    closes = 100 * np.exp(np.cumsum(returns))

    bars = []
    for i in range(n_bars):
        close = closes[i]
        # Add some random variation for high/low
        high = close * (1 + abs(np.random.randn()) * 0.01)
        low = close * (1 - abs(np.random.randn()) * 0.01)
        open_price = (high + low) / 2

        bars.append({
            "timestamp": f"2025-01-{i+1:02d}T16:00:00Z",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": int(1000000 + np.random.randn() * 100000),
        })

    return bars


class TestPrecomputedRegimeDetector:
    """Test PrecomputedRegimeDetector class."""

    def test_init_precomputes_inverses(self, sample_params):
        """Verify that __init__ precomputes inverse covariances."""
        detector = PrecomputedRegimeDetector(sample_params)

        assert len(detector._inv_covariances) == 4
        assert len(detector._log_dets) == 4

        # Verify inverse is correct
        for i, (cov, inv) in enumerate(
            zip(detector.gmm_covariances, detector._inv_covariances)
        ):
            identity = cov @ inv
            np.testing.assert_allclose(
                identity, np.eye(7), atol=1e-10,
                err_msg=f"Inverse covariance {i} is incorrect"
            )

    def test_scale_produces_correct_output(self, sample_params):
        """Verify _scale produces correctly shaped output."""
        detector = PrecomputedRegimeDetector(sample_params)

        # Test with a single feature vector
        features = np.array([0.15, 0.01, 0.005, 0.20, 0.04, 1.0, 0.015])
        scaled = detector._scale(features)

        assert scaled.shape == (7,)
        # At the mean, scaled should be close to zero
        np.testing.assert_allclose(scaled, np.zeros(7), atol=1e-10)

    def test_scale_matches_sklearn(self, sample_params):
        """Verify numpy scaling matches sklearn StandardScaler."""
        pytest.importorskip("sklearn")
        from sklearn.preprocessing import StandardScaler

        detector = PrecomputedRegimeDetector(sample_params)

        # Create sklearn scaler with same parameters
        scaler = StandardScaler()
        scaler.mean_ = np.array(sample_params.scaler_mean)
        scaler.scale_ = np.array(sample_params.scaler_scale)

        # Test with random features
        np.random.seed(123)
        features = np.random.randn(7) * 0.1 + 0.2

        sklearn_scaled = scaler.transform(features.reshape(1, -1))[0]
        numpy_scaled = detector._scale(features)

        np.testing.assert_allclose(sklearn_scaled, numpy_scaled, rtol=1e-10)

    def test_gmm_log_prob_shape(self, sample_params):
        """Verify _gmm_log_prob returns correct shape."""
        detector = PrecomputedRegimeDetector(sample_params)

        scaled = np.zeros(7)  # At the mean
        log_probs = detector._gmm_log_prob(scaled)

        assert log_probs.shape == (4,)
        # All log probs should be finite
        assert np.all(np.isfinite(log_probs))

    def test_gmm_log_prob_values_reasonable(self, sample_params):
        """Verify _gmm_log_prob produces reasonable values."""
        detector = PrecomputedRegimeDetector(sample_params)

        # Test at mean (should have relatively high probability)
        scaled_mean = np.zeros(7)
        log_probs_mean = detector._gmm_log_prob(scaled_mean)

        # Test far from mean (should have lower probability)
        scaled_far = np.ones(7) * 10
        log_probs_far = detector._gmm_log_prob(scaled_far)

        # At least one component should have higher prob at mean
        assert np.max(log_probs_mean) > np.max(log_probs_far)

    def test_predict_returns_valid_cluster(self, sample_params):
        """Verify predict returns valid cluster and probabilities."""
        detector = PrecomputedRegimeDetector(sample_params)

        features = np.array([0.15, 0.01, 0.005, 0.20, 0.04, 1.0, 0.015])
        cluster_id, probs = detector.predict(features)

        # Cluster ID should be 0-3
        assert 0 <= cluster_id <= 3

        # Probabilities should sum to 1
        assert probs.shape == (4,)
        np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)

        # All probabilities should be non-negative
        assert np.all(probs >= 0)

        # Cluster ID should be argmax
        assert cluster_id == np.argmax(probs)

    def test_compute_features_shape(self, sample_params, sample_bars):
        """Verify compute_features returns correct shape."""
        detector = PrecomputedRegimeDetector(sample_params)

        features = detector.compute_features(sample_bars, current_iv=0.20)

        # With 80 bars and min_idx=50, we should have 30 samples
        expected_samples = len(sample_bars) - 50
        assert features.shape == (expected_samples, 7)

    def test_compute_features_insufficient_data(self, sample_params):
        """Verify compute_features raises error for insufficient data."""
        from core.analysis.regime import InsufficientDataError

        detector = PrecomputedRegimeDetector(sample_params)

        # Only 50 bars (need at least 60)
        bars = [{"close": 100, "high": 101, "low": 99, "volume": 1000000}] * 50

        with pytest.raises(InsufficientDataError):
            detector.compute_features(bars, current_iv=0.20)

    def test_detect_regime_returns_regime_result(self, sample_params, sample_bars):
        """Verify detect_regime returns proper RegimeResult."""
        detector = PrecomputedRegimeDetector(sample_params)

        result = detector.detect_regime(sample_bars, current_iv=0.20, current_vix=15.0)

        assert isinstance(result, RegimeResult)
        assert isinstance(result.regime, MarketRegime)
        assert 0 <= result.probability <= 1
        assert 0 < result.position_multiplier <= 1
        assert len(result.features) == 7
        assert len(result.probabilities) == 4

    def test_detect_regime_high_vix_reduces_multiplier(self, sample_params, sample_bars):
        """Verify high VIX reduces position multiplier."""
        detector = PrecomputedRegimeDetector(sample_params)

        result_low_vix = detector.detect_regime(
            sample_bars, current_iv=0.20, current_vix=15.0
        )
        result_high_vix = detector.detect_regime(
            sample_bars, current_iv=0.20, current_vix=45.0
        )

        # High VIX should cap multiplier at 0.1
        assert result_high_vix.position_multiplier <= 0.1
        # Low VIX result should be >= high VIX result
        assert result_low_vix.position_multiplier >= result_high_vix.position_multiplier

    def test_regime_multipliers(self, sample_params):
        """Verify REGIME_MULTIPLIERS has correct values."""
        detector = PrecomputedRegimeDetector(sample_params)

        expected = {
            MarketRegime.BULL_LOW_VOL: 1.0,
            MarketRegime.BULL_HIGH_VOL: 0.5,
            MarketRegime.BEAR_LOW_VOL: 0.5,
            MarketRegime.BEAR_HIGH_VOL: 0.25,
        }

        assert detector.REGIME_MULTIPLIERS == expected

    def test_feature_names(self, sample_params):
        """Verify FEATURE_NAMES are correct."""
        detector = PrecomputedRegimeDetector(sample_params)

        expected = [
            "realized_vol_20",
            "momentum_20",
            "trend",
            "iv",
            "iv_rv_spread",
            "volume_ratio",
            "range_pct",
        ]

        assert detector.FEATURE_NAMES == expected


class TestPrecomputedRegimeDetectorConsistency:
    """Test consistency between PrecomputedRegimeDetector and MarketRegimeDetector."""

    def test_compute_features_matches_sklearn_version(self, sample_params, sample_bars):
        """Verify compute_features produces same results as sklearn version."""
        pytest.importorskip("sklearn")
        from core.analysis.regime import MarketRegimeDetector

        precomputed = PrecomputedRegimeDetector(sample_params)
        sklearn_detector = MarketRegimeDetector()

        precomputed_features = precomputed.compute_features(sample_bars, current_iv=0.20)
        sklearn_features = sklearn_detector.compute_features(sample_bars, current_iv=0.20)

        np.testing.assert_allclose(
            precomputed_features,
            sklearn_features,
            rtol=1e-10,
            err_msg="Feature computation differs between versions"
        )
