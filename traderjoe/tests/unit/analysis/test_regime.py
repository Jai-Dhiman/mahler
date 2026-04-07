"""Tests for market regime detection.

Tests the MarketRegimeDetector class and related functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.regime import (
    InsufficientDataError,
    MarketRegime,
    MarketRegimeDetector,
    RegimeResult,
)


class TestMarketRegimeDetector:
    """Test MarketRegimeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector with default settings."""
        return MarketRegimeDetector(lookback_days=60, n_regimes=4)

    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars for testing (80 days)."""
        np.random.seed(42)
        n_days = 80
        base_price = 450.0

        # Generate trending up data with some volatility
        returns = np.random.normal(0.0005, 0.01, n_days)
        prices = base_price * np.exp(np.cumsum(returns))

        bars = []
        for i in range(n_days):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_price = (high + low) / 2
            volume = int(np.random.normal(50_000_000, 10_000_000))

            bars.append({
                "timestamp": f"2024-{(i // 30) + 10:02d}-{(i % 30) + 1:02d}T16:00:00Z",
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": max(1_000_000, volume),
            })

        return bars

    @pytest.fixture
    def insufficient_bars(self):
        """Create insufficient bars (only 30 days)."""
        return [
            {
                "timestamp": f"2024-12-{i:02d}T16:00:00Z",
                "open": 450.0,
                "high": 455.0,
                "low": 445.0,
                "close": 450.0 + i * 0.5,
                "volume": 50_000_000,
            }
            for i in range(1, 31)
        ]


class TestComputeFeatures:
    """Test compute_features() method."""

    def test_computes_seven_features(self, detector, sample_bars):
        """Test that compute_features returns 7 features."""
        features = detector.compute_features(sample_bars, current_iv=0.20)

        assert features.shape[1] == 7, "Should have 7 features"
        assert features.shape[0] > 0, "Should have at least one sample"

    def test_raises_on_insufficient_data(self, detector, insufficient_bars):
        """Test that InsufficientDataError is raised with < 60 bars."""
        with pytest.raises(InsufficientDataError) as exc_info:
            detector.compute_features(insufficient_bars, current_iv=0.20)

        assert "60" in str(exc_info.value)
        assert "30" in str(exc_info.value)

    def test_features_are_finite(self, detector, sample_bars):
        """Test that all computed features are finite numbers."""
        features = detector.compute_features(sample_bars, current_iv=0.20)

        assert np.all(np.isfinite(features)), "All features should be finite"

    def test_realized_vol_positive(self, detector, sample_bars):
        """Test that realized volatility is positive."""
        features = detector.compute_features(sample_bars, current_iv=0.20)

        # First column is realized_vol_20
        realized_vol = features[:, 0]
        assert np.all(realized_vol > 0), "Realized vol should be positive"

    def test_volume_ratio_positive(self, detector, sample_bars):
        """Test that volume ratio is positive."""
        features = detector.compute_features(sample_bars, current_iv=0.20)

        # 6th column (index 5) is volume_ratio
        volume_ratio = features[:, 5]
        assert np.all(volume_ratio > 0), "Volume ratio should be positive"

    def test_current_iv_used_for_last_sample(self, detector, sample_bars):
        """Test that current_iv is used for the last observation."""
        test_iv = 0.35
        features = detector.compute_features(sample_bars, current_iv=test_iv)

        # 4th column (index 3) is iv
        last_iv = features[-1, 3]
        assert abs(last_iv - test_iv) < 0.01, f"Last IV should be {test_iv}, got {last_iv}"


class TestFitAndPredict:
    """Test fit_and_predict() method."""

    def test_returns_valid_regime(self, detector, sample_bars):
        """Test that fit_and_predict returns a valid regime."""
        features = detector.compute_features(sample_bars, current_iv=0.20)
        regime, probs, regime_map = detector.fit_and_predict(features)

        assert isinstance(regime, MarketRegime)
        assert regime in MarketRegime

    def test_returns_probability_array(self, detector, sample_bars):
        """Test that probabilities sum to ~1."""
        features = detector.compute_features(sample_bars, current_iv=0.20)
        regime, probs, regime_map = detector.fit_and_predict(features)

        assert len(probs) == 4, "Should have 4 regime probabilities"
        assert abs(sum(probs) - 1.0) < 0.01, "Probabilities should sum to 1"

    def test_regime_map_has_all_regimes(self, detector, sample_bars):
        """Test that regime map covers all cluster IDs."""
        features = detector.compute_features(sample_bars, current_iv=0.20)
        regime, probs, regime_map = detector.fit_and_predict(features)

        assert len(regime_map) == 4, "Should map all 4 clusters"
        assert all(isinstance(r, MarketRegime) for r in regime_map.values())


class TestGetPositionMultiplier:
    """Test get_position_multiplier() method."""

    def test_bull_low_vol_full_size(self, detector):
        """Test that BULL_LOW_VOL returns 1.0 multiplier."""
        mult = detector.get_position_multiplier(MarketRegime.BULL_LOW_VOL)
        assert mult == 1.0

    def test_bull_high_vol_reduced_size(self, detector):
        """Test that BULL_HIGH_VOL returns 0.75 multiplier."""
        mult = detector.get_position_multiplier(MarketRegime.BULL_HIGH_VOL)
        assert mult == 0.75

    def test_bear_low_vol_half_size(self, detector):
        """Test that BEAR_LOW_VOL returns 0.5 multiplier."""
        mult = detector.get_position_multiplier(MarketRegime.BEAR_LOW_VOL)
        assert mult == 0.5

    def test_bear_high_vol_reduced_size(self, detector):
        """Test that BEAR_HIGH_VOL returns 0.40 multiplier."""
        mult = detector.get_position_multiplier(MarketRegime.BEAR_HIGH_VOL)
        assert mult == 0.40

    def test_vix_override_at_40(self, detector):
        """Test that VIX > 40 overrides to 0.1."""
        mult = detector.get_position_multiplier(MarketRegime.BULL_LOW_VOL, current_vix=45.0)
        assert mult == 0.1

    def test_vix_override_takes_minimum(self, detector):
        """Test that VIX override uses min with regime multiplier."""
        # BEAR_HIGH_VOL is 0.40, VIX > 40 is 0.1
        mult = detector.get_position_multiplier(MarketRegime.BEAR_HIGH_VOL, current_vix=45.0)
        assert mult == 0.1

    def test_vix_below_40_no_override(self, detector):
        """Test that VIX < 40 doesn't override."""
        mult = detector.get_position_multiplier(MarketRegime.BULL_LOW_VOL, current_vix=35.0)
        assert mult == 1.0

    def test_vix_none_no_override(self, detector):
        """Test that None VIX doesn't override."""
        mult = detector.get_position_multiplier(MarketRegime.BULL_LOW_VOL, current_vix=None)
        assert mult == 1.0


class TestDetectRegime:
    """Test detect_regime() end-to-end method."""

    def test_returns_regime_result(self, detector, sample_bars):
        """Test that detect_regime returns a RegimeResult."""
        result = detector.detect_regime(sample_bars, current_iv=0.20, current_vix=18.0)

        assert isinstance(result, RegimeResult)
        assert isinstance(result.regime, MarketRegime)

    def test_result_has_all_fields(self, detector, sample_bars):
        """Test that RegimeResult has all required fields."""
        result = detector.detect_regime(sample_bars, current_iv=0.20, current_vix=18.0)

        assert result.regime is not None
        assert 0 <= result.probability <= 1
        assert isinstance(result.probabilities, dict)
        assert 0 <= result.position_multiplier <= 1
        assert isinstance(result.features, dict)
        assert result.detected_at is not None

    def test_result_features_match_names(self, detector, sample_bars):
        """Test that result features have expected keys."""
        result = detector.detect_regime(sample_bars, current_iv=0.20, current_vix=18.0)

        expected_keys = {
            "realized_vol_20",
            "momentum_20",
            "trend",
            "iv",
            "iv_rv_spread",
            "volume_ratio",
            "range_pct",
        }
        assert set(result.features.keys()) == expected_keys

    def test_result_to_dict_roundtrip(self, detector, sample_bars):
        """Test that to_dict and from_dict preserve data."""
        result = detector.detect_regime(sample_bars, current_iv=0.20, current_vix=18.0)

        data = result.to_dict()
        restored = RegimeResult.from_dict(data)

        assert restored.regime == result.regime
        assert restored.probability == result.probability
        assert restored.position_multiplier == result.position_multiplier
        assert restored.detected_at == result.detected_at

    def test_raises_on_insufficient_data(self, detector, insufficient_bars):
        """Test that detect_regime raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError):
            detector.detect_regime(insufficient_bars, current_iv=0.20)


class TestRegimeResult:
    """Test RegimeResult dataclass."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = RegimeResult(
            regime=MarketRegime.BULL_LOW_VOL,
            probability=0.85,
            probabilities={"bull_low_vol": 0.85, "bull_high_vol": 0.10},
            position_multiplier=1.0,
            features={"realized_vol_20": 0.15},
            detected_at="2024-12-29T10:00:00",
        )

        data = result.to_dict()

        assert data["regime"] == "bull_low_vol"
        assert data["probability"] == 0.85
        assert data["position_multiplier"] == 1.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "regime": "bear_high_vol",
            "probability": 0.75,
            "probabilities": {"bear_high_vol": 0.75},
            "position_multiplier": 0.25,
            "features": {"trend": -0.05},
            "detected_at": "2024-12-29T10:00:00",
        }

        result = RegimeResult.from_dict(data)

        assert result.regime == MarketRegime.BEAR_HIGH_VOL
        assert result.probability == 0.75
        assert result.position_multiplier == 0.25


class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_regime_values(self):
        """Test that regime enum has expected values."""
        assert MarketRegime.BULL_LOW_VOL.value == "bull_low_vol"
        assert MarketRegime.BULL_HIGH_VOL.value == "bull_high_vol"
        assert MarketRegime.BEAR_LOW_VOL.value == "bear_low_vol"
        assert MarketRegime.BEAR_HIGH_VOL.value == "bear_high_vol"

    def test_regime_from_string(self):
        """Test that regimes can be created from string values."""
        regime = MarketRegime("bull_low_vol")
        assert regime == MarketRegime.BULL_LOW_VOL


# Fixtures for detector and sample_bars used by multiple test classes
@pytest.fixture
def detector():
    """Create a detector with default settings."""
    return MarketRegimeDetector(lookback_days=60, n_regimes=4)


@pytest.fixture
def sample_bars():
    """Create sample OHLCV bars for testing (80 days)."""
    np.random.seed(42)
    n_days = 80
    base_price = 450.0

    returns = np.random.normal(0.0005, 0.01, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    bars = []
    for i in range(n_days):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = (high + low) / 2
        volume = int(np.random.normal(50_000_000, 10_000_000))

        bars.append({
            "timestamp": f"2024-{(i // 30) + 10:02d}-{(i % 30) + 1:02d}T16:00:00Z",
            "open": open_price,
            "high": max(high, open_price, close),
            "low": min(low, open_price, close),
            "close": close,
            "volume": max(1_000_000, volume),
        })

    return bars


@pytest.fixture
def insufficient_bars():
    """Create insufficient bars (only 30 days)."""
    return [
        {
            "timestamp": f"2024-12-{i:02d}T16:00:00Z",
            "open": 450.0,
            "high": 455.0,
            "low": 445.0,
            "close": 450.0 + i * 0.5,
            "volume": 50_000_000,
        }
        for i in range(1, 31)
    ]
