"""Tests for weight inference module."""

import pytest

from core.inference.model_loader import WeightModelParams
from core.inference.weight_inference import PrecomputedWeightProvider, ScoringWeights


@pytest.fixture
def sample_params():
    """Create sample weight model parameters."""
    return WeightModelParams(
        version="1.0.0",
        trained_at="2025-01-06T10:00:00",
        n_samples=500,
        weights_by_regime={
            "bull_low_vol": {"iv": 0.30, "delta": 0.25, "credit": 0.25, "ev": 0.20},
            "bull_high_vol": {"iv": 0.35, "delta": 0.20, "credit": 0.25, "ev": 0.20},
            "bear_low_vol": {"iv": 0.25, "delta": 0.30, "credit": 0.25, "ev": 0.20},
            "bear_high_vol": {"iv": 0.20, "delta": 0.35, "credit": 0.25, "ev": 0.20},
        },
        sharpe_by_regime={
            "bull_low_vol": 1.5,
            "bull_high_vol": 1.2,
            "bear_low_vol": 0.8,
            "bear_high_vol": 0.5,
        },
    )


class TestScoringWeights:
    """Test ScoringWeights dataclass."""

    def test_default_values(self):
        """Verify default weights sum to 1."""
        weights = ScoringWeights()

        total = (
            weights.iv_weight
            + weights.delta_weight
            + weights.credit_weight
            + weights.ev_weight
        )

        assert total == pytest.approx(1.0)
        assert weights.iv_weight == 0.25
        assert weights.delta_weight == 0.25
        assert weights.credit_weight == 0.25
        assert weights.ev_weight == 0.25


class TestPrecomputedWeightProvider:
    """Test PrecomputedWeightProvider class."""

    def test_get_weights_with_params(self, sample_params):
        """Verify get_weights returns optimized weights."""
        provider = PrecomputedWeightProvider(sample_params)

        weights = provider.get_weights("bull_low_vol")

        assert isinstance(weights, ScoringWeights)
        assert weights.iv_weight == 0.30
        assert weights.delta_weight == 0.25
        assert weights.credit_weight == 0.25
        assert weights.ev_weight == 0.20

    def test_get_weights_unknown_regime_returns_defaults(self, sample_params):
        """Verify unknown regime returns default weights."""
        provider = PrecomputedWeightProvider(sample_params)

        weights = provider.get_weights("unknown_regime")

        assert weights == provider.DEFAULT_WEIGHTS

    def test_get_weights_with_none_params_returns_defaults(self):
        """Verify None params returns default weights."""
        provider = PrecomputedWeightProvider(None)

        weights = provider.get_weights("bull_low_vol")

        assert weights == provider.DEFAULT_WEIGHTS

    def test_get_all_weights(self, sample_params):
        """Verify get_all_weights returns all regimes."""
        provider = PrecomputedWeightProvider(sample_params)

        all_weights = provider.get_all_weights()

        assert len(all_weights) == 4
        assert "bull_low_vol" in all_weights
        assert "bull_high_vol" in all_weights
        assert "bear_low_vol" in all_weights
        assert "bear_high_vol" in all_weights

    def test_get_all_weights_with_none_params(self):
        """Verify get_all_weights with None params returns empty dict."""
        provider = PrecomputedWeightProvider(None)

        all_weights = provider.get_all_weights()

        assert all_weights == {}

    def test_get_sharpe_ratio(self, sample_params):
        """Verify get_sharpe_ratio returns correct value."""
        provider = PrecomputedWeightProvider(sample_params)

        sharpe = provider.get_sharpe_ratio("bull_low_vol")

        assert sharpe == 1.5

    def test_get_sharpe_ratio_unknown_regime(self, sample_params):
        """Verify unknown regime returns None for Sharpe."""
        provider = PrecomputedWeightProvider(sample_params)

        sharpe = provider.get_sharpe_ratio("unknown_regime")

        assert sharpe is None

    def test_get_sharpe_ratio_with_none_params(self):
        """Verify None params returns None for Sharpe."""
        provider = PrecomputedWeightProvider(None)

        sharpe = provider.get_sharpe_ratio("bull_low_vol")

        assert sharpe is None

    def test_has_weights_for_regime(self, sample_params):
        """Verify has_weights_for_regime works correctly."""
        provider = PrecomputedWeightProvider(sample_params)

        assert provider.has_weights_for_regime("bull_low_vol") is True
        assert provider.has_weights_for_regime("unknown_regime") is False

    def test_has_weights_for_regime_with_none_params(self):
        """Verify has_weights_for_regime with None params returns False."""
        provider = PrecomputedWeightProvider(None)

        assert provider.has_weights_for_regime("bull_low_vol") is False

    def test_default_weights_are_correct(self):
        """Verify DEFAULT_WEIGHTS has correct values."""
        expected = ScoringWeights(
            iv_weight=0.25,
            delta_weight=0.25,
            credit_weight=0.25,
            ev_weight=0.25,
        )

        assert PrecomputedWeightProvider.DEFAULT_WEIGHTS == expected
