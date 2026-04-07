"""Tests for dynamic beta calculation module.

Tests the DynamicBetaCalculator class and related functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.risk.dynamic_beta import DynamicBetaCalculator, DynamicBetaResult


# Module-level fixtures
@pytest.fixture
def calculator():
    """Create calculator with default settings."""
    return DynamicBetaCalculator()


@pytest.fixture
def sample_bars():
    """Create sample OHLCV bars (80 days)."""
    np.random.seed(42)
    n_days = 80
    base_price = 100.0

    bars = []
    price = base_price
    for i in range(n_days):
        change = np.random.randn() * 0.02
        price *= 1 + change
        bars.append({
            "timestamp": f"2024-{(i // 30) + 10:02d}-{(i % 30) + 1:02d}T16:00:00Z",
            "open": price * 0.999,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000_000,
        })

    return bars


@pytest.fixture
def correlated_bars(sample_bars):
    """Create bars that are correlated with sample_bars (like QQQ to SPY)."""
    np.random.seed(43)
    bars = []
    for i, spy_bar in enumerate(sample_bars):
        # Beta of ~1.2 relative to sample_bars
        spy_return = np.log(spy_bar["close"] / sample_bars[i - 1]["close"]) if i > 0 else 0
        my_return = spy_return * 1.2 + np.random.randn() * 0.005

        price = bars[-1]["close"] * np.exp(my_return) if i > 0 else 100.0
        bars.append({
            "timestamp": spy_bar["timestamp"],
            "open": price * 0.999,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 500_000,
        })

    return bars


@pytest.fixture
def anticorrelated_bars(sample_bars):
    """Create bars that are negatively correlated (like TLT to SPY)."""
    np.random.seed(44)
    bars = []
    for i, spy_bar in enumerate(sample_bars):
        spy_return = np.log(spy_bar["close"] / sample_bars[i - 1]["close"]) if i > 0 else 0
        my_return = -spy_return * 0.3 + np.random.randn() * 0.008

        price = bars[-1]["close"] * np.exp(my_return) if i > 0 else 100.0
        bars.append({
            "timestamp": spy_bar["timestamp"],
            "open": price * 0.999,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": 200_000,
        })

    return bars


class TestDynamicBetaResult:
    """Test DynamicBetaResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = DynamicBetaResult(
            symbol="QQQ",
            beta_ewma=1.15,
            beta_rolling_20=1.10,
            beta_rolling_60=1.20,
            beta_blended=1.16,
            correlation_spy=0.92,
            data_days=80,
            calculated_at="2024-12-29T10:00:00",
            is_fallback=False,
        )

        d = result.to_dict()
        assert d["symbol"] == "QQQ"
        assert d["beta_ewma"] == 1.15
        assert d["beta_rolling_20"] == 1.10
        assert d["beta_rolling_60"] == 1.20
        assert d["beta_blended"] == 1.16
        assert d["correlation_spy"] == 0.92
        assert d["data_days"] == 80
        assert d["is_fallback"] is False

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "symbol": "TLT",
            "beta_ewma": -0.3,
            "beta_rolling_20": -0.25,
            "beta_rolling_60": -0.35,
            "beta_blended": -0.30,
            "correlation_spy": -0.4,
            "data_days": 65,
            "calculated_at": "2024-12-29T10:00:00",
            "is_fallback": False,
        }

        result = DynamicBetaResult.from_dict(data)
        assert result.symbol == "TLT"
        assert result.beta_ewma == -0.3
        assert result.beta_blended == -0.30
        assert result.is_fallback is False

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "symbol": "GLD",
            "beta_ewma": 0.1,
            "beta_blended": 0.1,
            "data_days": 30,
            "calculated_at": "2024-12-29T10:00:00",
        }

        result = DynamicBetaResult.from_dict(data)
        assert result.beta_rolling_20 is None
        assert result.beta_rolling_60 is None
        assert result.correlation_spy is None
        assert result.is_fallback is False


class TestCalculateReturns:
    """Test _calculate_returns method."""

    def test_basic_returns(self, calculator):
        """Test basic log return calculation."""
        bars = [
            {"close": 100.0},
            {"close": 101.0},
            {"close": 102.01},
        ]
        returns = calculator._calculate_returns(bars)

        # log(101/100) ~= 0.00995
        assert returns[0] == pytest.approx(np.log(101 / 100), rel=0.001)
        # log(102.01/101) ~= 0.00995
        assert returns[1] == pytest.approx(np.log(102.01 / 101), rel=0.001)

    def test_returns_length(self, calculator):
        """Test that returns array is one shorter than bars."""
        bars = [{"close": float(i)} for i in range(100, 150)]
        returns = calculator._calculate_returns(bars)

        assert len(returns) == len(bars) - 1


class TestEWMABeta:
    """Test ewma_beta method."""

    def test_perfect_correlation_beta_one(self, calculator):
        """Test that identical returns give beta ~1."""
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005] * 10)
        beta = calculator.ewma_beta(returns, returns)

        assert beta == pytest.approx(1.0, rel=0.01)

    def test_double_returns_beta_two(self, calculator):
        """Test that doubled returns give beta ~2."""
        market_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005] * 10)
        asset_returns = market_returns * 2
        beta = calculator.ewma_beta(asset_returns, market_returns)

        assert beta == pytest.approx(2.0, rel=0.1)

    def test_mismatched_length_raises(self, calculator):
        """Test that mismatched arrays raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            calculator.ewma_beta(np.array([1, 2, 3]), np.array([1, 2]))

    def test_insufficient_data_raises(self, calculator):
        """Test that insufficient data raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            calculator.ewma_beta(np.array([1.0]), np.array([1.0]))


class TestRollingBeta:
    """Test rolling_beta method."""

    def test_insufficient_window(self, calculator):
        """Test that insufficient data returns None."""
        returns = np.array([0.01] * 10)
        result = calculator.rolling_beta(returns, returns, window=20)

        assert result is None

    def test_sufficient_window(self, calculator):
        """Test with sufficient data."""
        np.random.seed(42)
        returns = np.random.randn(30) * 0.02
        result = calculator.rolling_beta(returns, returns, window=20)

        assert result is not None
        assert result == pytest.approx(1.0, rel=0.1)


class TestRollingBetaMultiwindow:
    """Test rolling_beta_multiwindow method."""

    def test_default_windows(self, calculator):
        """Test with default windows [20, 60]."""
        np.random.seed(42)
        returns = np.random.randn(80) * 0.02
        result = calculator.rolling_beta_multiwindow(returns, returns)

        assert 20 in result
        assert 60 in result
        assert result[20] is not None
        assert result[60] is not None

    def test_custom_windows(self, calculator):
        """Test with custom windows."""
        np.random.seed(42)
        returns = np.random.randn(50) * 0.02
        result = calculator.rolling_beta_multiwindow(returns, returns, windows=[10, 25])

        assert 10 in result
        assert 25 in result


class TestCalculateCorrelation:
    """Test calculate_correlation method."""

    def test_perfect_correlation(self, calculator):
        """Test identical returns give correlation ~1."""
        returns = np.array([0.01, -0.02, 0.015] * 10)
        corr = calculator.calculate_correlation(returns, returns)

        assert corr == pytest.approx(1.0, rel=0.01)

    def test_perfect_anticorrelation(self, calculator):
        """Test opposite returns give correlation ~-1."""
        returns = np.array([0.01, -0.02, 0.015] * 10)
        corr = calculator.calculate_correlation(returns, -returns)

        assert corr == pytest.approx(-1.0, rel=0.01)

    def test_insufficient_data(self, calculator):
        """Test insufficient data returns None."""
        returns = np.array([0.01] * 10)
        corr = calculator.calculate_correlation(returns, returns)

        assert corr is None


class TestBlendedBeta:
    """Test blended_beta method."""

    def test_all_components_available(self, calculator):
        """Test blending with all components."""
        beta_ewma = 1.2
        rolling_betas = {20: 1.1, 60: 1.3}
        static_beta = 1.0

        blended = calculator.blended_beta(beta_ewma, rolling_betas, static_beta)

        # 0.5 * 1.2 + 0.3 * 1.1 + 0.2 * 1.3 = 0.6 + 0.33 + 0.26 = 1.19
        expected = 0.5 * 1.2 + 0.3 * 1.1 + 0.2 * 1.3
        assert blended == pytest.approx(expected)

    def test_missing_rolling_60_uses_fallback(self, calculator):
        """Test that missing rolling_60 uses static fallback."""
        beta_ewma = 1.2
        rolling_betas = {20: 1.1, 60: None}
        static_beta = 1.0

        blended = calculator.blended_beta(beta_ewma, rolling_betas, static_beta)

        # 0.5 * 1.2 + 0.3 * 1.1 + 0.2 * 1.0 = 0.6 + 0.33 + 0.2 = 1.13
        expected = 0.5 * 1.2 + 0.3 * 1.1 + 0.2 * 1.0
        assert blended == pytest.approx(expected)


class TestCalculateForSymbol:
    """Test calculate_for_symbol method."""

    def test_correlated_symbol(self, calculator, sample_bars, correlated_bars):
        """Test beta calculation for correlated asset."""
        result = calculator.calculate_for_symbol("QQQ", correlated_bars, sample_bars)

        assert result.symbol == "QQQ"
        assert not result.is_fallback
        assert result.data_days >= 60
        # Beta should be positive and greater than 1 for QQQ-like asset
        assert result.beta_blended > 0.5
        assert result.correlation_spy is not None
        assert result.correlation_spy > 0

    def test_anticorrelated_symbol(self, calculator, sample_bars, anticorrelated_bars):
        """Test beta calculation for negatively correlated asset."""
        result = calculator.calculate_for_symbol("TLT", anticorrelated_bars, sample_bars)

        assert result.symbol == "TLT"
        assert not result.is_fallback
        # Beta should be negative for TLT-like asset
        assert result.beta_blended < 0.5
        # Correlation should be negative
        assert result.correlation_spy is not None
        assert result.correlation_spy < 0

    def test_insufficient_data_returns_fallback(self, calculator):
        """Test that insufficient data returns fallback beta."""
        short_bars = [{"close": 100.0 + i} for i in range(30)]
        spy_bars = [{"close": 450.0 + i} for i in range(30)]

        result = calculator.calculate_for_symbol("QQQ", short_bars, spy_bars)

        assert result.is_fallback
        assert result.data_days < 60
        # Should use static beta
        assert result.beta_blended == result.beta_ewma

    def test_custom_fallback_betas(self, sample_bars):
        """Test calculator with custom fallback betas."""
        custom = {"TEST": 0.5}
        calc = DynamicBetaCalculator(fallback_betas=custom)

        short_bars = [{"close": 100.0}] * 30
        spy_bars = [{"close": 450.0}] * 30

        result = calc.calculate_for_symbol("TEST", short_bars, spy_bars)

        assert result.is_fallback
        assert result.beta_blended == 0.5
