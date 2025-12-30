"""Tests for technical indicators module.

Tests the indicator calculations for correctness and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.indicators import (
    calculate_atr,
    calculate_bollinger_position,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    normalize_macd_signal,
    normalize_rsi_signal,
)


class TestSMA:
    """Test calculate_sma function."""

    def test_basic_sma(self):
        """Test basic SMA calculation."""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_sma(prices, period=3)

        # First 2 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # SMA at index 2: (1+2+3)/3 = 2
        assert result[2] == pytest.approx(2.0)
        # SMA at index 3: (2+3+4)/3 = 3
        assert result[3] == pytest.approx(3.0)
        # SMA at index 4: (3+4+5)/3 = 4
        assert result[4] == pytest.approx(4.0)

    def test_insufficient_data(self):
        """Test SMA with insufficient data."""
        prices = np.array([1.0, 2.0])
        result = calculate_sma(prices, period=5)

        assert len(result) == 2
        assert np.all(np.isnan(result))

    def test_single_period(self):
        """Test SMA with period=1."""
        prices = np.array([1.0, 2.0, 3.0])
        result = calculate_sma(prices, period=1)

        # Should equal the prices themselves
        np.testing.assert_array_almost_equal(result, prices)


class TestEMA:
    """Test calculate_ema function."""

    def test_basic_ema(self):
        """Test basic EMA calculation."""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = calculate_ema(prices, period=3)

        # First 2 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # EMA at index 2 should be SMA of first 3 values
        assert result[2] == pytest.approx(2.0)
        # Subsequent values should use EMA formula
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])

    def test_ema_trend_following(self):
        """Test that EMA follows trend."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        result = calculate_ema(prices, period=3)

        # EMA should be increasing for uptrend
        valid = result[~np.isnan(result)]
        assert all(valid[i] < valid[i + 1] for i in range(len(valid) - 1))

    def test_insufficient_data(self):
        """Test EMA with insufficient data."""
        prices = np.array([1.0, 2.0])
        result = calculate_ema(prices, period=5)

        assert len(result) == 2
        assert np.all(np.isnan(result))


class TestRSI:
    """Test calculate_rsi function."""

    def test_uptrend_rsi(self):
        """Test RSI in a strong uptrend."""
        prices = np.array([float(i) for i in range(1, 25)])  # 1 to 24
        result = calculate_rsi(prices, period=14)

        # RSI should be high (above 70) in strong uptrend
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] > 70

    def test_downtrend_rsi(self):
        """Test RSI in a strong downtrend."""
        prices = np.array([float(i) for i in range(100, 76, -1)])  # 100 down to 77
        result = calculate_rsi(prices, period=14)

        # RSI should be low (below 30) in strong downtrend
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] < 30

    def test_rsi_range(self):
        """Test that RSI stays between 0 and 100."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        result = calculate_rsi(prices, period=14)

        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = np.array([1.0, 2.0, 3.0])
        result = calculate_rsi(prices, period=14)

        assert np.all(np.isnan(result))


class TestMACD:
    """Test calculate_macd function."""

    def test_basic_macd(self):
        """Test basic MACD calculation."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        macd, signal, histogram = calculate_macd(prices)

        # Should have some valid values after slow period + signal period
        valid_idx = 26 + 9 - 1  # slow(26) + signal(9) - 1
        assert not np.isnan(macd[valid_idx])
        assert not np.isnan(signal[valid_idx])
        assert not np.isnan(histogram[valid_idx])

    def test_macd_histogram_is_difference(self):
        """Test that histogram = macd - signal."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        macd, signal, histogram = calculate_macd(prices)

        valid_mask = ~np.isnan(macd) & ~np.isnan(signal)
        np.testing.assert_array_almost_equal(
            histogram[valid_mask], macd[valid_mask] - signal[valid_mask]
        )

    def test_insufficient_data(self):
        """Test MACD with insufficient data."""
        prices = np.array([float(i) for i in range(20)])
        macd, signal, histogram = calculate_macd(prices)

        # All should be NaN
        assert np.all(np.isnan(macd))
        assert np.all(np.isnan(signal))
        assert np.all(np.isnan(histogram))


class TestBollingerPosition:
    """Test calculate_bollinger_position function."""

    def test_position_at_sma(self):
        """Test position when price equals SMA."""
        prices = np.array([50.0] * 25)  # Constant price
        result = calculate_bollinger_position(prices, period=20)

        # Price at SMA should give position = 0.5
        valid = result[~np.isnan(result)]
        assert all(v == pytest.approx(0.5) for v in valid)

    def test_position_near_upper_band(self):
        """Test position near upper Bollinger Band."""
        base = [50.0] * 19
        prices = np.array(base + [55.0, 56.0, 57.0, 58.0, 59.0, 60.0])
        result = calculate_bollinger_position(prices, period=20)

        # Last value should be above 0.5 (closer to upper band)
        assert result[-1] > 0.5

    def test_position_near_lower_band(self):
        """Test position near lower Bollinger Band."""
        base = [50.0] * 19
        prices = np.array(base + [45.0, 44.0, 43.0, 42.0, 41.0, 40.0])
        result = calculate_bollinger_position(prices, period=20)

        # Last value should be below 0.5 (closer to lower band)
        assert result[-1] < 0.5

    def test_insufficient_data(self):
        """Test Bollinger with insufficient data."""
        prices = np.array([50.0] * 10)
        result = calculate_bollinger_position(prices, period=20)

        assert np.all(np.isnan(result))


class TestATR:
    """Test calculate_atr function."""

    def test_basic_atr(self):
        """Test basic ATR calculation."""
        n = 30
        high = np.array([110.0] * n)
        low = np.array([90.0] * n)
        close = np.array([100.0] * n)
        result = calculate_atr(high, low, close, period=14)

        # ATR should be around 20 (high - low) for constant range
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] == pytest.approx(20.0, rel=0.1)

    def test_increasing_volatility(self):
        """Test ATR with increasing volatility."""
        n = 30
        # Increasing range over time
        high = np.array([100 + i * 2 for i in range(n)], dtype=float)
        low = np.array([100 - i * 2 for i in range(n)], dtype=float)
        close = np.array([100.0] * n)
        result = calculate_atr(high, low, close, period=14)

        # ATR should be increasing
        valid = result[~np.isnan(result)]
        assert valid[-1] > valid[0]

    def test_insufficient_data(self):
        """Test ATR with insufficient data."""
        high = np.array([110.0] * 10)
        low = np.array([90.0] * 10)
        close = np.array([100.0] * 10)
        result = calculate_atr(high, low, close, period=14)

        assert np.all(np.isnan(result))


class TestNormalizeRSISignal:
    """Test normalize_rsi_signal function."""

    def test_neutral_rsi(self):
        """Test RSI of 50 gives highest score."""
        assert normalize_rsi_signal(50.0) == pytest.approx(1.0)

    def test_overbought_rsi(self):
        """Test overbought RSI gives low score."""
        assert normalize_rsi_signal(100.0) == pytest.approx(0.0)

    def test_oversold_rsi(self):
        """Test oversold RSI gives low score."""
        assert normalize_rsi_signal(0.0) == pytest.approx(0.0)

    def test_intermediate_rsi(self):
        """Test intermediate RSI values."""
        assert normalize_rsi_signal(75.0) == pytest.approx(0.5)
        assert normalize_rsi_signal(25.0) == pytest.approx(0.5)

    def test_nan_rsi(self):
        """Test NaN RSI returns 0.5."""
        assert normalize_rsi_signal(np.nan) == pytest.approx(0.5)


class TestNormalizeMACDSignal:
    """Test normalize_macd_signal function."""

    def test_positive_histogram(self):
        """Test positive histogram returns 1.0."""
        assert normalize_macd_signal(0.5) == 1.0
        assert normalize_macd_signal(100.0) == 1.0

    def test_negative_histogram(self):
        """Test negative histogram returns 0.0."""
        assert normalize_macd_signal(-0.5) == 0.0
        assert normalize_macd_signal(-100.0) == 0.0

    def test_zero_histogram(self):
        """Test zero histogram returns 0.0."""
        assert normalize_macd_signal(0.0) == 0.0

    def test_nan_histogram(self):
        """Test NaN histogram returns 0.5."""
        assert normalize_macd_signal(np.nan) == pytest.approx(0.5)
