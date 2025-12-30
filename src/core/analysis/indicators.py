"""Technical indicators for options analysis.

All functions work with NumPy arrays for Pyodide compatibility.
Returns NaN for periods where calculation is not possible.
"""

import numpy as np


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average.

    Args:
        prices: Array of prices
        period: SMA period

    Returns:
        Array of SMA values, NaN for first (period-1) elements
    """
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    result = np.full(len(prices), np.nan)
    cumsum = np.cumsum(prices)
    result[period - 1 :] = (cumsum[period - 1 :] - np.concatenate([[0], cumsum[:-period]])) / period
    return result


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average.

    Uses the standard EMA formula with alpha = 2 / (period + 1).

    Args:
        prices: Array of prices
        period: EMA period

    Returns:
        Array of EMA values, NaN for first (period-1) elements
    """
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    alpha = 2.0 / (period + 1)
    result = np.full(len(prices), np.nan)

    # Initialize with SMA of first 'period' values
    result[period - 1] = np.mean(prices[:period])

    # Calculate EMA for remaining values
    for i in range(period, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

    return result


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index using Wilder's smoothing.

    Args:
        prices: Array of closing prices
        period: RSI period (default 14)

    Returns:
        Array of RSI values (0-100), NaN for first 'period' elements
    """
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)

    # Calculate price changes
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(prices), np.nan)

    # Initial average gain/loss using SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate remaining RSI values using Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def calculate_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD, signal line, and histogram.

    Args:
        prices: Array of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if len(prices) < slow + signal:
        nan_array = np.full(len(prices), np.nan)
        return nan_array, nan_array.copy(), nan_array.copy()

    # Calculate fast and slow EMAs
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    # MACD line is the difference
    macd_line = ema_fast - ema_slow

    # Signal line is EMA of MACD line
    # Only calculate where MACD is valid (after slow period)
    signal_line = np.full(len(prices), np.nan)

    # Find first valid MACD value
    first_valid = slow - 1
    if first_valid + signal <= len(prices):
        macd_for_signal = macd_line[first_valid:]
        signal_ema = calculate_ema(macd_for_signal, signal)
        signal_line[first_valid:] = signal_ema

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_position(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> np.ndarray:
    """Calculate position within Bollinger Bands (0-1 scale).

    Args:
        prices: Array of closing prices
        period: Bollinger Band period (default 20)
        num_std: Number of standard deviations (default 2)

    Returns:
        Array where 0 = at lower band, 0.5 = at SMA, 1 = at upper band
        Values can exceed [0, 1] if price is outside bands
    """
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    result = np.full(len(prices), np.nan)

    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=1)  # Use sample std

        if std == 0:
            result[i] = 0.5  # Price at SMA, no volatility
        else:
            band_width = 2 * num_std * std
            lower_band = sma - num_std * std
            result[i] = (prices[i] - lower_band) / band_width

    return result


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calculate Average True Range.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        period: ATR period (default 14)

    Returns:
        Array of ATR values
    """
    if len(high) < period + 1:
        return np.full(len(high), np.nan)

    # Calculate True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Calculate ATR using Wilder's smoothing (similar to RSI)
    result = np.full(len(high), np.nan)

    # Initial ATR is simple average
    result[period] = np.mean(true_range[1 : period + 1])

    # Wilder's smoothing for remaining values
    for i in range(period + 1, len(high)):
        result[i] = (result[i - 1] * (period - 1) + true_range[i]) / period

    return result


def normalize_rsi_signal(rsi: float) -> float:
    """Normalize RSI to a 0-1 signal where 0.5 is neutral.

    Values near 50 (neutral RSI) return higher scores.
    Extreme values (overbought/oversold) return lower scores.

    Args:
        rsi: RSI value (0-100)

    Returns:
        Normalized signal (0-1)
    """
    if np.isnan(rsi):
        return 0.5
    return 1.0 - abs(rsi - 50.0) / 50.0


def normalize_macd_signal(histogram: float) -> float:
    """Normalize MACD histogram to a 0-1 signal.

    Positive histogram -> bullish (higher score for bull puts)
    Negative histogram -> bearish (higher score for bear calls)

    Args:
        histogram: MACD histogram value

    Returns:
        1.0 if positive, 0.0 if negative, 0.5 if zero/nan
    """
    if np.isnan(histogram):
        return 0.5
    return 1.0 if histogram > 0 else 0.0
