"""TradingGroup Dynamic Exit Management.

Implements the TradingGroup paper's (arXiv:2508.17565) dynamic exit formula:

    T_SL = m_s^sl * sigma_d,10
    T_TP = m_s^tp * sigma_d,10

Where:
- sigma_d,10 = 10-day standard deviation of daily log-returns (unannualized)
- m_s^sl, m_s^tp = style-specific multipliers

Style Multipliers:
| Style        | SL Mult | TP Mult |
|--------------|---------|---------|
| aggressive   | 2.5     | 1.5     |
| neutral      | 2.0     | 1.0     |
| conservative | 1.5     | 0.75    |

The trading style is determined dynamically based on:
- Current VIX level
- Recent P&L performance
- Market regime
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TradingStyle(str, Enum):
    """Trading style for dynamic exit calculations.

    Each style has different stop-loss and take-profit multipliers
    that adjust based on market conditions and agent risk appetite.
    """

    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    CONSERVATIVE = "conservative"


@dataclass
class StyleMultipliers:
    """Multipliers for a trading style."""

    stop_loss: float
    take_profit: float


# TradingGroup paper multipliers
STYLE_MULTIPLIERS: dict[TradingStyle, StyleMultipliers] = {
    TradingStyle.AGGRESSIVE: StyleMultipliers(stop_loss=2.5, take_profit=1.5),
    TradingStyle.NEUTRAL: StyleMultipliers(stop_loss=2.0, take_profit=1.0),
    TradingStyle.CONSERVATIVE: StyleMultipliers(stop_loss=1.5, take_profit=0.75),
}


@dataclass
class DynamicThresholds:
    """Dynamic exit thresholds calculated from volatility and style."""

    stop_loss_threshold: float  # T_SL as percentage
    take_profit_threshold: float  # T_TP as percentage
    sigma_d_10: float  # 10-day log-return volatility
    trading_style: TradingStyle
    style_multipliers: StyleMultipliers

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "stop_loss_threshold": self.stop_loss_threshold,
            "take_profit_threshold": self.take_profit_threshold,
            "sigma_d_10": self.sigma_d_10,
            "trading_style": self.trading_style.value,
            "style_multipliers": {
                "stop_loss": self.style_multipliers.stop_loss,
                "take_profit": self.style_multipliers.take_profit,
            },
        }


def calculate_sigma_d_10(bars: list[dict]) -> float:
    """Calculate 10-day standard deviation of daily log-returns.

    This is the sigma_d,10 term from the TradingGroup paper.
    It measures recent realized volatility without annualization.

    Args:
        bars: List of OHLCV bars, most recent last. Each bar should have
              a 'close' key with the closing price.

    Returns:
        10-day standard deviation of daily log-returns (unannualized).
        Returns 0.0 if insufficient data (< 11 bars needed for 10 returns).
    """
    if not bars or len(bars) < 11:
        # Need at least 11 bars to get 10 log-returns
        return 0.0

    # Use the most recent 11 bars to get 10 log-returns
    recent_bars = bars[-11:]

    # Calculate log-returns
    log_returns = []
    for i in range(1, len(recent_bars)):
        prev_close = recent_bars[i - 1].get("close", 0)
        curr_close = recent_bars[i].get("close", 0)

        if prev_close <= 0 or curr_close <= 0:
            continue

        log_return = math.log(curr_close / prev_close)
        log_returns.append(log_return)

    if len(log_returns) < 10:
        return 0.0

    # Calculate standard deviation
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    sigma = math.sqrt(variance)

    return sigma


def get_dynamic_thresholds(
    bars: list[dict],
    style: TradingStyle | str,
) -> DynamicThresholds:
    """Calculate dynamic exit thresholds using the TradingGroup formula.

    Formula:
        T_SL = m_s^sl * sigma_d,10
        T_TP = m_s^tp * sigma_d,10

    Args:
        bars: List of OHLCV bars for volatility calculation
        style: Trading style (aggressive, neutral, conservative)

    Returns:
        DynamicThresholds with stop-loss and take-profit levels
    """
    # Convert string to enum if needed
    if isinstance(style, str):
        style = TradingStyle(style.lower())

    # Calculate realized volatility
    sigma_d_10 = calculate_sigma_d_10(bars)

    # Get style multipliers
    multipliers = STYLE_MULTIPLIERS[style]

    # Calculate thresholds
    # These are percentages: e.g., 0.02 = 2% move threshold
    stop_loss_threshold = multipliers.stop_loss * sigma_d_10
    take_profit_threshold = multipliers.take_profit * sigma_d_10

    # Apply minimum thresholds to avoid extremely tight exits in low-vol
    # Minimum SL: 1%, Minimum TP: 0.5%
    stop_loss_threshold = max(stop_loss_threshold, 0.01)
    take_profit_threshold = max(take_profit_threshold, 0.005)

    return DynamicThresholds(
        stop_loss_threshold=stop_loss_threshold,
        take_profit_threshold=take_profit_threshold,
        sigma_d_10=sigma_d_10,
        trading_style=style,
        style_multipliers=multipliers,
    )


def determine_trading_style(
    vix: float | None = None,
    recent_pnl_percent: float | None = None,
    market_regime: str | None = None,
) -> TradingStyle:
    """Determine trading style based on market conditions and performance.

    The style selection follows a hierarchical approach:
    1. VIX level is the primary factor (market stress indicator)
    2. Recent P&L adjusts within VIX-based tier
    3. Market regime provides additional context

    Decision Logic:
    - VIX > 30: Conservative (high stress environment)
    - VIX > 25: Neutral to Conservative (elevated caution)
    - VIX > 20: Neutral (typical conditions)
    - VIX <= 20: Neutral to Aggressive (favorable conditions)

    P&L Adjustment:
    - Recent losses (< -5%): Shift one tier more conservative
    - Recent gains (> 5%): Shift one tier more aggressive

    Regime Adjustment:
    - Crisis/Bearish regime: Always conservative regardless of VIX
    - Bullish regime: Allow aggressive if VIX supports it

    Args:
        vix: Current VIX level (None defaults to neutral)
        recent_pnl_percent: Recent P&L as percentage (e.g., -0.05 = -5%)
        market_regime: Market regime string (e.g., "bullish", "bearish", "neutral", "crisis")

    Returns:
        TradingStyle enum value
    """
    # Start with neutral as baseline
    style_score = 1  # 0=conservative, 1=neutral, 2=aggressive

    # VIX-based baseline
    if vix is not None:
        if vix > 30:
            style_score = 0  # Conservative in high VIX
        elif vix > 25:
            style_score = 0  # Conservative
        elif vix > 20:
            style_score = 1  # Neutral
        else:
            style_score = 2  # Aggressive in low VIX

    # P&L adjustment
    if recent_pnl_percent is not None:
        if recent_pnl_percent < -0.05:  # Recent losses > 5%
            style_score = max(0, style_score - 1)  # Shift conservative
        elif recent_pnl_percent > 0.05:  # Recent gains > 5%
            style_score = min(2, style_score + 1)  # Shift aggressive

    # Regime override
    if market_regime is not None:
        regime_lower = market_regime.lower()
        if regime_lower in ("crisis", "bearish"):
            style_score = 0  # Always conservative in crisis/bearish
        elif regime_lower == "highly_bullish" and style_score >= 1:
            style_score = 2  # Allow aggressive in highly bullish

    # Map score to style
    style_map = {0: TradingStyle.CONSERVATIVE, 1: TradingStyle.NEUTRAL, 2: TradingStyle.AGGRESSIVE}
    return style_map[style_score]


def calculate_dynamic_exit_prices(
    entry_credit: float,
    bars: list[dict],
    style: TradingStyle | str,
) -> tuple[float, float, DynamicThresholds]:
    """Calculate dynamic exit prices for a credit spread position.

    For credit spreads:
    - Take profit: Close when spread value drops to entry_credit * (1 - TP threshold)
    - Stop loss: Close when spread value rises to entry_credit * (1 + SL threshold)

    Args:
        entry_credit: Entry credit per spread
        bars: Historical bars for volatility calculation
        style: Trading style

    Returns:
        Tuple of (take_profit_price, stop_loss_price, thresholds)
        - take_profit_price: Close position when spread value falls to this
        - stop_loss_price: Close position when spread value rises to this
    """
    thresholds = get_dynamic_thresholds(bars, style)

    # For credit spreads:
    # Profit when spread value decreases (we keep more credit)
    take_profit_price = entry_credit * (1 - thresholds.take_profit_threshold)

    # Loss when spread value increases (costs more to close)
    stop_loss_price = entry_credit * (1 + thresholds.stop_loss_threshold)

    return take_profit_price, stop_loss_price, thresholds
