"""Options Greeks calculations using vollib library.

This module provides an adapter for py_vollib to calculate first-order Greeks
and implied volatility. vollib uses Jaeckel's "Let's Be Rational" algorithm
for accurate and fast IV calculation.

Second-order Greeks (vanna, volga, charm) are not provided by vollib and
remain in the custom greeks.py module.
"""

from dataclasses import dataclass
from typing import Literal

from py_lets_be_rational.exceptions import (
    AboveMaximumException,
    BelowIntrinsicException,
)
from py_vollib.black_scholes.greeks.analytical import delta, gamma, rho, theta, vega
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic


class IVCalculationError(Exception):
    """Raised when implied volatility cannot be calculated."""

    pass


class InvalidOptionPriceError(Exception):
    """Raised when option price is invalid for IV calculation."""

    pass


@dataclass
class GreeksResult:
    """Calculated Greeks for an option."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _get_vollib_flag(option_type: Literal["call", "put"]) -> str:
    """Convert option type to vollib flag.

    Args:
        option_type: "call" or "put"

    Returns:
        "c" for call, "p" for put
    """
    return "c" if option_type == "call" else "p"


def calculate_greeks_vollib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: Literal["call", "put"] = "call",
) -> GreeksResult:
    """Calculate first-order Greeks using vollib.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years (e.g., 30 days = 30/365)
        volatility: Implied volatility as decimal (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free interest rate as decimal
        option_type: "call" or "put"

    Returns:
        GreeksResult with delta, gamma, theta, vega, rho

    Raises:
        ValueError: If inputs are invalid (negative values, etc.)
    """
    # Edge case: at expiration
    if time_to_expiry <= 0:
        if option_type == "call":
            delta_val = 1.0 if spot > strike else 0.0
        else:
            delta_val = -1.0 if spot < strike else 0.0
        return GreeksResult(delta=delta_val, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    # Edge case: zero or negative volatility
    if volatility <= 0:
        raise ValueError(f"Volatility must be positive, got {volatility}")

    # Edge case: invalid prices
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"Strike price must be positive, got {strike}")

    flag = _get_vollib_flag(option_type)

    # Calculate Greeks using vollib
    # vollib uses convention: delta(flag, S, K, t, r, sigma)
    delta_val = delta(flag, spot, strike, time_to_expiry, risk_free_rate, volatility)
    gamma_val = gamma(flag, spot, strike, time_to_expiry, risk_free_rate, volatility)

    # vollib theta is per-day (same as our custom implementation)
    theta_val = theta(flag, spot, strike, time_to_expiry, risk_free_rate, volatility)

    # vollib vega is per 1% vol change (same as our custom implementation)
    vega_val = vega(flag, spot, strike, time_to_expiry, risk_free_rate, volatility)

    # vollib rho is per 1% rate change (same as our custom implementation)
    rho_val = rho(flag, spot, strike, time_to_expiry, risk_free_rate, volatility)

    return GreeksResult(
        delta=delta_val,
        gamma=gamma_val,
        theta=theta_val,
        vega=vega_val,
        rho=rho_val,
    )


def calculate_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: Literal["call", "put"],
) -> float:
    """Calculate implied volatility from option price using Jaeckel's algorithm.

    Uses the "Let's Be Rational" algorithm which is accurate and fast,
    handling edge cases like deep ITM/OTM options.

    Args:
        option_price: Market price of the option
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate as decimal
        option_type: "call" or "put"

    Returns:
        Implied volatility as decimal (e.g., 0.20 for 20%)

    Raises:
        InvalidOptionPriceError: If option price is below intrinsic value
        IVCalculationError: If IV calculation fails for other reasons
    """
    # Edge case: at expiration
    if time_to_expiry <= 0:
        raise IVCalculationError("Cannot calculate IV at expiration (t=0)")

    # Edge case: invalid prices
    if option_price <= 0:
        raise InvalidOptionPriceError(f"Option price must be positive, got {option_price}")
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"Strike price must be positive, got {strike}")

    flag = _get_vollib_flag(option_type)

    try:
        iv = implied_volatility(
            option_price, spot, strike, time_to_expiry, risk_free_rate, flag
        )
        return iv
    except (PriceIsBelowIntrinsic, BelowIntrinsicException) as e:
        raise InvalidOptionPriceError(
            f"Option price ${option_price:.2f} is below intrinsic value"
        ) from e
    except (PriceIsAboveMaximum, AboveMaximumException) as e:
        # Deep ITM options where price exceeds theoretical maximum
        raise InvalidOptionPriceError(
            f"Option price ${option_price:.2f} exceeds theoretical maximum (deep ITM)"
        ) from e
    except Exception as e:
        raise IVCalculationError(f"Failed to calculate IV: {e}") from e
