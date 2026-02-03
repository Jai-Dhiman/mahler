"""Options Greeks calculations using pure Python Black-Scholes implementation.

This module provides first-order Greeks and implied volatility calculations
using pure Python implementations compatible with Pyodide/Cloudflare Workers.

Second-order Greeks (vanna, volga, charm) remain in the custom greeks.py module.
"""

import math
from dataclasses import dataclass
from typing import Literal


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


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _calculate_d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
) -> tuple[float, float]:
    """Calculate d1 and d2 for Black-Scholes formula."""
    if time_to_expiry <= 0 or volatility <= 0:
        return 0.0, 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return d1, d2


def _black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: Literal["call", "put"],
) -> float:
    """Calculate Black-Scholes option price.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        volatility: Implied volatility as decimal
        risk_free_rate: Risk-free interest rate as decimal
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if time_to_expiry <= 0:
        # At expiration, return intrinsic value
        if option_type == "call":
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)

    d1, d2 = _calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)
    exp_rt = math.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * exp_rt * _norm_cdf(d2)
    else:
        return strike * exp_rt * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def calculate_greeks_vollib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: Literal["call", "put"] = "call",
) -> GreeksResult:
    """Calculate first-order Greeks using Black-Scholes formulas.

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

    d1, d2 = _calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)
    sqrt_t = math.sqrt(time_to_expiry)

    # Common calculations
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    npd1 = _norm_pdf(d1)
    exp_rt = math.exp(-risk_free_rate * time_to_expiry)

    # Delta
    if option_type == "call":
        delta_val = nd1
    else:
        delta_val = nd1 - 1

    # Gamma (same for calls and puts)
    gamma_val = npd1 / (spot * volatility * sqrt_t)

    # Theta (per day, negative for long options)
    theta_common = -(spot * npd1 * volatility) / (2 * sqrt_t)
    if option_type == "call":
        theta_val = (theta_common - risk_free_rate * strike * exp_rt * nd2) / 365
    else:
        theta_val = (theta_common + risk_free_rate * strike * exp_rt * (1 - nd2)) / 365

    # Vega (per 1% change in volatility)
    vega_val = spot * sqrt_t * npd1 / 100

    # Rho (per 1% change in interest rate)
    if option_type == "call":
        rho_val = strike * time_to_expiry * exp_rt * nd2 / 100
    else:
        rho_val = -strike * time_to_expiry * exp_rt * (1 - nd2) / 100

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
    """Calculate implied volatility from option price using Newton-Raphson method.

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
        InvalidOptionPriceError: If option price is below intrinsic value or above maximum
        IVCalculationError: If IV calculation fails to converge
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

    exp_rt = math.exp(-risk_free_rate * time_to_expiry)

    # Calculate intrinsic value
    if option_type == "call":
        intrinsic = max(0.0, spot - strike * exp_rt)
    else:
        intrinsic = max(0.0, strike * exp_rt - spot)

    # Check if price is below intrinsic value
    if option_price < intrinsic - 1e-10:  # Small tolerance for floating point
        raise InvalidOptionPriceError(
            f"Option price ${option_price:.2f} is below intrinsic value ${intrinsic:.2f}"
        )

    # Check if price exceeds theoretical maximum
    if option_type == "call":
        max_price = spot  # Call can't be worth more than the stock
    else:
        max_price = strike * exp_rt  # Put can't be worth more than PV of strike

    if option_price > max_price + 1e-10:
        raise InvalidOptionPriceError(
            f"Option price ${option_price:.2f} exceeds theoretical maximum ${max_price:.2f}"
        )

    # Newton-Raphson iteration
    sigma = 0.3  # Initial guess: 30% volatility
    max_iterations = 100
    tolerance = 1e-6

    for _ in range(max_iterations):
        # Calculate price and vega at current sigma
        price = _black_scholes_price(
            spot, strike, time_to_expiry, sigma, risk_free_rate, option_type
        )

        # Calculate vega (not scaled by 100 for IV solver)
        d1, _ = _calculate_d1_d2(spot, strike, time_to_expiry, sigma, risk_free_rate)
        sqrt_t = math.sqrt(time_to_expiry)
        vega = spot * sqrt_t * _norm_pdf(d1)

        # Check for convergence
        price_diff = price - option_price
        if abs(price_diff) < tolerance:
            return sigma

        # Avoid division by zero (vega near zero means we're at extreme moneyness)
        if abs(vega) < 1e-10:
            # Switch to bisection or adjust sigma
            if price_diff > 0:
                sigma = sigma * 0.5  # Price too high, reduce vol
            else:
                sigma = sigma * 1.5  # Price too low, increase vol
            continue

        # Newton-Raphson update
        sigma_new = sigma - price_diff / vega

        # Ensure sigma stays positive and reasonable
        sigma_new = max(0.001, min(sigma_new, 5.0))  # Bound between 0.1% and 500%

        # Check for convergence in sigma
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new

        sigma = sigma_new

    raise IVCalculationError(
        f"IV calculation did not converge after {max_iterations} iterations"
    )
