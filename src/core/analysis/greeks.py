"""Options Greeks calculations using Black-Scholes model."""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class GreeksResult:
    """Calculated Greeks for an option."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class SecondOrderGreeks:
    """Second-order Greeks for risk management.

    These measure how first-order Greeks change with respect to other variables:
    - Vanna: dDelta/dVol (or equivalently dVega/dSpot) - spot-vol correlation sensitivity
    - Volga: dVega/dVol (also called Vomma) - vol-of-vol exposure
    - Charm: dDelta/dTime - delta decay rate (how delta changes as time passes)
    """

    vanna: float
    volga: float
    charm: float


@dataclass
class ExtendedGreeksResult:
    """Complete Greeks including first and second-order."""

    # First-order
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Second-order
    vanna: float
    volga: float
    charm: float


@dataclass
class SpreadSecondOrderGreeks:
    """Second-order Greeks for a credit spread (net position)."""

    vanna: float
    volga: float
    charm: float


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def calculate_d1_d2(
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
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t

    return d1, d2


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: Literal["call", "put"] = "call",
) -> GreeksResult:
    """Calculate all Greeks for an option.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years (e.g., 30 days = 30/365)
        volatility: Implied volatility as decimal (e.g., 0.20 for 20%)
        risk_free_rate: Risk-free interest rate as decimal
        option_type: "call" or "put"

    Returns:
        GreeksResult with delta, gamma, theta, vega, rho
    """
    if time_to_expiry <= 0:
        # At expiration
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return GreeksResult(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)
    sqrt_t = math.sqrt(time_to_expiry)

    # Common calculations
    nd1 = norm_cdf(d1)
    nd2 = norm_cdf(d2)
    npd1 = norm_pdf(d1)
    exp_rt = math.exp(-risk_free_rate * time_to_expiry)

    # Delta
    if option_type == "call":
        delta = nd1
    else:
        delta = nd1 - 1

    # Gamma (same for calls and puts)
    gamma = npd1 / (spot * volatility * sqrt_t)

    # Theta (per day, negative for long options)
    theta_common = -(spot * npd1 * volatility) / (2 * sqrt_t)
    if option_type == "call":
        theta = (theta_common - risk_free_rate * strike * exp_rt * nd2) / 365  # Per day
    else:
        theta = (theta_common + risk_free_rate * strike * exp_rt * (1 - nd2)) / 365  # Per day

    # Vega (per 1% change in volatility)
    vega = spot * sqrt_t * npd1 / 100

    # Rho (per 1% change in interest rate)
    if option_type == "call":
        rho = strike * time_to_expiry * exp_rt * nd2 / 100
    else:
        rho = -strike * time_to_expiry * exp_rt * (1 - nd2) / 100

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
    )


def calculate_spread_greeks(
    short_greeks: GreeksResult,
    long_greeks: GreeksResult,
    contracts: int = 1,
) -> GreeksResult:
    """Calculate net Greeks for a credit spread (short - long).

    For credit spreads, we're short the higher-premium option and
    long the lower-premium option as protection.
    """
    multiplier = contracts * 100  # Options multiplier

    return GreeksResult(
        delta=(short_greeks.delta - long_greeks.delta) * multiplier * -1,  # Short position
        gamma=(short_greeks.gamma - long_greeks.gamma) * multiplier * -1,
        theta=(short_greeks.theta - long_greeks.theta)
        * multiplier
        * -1,  # Positive for credit spreads
        vega=(short_greeks.vega - long_greeks.vega) * multiplier * -1,
        rho=(short_greeks.rho - long_greeks.rho) * multiplier * -1,
    )


def calculate_second_order_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: Literal["call", "put"] = "call",
) -> SecondOrderGreeks:
    """Calculate second-order Greeks for an option using Black-Scholes.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        volatility: Implied volatility as decimal
        risk_free_rate: Risk-free interest rate as decimal
        option_type: "call" or "put"

    Returns:
        SecondOrderGreeks with vanna, volga, charm

    Formulas (for calls, puts have same vanna/volga, different charm):
        Vanna = -e^(-qT) * N'(d1) * d2 / sigma
        Volga = S * e^(-qT) * sqrt(T) * N'(d1) * d1 * d2 / sigma
        Charm = -e^(-qT) * N'(d1) * [2(r-q)T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T))
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return SecondOrderGreeks(vanna=0.0, volga=0.0, charm=0.0)

    d1, d2 = calculate_d1_d2(spot, strike, time_to_expiry, volatility, risk_free_rate)
    sqrt_t = math.sqrt(time_to_expiry)
    npd1 = norm_pdf(d1)

    # Vanna: dDelta/dVol = -N'(d1) * d2 / sigma
    # Measures sensitivity of delta to changes in volatility
    vanna = -npd1 * d2 / volatility

    # Volga (Vomma): dVega/dVol = Vega * d1 * d2 / sigma
    # Where Vega = S * sqrt(T) * N'(d1)
    # Measures sensitivity of vega to changes in volatility
    vega = spot * sqrt_t * npd1
    volga = vega * d1 * d2 / volatility

    # Charm: dDelta/dTime (delta decay)
    # Measures how delta changes as time passes
    # For calls: -N'(d1) * [2*r*T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T)) - q*N(d1)
    # Simplified for q=0 (no dividend):
    charm_numerator = 2 * risk_free_rate * time_to_expiry - d2 * volatility * sqrt_t
    charm_denominator = 2 * time_to_expiry * volatility * sqrt_t
    charm = -npd1 * charm_numerator / charm_denominator

    # For puts, charm has opposite sign for the N(d1) term, but same for the N'(d1) term
    # The main component is the same for both
    if option_type == "put":
        # Adjustment for put charm is typically negligible for our use case
        pass

    # Normalize charm to per-day (similar to theta)
    charm = charm / 365

    return SecondOrderGreeks(
        vanna=vanna,
        volga=volga,
        charm=charm,
    )


def calculate_extended_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: Literal["call", "put"] = "call",
) -> ExtendedGreeksResult:
    """Calculate all first and second-order Greeks for an option.

    This combines calculate_greeks() and calculate_second_order_greeks()
    into a single result for convenience.
    """
    first_order = calculate_greeks(
        spot, strike, time_to_expiry, volatility, risk_free_rate, option_type
    )
    second_order = calculate_second_order_greeks(
        spot, strike, time_to_expiry, volatility, risk_free_rate, option_type
    )

    return ExtendedGreeksResult(
        delta=first_order.delta,
        gamma=first_order.gamma,
        theta=first_order.theta,
        vega=first_order.vega,
        rho=first_order.rho,
        vanna=second_order.vanna,
        volga=second_order.volga,
        charm=second_order.charm,
    )


def calculate_spread_second_order_greeks(
    short_greeks: SecondOrderGreeks,
    long_greeks: SecondOrderGreeks,
    contracts: int = 1,
) -> SpreadSecondOrderGreeks:
    """Calculate net second-order Greeks for a credit spread.

    For credit spreads, we're short the higher-premium option and
    long the lower-premium option as protection.
    """
    multiplier = contracts * 100  # Options multiplier

    return SpreadSecondOrderGreeks(
        vanna=(short_greeks.vanna - long_greeks.vanna) * multiplier * -1,
        volga=(short_greeks.volga - long_greeks.volga) * multiplier * -1,
        charm=(short_greeks.charm - long_greeks.charm) * multiplier * -1,
    )


def days_to_expiry(expiration: str) -> int:
    """Calculate days until expiration."""
    exp_date = datetime.strptime(expiration, "%Y-%m-%d")
    return max(0, (exp_date - datetime.now()).days)


def years_to_expiry(expiration: str) -> float:
    """Calculate years until expiration."""
    return days_to_expiry(expiration) / 365.0
