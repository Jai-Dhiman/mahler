"""Tests for vollib Greeks adapter."""

import math

import pytest

from core.analysis.greeks import calculate_greeks
from core.analysis.greeks_vollib import (
    IVCalculationError,
    InvalidOptionPriceError,
    calculate_greeks_vollib,
    calculate_implied_volatility,
)


class TestCalculateGreeksVollib:
    """Tests for calculate_greeks_vollib function."""

    def test_call_delta_atm(self):
        """ATM call should have delta near 0.5."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        assert 0.45 < result.delta < 0.55

    def test_put_delta_atm(self):
        """ATM put should have delta near -0.5."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="put",
        )
        assert -0.55 < result.delta < -0.45

    def test_call_delta_deep_itm(self):
        """Deep ITM call should have delta near 1.0."""
        result = calculate_greeks_vollib(
            spot=120.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        assert result.delta > 0.95

    def test_put_delta_deep_otm(self):
        """Deep OTM put should have delta near 0."""
        result = calculate_greeks_vollib(
            spot=120.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="put",
        )
        assert result.delta > -0.10

    def test_gamma_atm_highest(self):
        """Gamma should be highest for ATM options."""
        atm = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        itm = calculate_greeks_vollib(
            spot=110.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        otm = calculate_greeks_vollib(
            spot=90.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        assert atm.gamma > itm.gamma
        assert atm.gamma > otm.gamma

    def test_theta_negative_for_long(self):
        """Theta should be negative for long options (time decay)."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        assert result.theta < 0

    def test_vega_positive(self):
        """Vega should be positive (options gain value with higher vol)."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.20,
            option_type="call",
        )
        assert result.vega > 0

    def test_expiration_call_itm(self):
        """At expiration, ITM call should have delta = 1."""
        result = calculate_greeks_vollib(
            spot=105.0,
            strike=100.0,
            time_to_expiry=0,
            volatility=0.20,
            option_type="call",
        )
        assert result.delta == 1.0
        assert result.gamma == 0.0
        assert result.theta == 0.0
        assert result.vega == 0.0

    def test_expiration_put_itm(self):
        """At expiration, ITM put should have delta = -1."""
        result = calculate_greeks_vollib(
            spot=95.0,
            strike=100.0,
            time_to_expiry=0,
            volatility=0.20,
            option_type="put",
        )
        assert result.delta == -1.0

    def test_expiration_call_otm(self):
        """At expiration, OTM call should have delta = 0."""
        result = calculate_greeks_vollib(
            spot=95.0,
            strike=100.0,
            time_to_expiry=0,
            volatility=0.20,
            option_type="call",
        )
        assert result.delta == 0.0

    def test_invalid_volatility_raises(self):
        """Zero or negative volatility should raise ValueError."""
        with pytest.raises(ValueError, match="Volatility must be positive"):
            calculate_greeks_vollib(
                spot=100.0,
                strike=100.0,
                time_to_expiry=30 / 365,
                volatility=0.0,
                option_type="call",
            )

        with pytest.raises(ValueError, match="Volatility must be positive"):
            calculate_greeks_vollib(
                spot=100.0,
                strike=100.0,
                time_to_expiry=30 / 365,
                volatility=-0.10,
                option_type="call",
            )

    def test_invalid_spot_raises(self):
        """Non-positive spot price should raise ValueError."""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            calculate_greeks_vollib(
                spot=0.0,
                strike=100.0,
                time_to_expiry=30 / 365,
                volatility=0.20,
                option_type="call",
            )

    def test_invalid_strike_raises(self):
        """Non-positive strike price should raise ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            calculate_greeks_vollib(
                spot=100.0,
                strike=0.0,
                time_to_expiry=30 / 365,
                volatility=0.20,
                option_type="call",
            )


class TestVolLibVsCustomComparison:
    """Compare vollib results with custom implementation.

    vollib uses the same Black-Scholes formulas and conventions as our
    custom implementation, so results should match exactly (within float precision).
    """

    def test_delta_matches_custom(self):
        """vollib delta should match custom exactly."""
        params = {
            "spot": 450.0,
            "strike": 445.0,
            "time_to_expiry": 30 / 365,
            "volatility": 0.20,
            "risk_free_rate": 0.05,
            "option_type": "put",
        }

        vollib_result = calculate_greeks_vollib(**params)
        custom_result = calculate_greeks(**params)

        # Delta should match within float precision
        assert abs(vollib_result.delta - custom_result.delta) < 1e-10

    def test_gamma_matches_custom(self):
        """vollib gamma should match custom exactly."""
        params = {
            "spot": 450.0,
            "strike": 445.0,
            "time_to_expiry": 30 / 365,
            "volatility": 0.20,
            "risk_free_rate": 0.05,
            "option_type": "put",
        }

        vollib_result = calculate_greeks_vollib(**params)
        custom_result = calculate_greeks(**params)

        # Gamma should match within float precision
        assert abs(vollib_result.gamma - custom_result.gamma) < 1e-10

    def test_theta_matches_custom(self):
        """vollib theta should match custom exactly."""
        params = {
            "spot": 450.0,
            "strike": 445.0,
            "time_to_expiry": 30 / 365,
            "volatility": 0.20,
            "risk_free_rate": 0.05,
            "option_type": "call",
        }

        vollib_result = calculate_greeks_vollib(**params)
        custom_result = calculate_greeks(**params)

        # Theta should match within float precision
        assert abs(vollib_result.theta - custom_result.theta) < 1e-10

    def test_vega_matches_custom(self):
        """vollib vega should match custom exactly."""
        params = {
            "spot": 450.0,
            "strike": 445.0,
            "time_to_expiry": 30 / 365,
            "volatility": 0.20,
            "risk_free_rate": 0.05,
            "option_type": "call",
        }

        vollib_result = calculate_greeks_vollib(**params)
        custom_result = calculate_greeks(**params)

        # Vega should match within float precision
        assert abs(vollib_result.vega - custom_result.vega) < 1e-10

    def test_rho_matches_custom(self):
        """vollib rho should match custom exactly."""
        params = {
            "spot": 450.0,
            "strike": 445.0,
            "time_to_expiry": 30 / 365,
            "volatility": 0.20,
            "risk_free_rate": 0.05,
            "option_type": "call",
        }

        vollib_result = calculate_greeks_vollib(**params)
        custom_result = calculate_greeks(**params)

        # Rho should match within float precision
        assert abs(vollib_result.rho - custom_result.rho) < 1e-10


class TestCalculateImpliedVolatility:
    """Tests for calculate_implied_volatility function."""

    def test_iv_round_trip(self):
        """Price -> IV -> Price should return original price."""
        from py_vollib.black_scholes import black_scholes

        spot = 100.0
        strike = 100.0
        time_to_expiry = 30 / 365
        risk_free_rate = 0.05
        original_vol = 0.25

        # Calculate price from known vol
        option_price = black_scholes("c", spot, strike, time_to_expiry, risk_free_rate, original_vol)

        # Calculate IV from price
        calculated_iv = calculate_implied_volatility(
            option_price=option_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type="call",
        )

        # Should match within 0.1%
        assert abs(calculated_iv - original_vol) < 0.001

    def test_iv_put_option(self):
        """IV calculation should work for puts too."""
        from py_vollib.black_scholes import black_scholes

        spot = 450.0
        strike = 440.0
        time_to_expiry = 45 / 365
        risk_free_rate = 0.05
        original_vol = 0.22

        option_price = black_scholes("p", spot, strike, time_to_expiry, risk_free_rate, original_vol)

        calculated_iv = calculate_implied_volatility(
            option_price=option_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type="put",
        )

        assert abs(calculated_iv - original_vol) < 0.001

    def test_iv_at_expiration_raises(self):
        """IV calculation at expiration should raise IVCalculationError."""
        with pytest.raises(IVCalculationError, match="Cannot calculate IV at expiration"):
            calculate_implied_volatility(
                option_price=5.0,
                spot=100.0,
                strike=95.0,
                time_to_expiry=0,
                risk_free_rate=0.05,
                option_type="call",
            )

    def test_iv_negative_price_raises(self):
        """Negative option price should raise InvalidOptionPriceError."""
        with pytest.raises(InvalidOptionPriceError, match="Option price must be positive"):
            calculate_implied_volatility(
                option_price=-1.0,
                spot=100.0,
                strike=100.0,
                time_to_expiry=30 / 365,
                risk_free_rate=0.05,
                option_type="call",
            )

    def test_iv_zero_price_raises(self):
        """Zero option price should raise InvalidOptionPriceError."""
        with pytest.raises(InvalidOptionPriceError, match="Option price must be positive"):
            calculate_implied_volatility(
                option_price=0.0,
                spot=100.0,
                strike=100.0,
                time_to_expiry=30 / 365,
                risk_free_rate=0.05,
                option_type="call",
            )

    def test_iv_below_intrinsic_raises(self):
        """Price below intrinsic value should raise InvalidOptionPriceError."""
        # For deep ITM call with spot=120, strike=100, intrinsic is ~20
        # Price of 5 is below intrinsic
        with pytest.raises(InvalidOptionPriceError, match="below intrinsic"):
            calculate_implied_volatility(
                option_price=5.0,
                spot=120.0,
                strike=100.0,
                time_to_expiry=30 / 365,
                risk_free_rate=0.05,
                option_type="call",
            )

    def test_iv_realistic_spy_option(self):
        """Test IV calculation for realistic SPY option."""
        # SPY at 450, 30 DTE put spread scenario
        # Typical 0.20 delta put might have IV around 15-20%
        from py_vollib.black_scholes import black_scholes

        spot = 450.0
        strike = 430.0  # ~0.20 delta put
        time_to_expiry = 30 / 365
        risk_free_rate = 0.05
        expected_iv = 0.18

        option_price = black_scholes("p", spot, strike, time_to_expiry, risk_free_rate, expected_iv)

        calculated_iv = calculate_implied_volatility(
            option_price=option_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type="put",
        )

        assert abs(calculated_iv - expected_iv) < 0.001


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_dte(self):
        """Greeks should be valid for very short DTE."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1 / 365,  # 1 day
            volatility=0.30,
            option_type="call",
        )
        assert not math.isnan(result.delta)
        assert not math.isnan(result.gamma)
        assert not math.isnan(result.theta)
        assert not math.isnan(result.vega)

    def test_high_volatility(self):
        """Greeks should handle high volatility."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=1.0,  # 100% vol
            option_type="call",
        )
        assert not math.isnan(result.delta)
        assert result.vega > 0

    def test_low_volatility(self):
        """Greeks should handle low volatility."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.05,  # 5% vol
            option_type="call",
        )
        assert not math.isnan(result.delta)
        assert result.gamma > 0

    def test_long_dated_option(self):
        """Greeks should be valid for long-dated options."""
        result = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=365 / 365,  # 1 year
            volatility=0.25,
            option_type="call",
        )
        assert not math.isnan(result.delta)
        # Long dated options have higher vega
        short_dated = calculate_greeks_vollib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=30 / 365,
            volatility=0.25,
            option_type="call",
        )
        assert result.vega > short_dated.vega
