"""Tests for core types and enum parsing.

These tests ensure all enum types properly handle edge cases
and prevent similar issues to the OrderSide bug.
"""

from __future__ import annotations

import pytest

from core.types import (
    AssetClass,
    Confidence,
    CreditSpread,
    RecommendationStatus,
    SpreadType,
    TradeStatus,
)


class TestSpreadTypeEnum:
    """Test SpreadType enum parsing."""

    def test_valid_bull_put(self):
        """Test valid bull_put value."""
        assert SpreadType("bull_put") == SpreadType.BULL_PUT

    def test_valid_bear_call(self):
        """Test valid bear_call value."""
        assert SpreadType("bear_call") == SpreadType.BEAR_CALL

    def test_empty_string_raises_error(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SpreadType("")
        assert "is not a valid SpreadType" in str(exc_info.value)

    def test_invalid_value_raises_error(self):
        """Test invalid values raise errors."""
        with pytest.raises(ValueError):
            SpreadType("iron_condor")
        with pytest.raises(ValueError):
            SpreadType("BULL_PUT")  # Case sensitive


class TestRecommendationStatusEnum:
    """Test RecommendationStatus enum parsing."""

    def test_all_valid_statuses(self):
        """Test all valid status values."""
        valid_statuses = [
            ("pending", RecommendationStatus.PENDING),
            ("approved", RecommendationStatus.APPROVED),
            ("rejected", RecommendationStatus.REJECTED),
            ("expired", RecommendationStatus.EXPIRED),
            ("executed", RecommendationStatus.EXECUTED),
        ]
        for value, expected in valid_statuses:
            assert RecommendationStatus(value) == expected

    def test_empty_status_raises_error(self):
        """Test empty status raises error."""
        with pytest.raises(ValueError):
            RecommendationStatus("")

    def test_invalid_status_raises_error(self):
        """Test invalid status raises error."""
        with pytest.raises(ValueError):
            RecommendationStatus("cancelled")  # Not a valid status


class TestTradeStatusEnum:
    """Test TradeStatus enum parsing."""

    def test_valid_statuses(self):
        """Test valid trade statuses."""
        assert TradeStatus("open") == TradeStatus.OPEN
        assert TradeStatus("closed") == TradeStatus.CLOSED

    def test_empty_status_raises_error(self):
        """Test empty status raises error."""
        with pytest.raises(ValueError):
            TradeStatus("")


class TestConfidenceEnum:
    """Test Confidence enum parsing."""

    def test_valid_confidence_levels(self):
        """Test all valid confidence levels."""
        assert Confidence("low") == Confidence.LOW
        assert Confidence("medium") == Confidence.MEDIUM
        assert Confidence("high") == Confidence.HIGH

    def test_empty_confidence_raises_error(self):
        """Test empty confidence raises error."""
        with pytest.raises(ValueError):
            Confidence("")

    def test_invalid_confidence_raises_error(self):
        """Test invalid confidence raises error."""
        with pytest.raises(ValueError):
            Confidence("very_high")


class TestAssetClassEnum:
    """Test AssetClass enum parsing."""

    def test_valid_asset_classes(self):
        """Test all valid asset classes."""
        assert AssetClass("equity") == AssetClass.EQUITY
        assert AssetClass("treasury") == AssetClass.TREASURY
        assert AssetClass("commodity") == AssetClass.COMMODITY

    def test_empty_asset_class_raises_error(self):
        """Test empty asset class raises error."""
        with pytest.raises(ValueError):
            AssetClass("")


class TestCreditSpreadProperties:
    """Test CreditSpread computed properties."""

    @pytest.fixture
    def mock_option_contract(self):
        """Create a mock option contract."""
        from core.types import OptionContract

        return OptionContract(
            symbol="SPY240215P00470000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=470.0,
            option_type="put",
            bid=1.20,
            ask=1.30,
            last=1.25,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.22,
        )

    def test_width_calculation(self, mock_option_contract):
        """Test spread width calculation."""
        from core.types import OptionContract

        short_contract = mock_option_contract
        long_contract = OptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.70,
            ask=0.80,
            last=0.75,
            volume=800,
            open_interest=4000,
            implied_volatility=0.20,
        )

        spread = CreditSpread(
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            short_contract=short_contract,
            long_contract=long_contract,
        )

        assert spread.width == 5.0

    def test_credit_calculation(self, mock_option_contract):
        """Test credit calculation uses midpoints."""
        from core.types import OptionContract

        # Short contract: mid = (1.20 + 1.30) / 2 = 1.25
        short_contract = mock_option_contract
        # Long contract: mid = (0.70 + 0.80) / 2 = 0.75
        long_contract = OptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.70,
            ask=0.80,
            last=0.75,
            volume=800,
            open_interest=4000,
            implied_volatility=0.20,
        )

        spread = CreditSpread(
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            short_contract=short_contract,
            long_contract=long_contract,
        )

        # Credit = short_mid - long_mid = 1.25 - 0.75 = 0.50
        assert spread.credit == 0.50

    def test_max_loss_calculation(self, mock_option_contract):
        """Test max loss calculation."""
        from core.types import OptionContract

        short_contract = mock_option_contract
        long_contract = OptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.70,
            ask=0.80,
            last=0.75,
            volume=800,
            open_interest=4000,
            implied_volatility=0.20,
        )

        spread = CreditSpread(
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            short_contract=short_contract,
            long_contract=long_contract,
        )

        # Max loss = (width - credit) * 100 = (5.0 - 0.50) * 100 = 450.0
        assert spread.max_loss == 450.0

    def test_max_profit_calculation(self, mock_option_contract):
        """Test max profit calculation."""
        from core.types import OptionContract

        short_contract = mock_option_contract
        long_contract = OptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.70,
            ask=0.80,
            last=0.75,
            volume=800,
            open_interest=4000,
            implied_volatility=0.20,
        )

        spread = CreditSpread(
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            short_contract=short_contract,
            long_contract=long_contract,
        )

        # Max profit = credit * 100 = 0.50 * 100 = 50.0
        assert spread.max_profit == 50.0

    def test_zero_bid_ask_handling(self):
        """Test handling when bid/ask is zero."""
        from core.types import OptionContract

        # Edge case: zero bid
        zero_bid_contract = OptionContract(
            symbol="SPY240215P00470000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=470.0,
            option_type="put",
            bid=0.0,
            ask=0.10,
            last=0.05,
            volume=100,
            open_interest=500,
            implied_volatility=0.15,
        )

        normal_contract = OptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.0,
            ask=0.05,
            last=0.02,
            volume=50,
            open_interest=200,
            implied_volatility=0.12,
        )

        spread = CreditSpread(
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            short_contract=zero_bid_contract,
            long_contract=normal_contract,
        )

        # Should handle without division by zero
        # Credit = (0 + 0.10)/2 - (0 + 0.05)/2 = 0.05 - 0.025 = 0.025
        assert spread.credit == 0.025
