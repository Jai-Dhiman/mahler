"""Tests for options screener.

These tests ensure the screener correctly filters and scores
credit spread opportunities, handling edge cases in calculations.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.broker.types import OptionContract, OptionsChain
from core.analysis.iv_rank import IVMetrics


class TestScreenerFiltering:
    """Test OptionsScreener filtering logic."""

    @pytest.fixture
    def valid_expiration(self):
        """Return an expiration 35 days out (within 30-45 DTE)."""
        return (datetime.now() + timedelta(days=35)).strftime("%Y-%m-%d")

    @pytest.fixture
    def mock_chain(self, valid_expiration):
        """Create a mock options chain with valid contracts.

        Contracts must meet:
        - volume >= 10, open_interest >= 100
        - bid > 0, ask > 0
        - bid-ask spread <= 10% of mid
        - short strike delta in 0.20-0.30 range
        - credit >= 25% of width
        """
        contracts = [
            # Valid put contracts for bull put spread
            # Short strike: delta=-0.25 (within 0.20-0.30 range)
            OptionContract(
                symbol="SPY240215P00570000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=570.0,  # $5 width with long at 565
                option_type="put",
                bid=2.00,  # Mid = 2.05, spread = 0.10/2.05 = 4.9% (OK)
                ask=2.10,
                last=2.05,
                volume=500,  # Above min 10
                open_interest=2000,  # Above min 100
                delta=-0.25,  # In 0.20-0.30 range
                gamma=0.02,
                theta=-0.03,
                vega=0.15,
                implied_volatility=0.22,
            ),
            # Long strike for bull put spread
            # Bid-ask spread must be <= 10% of mid: 0.05/0.75 = 6.7% (OK)
            OptionContract(
                symbol="SPY240215P00565000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=565.0,
                option_type="put",
                bid=0.72,  # Mid = 0.75, spread = 0.06/0.75 = 8% (OK)
                ask=0.78,
                last=0.75,
                volume=400,
                open_interest=1500,
                delta=-0.15,
                gamma=0.015,
                theta=-0.02,
                vega=0.12,
                implied_volatility=0.20,
            ),
            # Valid call contracts for bear call spread
            OptionContract(
                symbol="SPY240215C00590000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=590.0,
                option_type="call",
                bid=1.90,  # Mid = 1.95
                ask=2.00,
                last=1.95,
                volume=450,
                open_interest=1800,
                delta=0.25,  # In 0.20-0.30 range
                gamma=0.02,
                theta=-0.03,
                vega=0.14,
                implied_volatility=0.21,
            ),
            # Long strike for bear call spread
            # Bid-ask spread must be <= 10% of mid: 0.06/0.75 = 8% (OK)
            OptionContract(
                symbol="SPY240215C00595000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=595.0,
                option_type="call",
                bid=0.72,  # Mid = 0.75, spread = 0.06/0.75 = 8% (OK)
                ask=0.78,
                last=0.75,
                volume=350,
                open_interest=1400,
                delta=0.15,
                gamma=0.015,
                theta=-0.02,
                vega=0.11,
                implied_volatility=0.19,
            ),
        ]

        return OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[valid_expiration],
            contracts=contracts,
        )

    @pytest.fixture
    def high_iv_metrics(self):
        """Create IV metrics with high IV percentile."""
        return IVMetrics(
            current_iv=0.22,
            iv_rank=65.0,
            iv_percentile=70.0,  # Above 50% threshold
            iv_high=0.30,
            iv_low=0.15,
        )

    @pytest.fixture
    def low_iv_metrics(self):
        """Create IV metrics with low IV percentile."""
        return IVMetrics(
            current_iv=0.15,
            iv_rank=30.0,
            iv_percentile=25.0,  # Below 50% threshold
            iv_high=0.25,
            iv_low=0.12,
        )

    def test_screen_with_high_iv(self, mock_chain, high_iv_metrics):
        """Test screener finds opportunities when IV is high."""
        from core.analysis.screener import OptionsScreener

        screener = OptionsScreener()
        opportunities = screener.screen_chain(mock_chain, high_iv_metrics)

        # Should find at least one opportunity
        assert len(opportunities) > 0

    def test_screen_with_low_iv_returns_empty(self, mock_chain, low_iv_metrics):
        """Test screener returns empty when IV percentile is too low."""
        from core.analysis.screener import OptionsScreener

        screener = OptionsScreener()
        opportunities = screener.screen_chain(mock_chain, low_iv_metrics)

        # Should return empty - IV percentile below threshold
        assert len(opportunities) == 0

    def test_liquidity_filter(self, valid_expiration, high_iv_metrics):
        """Test contracts with low liquidity are filtered out."""
        from core.analysis.screener import OptionsScreener

        # Create chain with low liquidity contracts
        illiquid_contracts = [
            OptionContract(
                symbol="SPY240215P00470000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=470.0,
                option_type="put",
                bid=1.20,
                ask=1.30,
                last=1.25,
                volume=5,  # Below min_volume (10)
                open_interest=50,  # Below min_open_interest (100)
                delta=-0.25,
            ),
            OptionContract(
                symbol="SPY240215P00465000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=465.0,
                option_type="put",
                bid=0.70,
                ask=0.80,
                last=0.75,
                volume=3,
                open_interest=30,
                delta=-0.15,
            ),
        ]

        chain = OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[valid_expiration],
            contracts=illiquid_contracts,
        )

        screener = OptionsScreener()
        opportunities = screener.screen_chain(chain, high_iv_metrics)

        # Should return empty - all contracts filtered by liquidity
        assert len(opportunities) == 0

    def test_bid_ask_spread_filter(self, valid_expiration, high_iv_metrics):
        """Test contracts with wide bid-ask spread are filtered."""
        from core.analysis.screener import OptionsScreener

        # Create chain with wide bid-ask spreads
        wide_spread_contracts = [
            OptionContract(
                symbol="SPY240215P00470000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=470.0,
                option_type="put",
                bid=1.00,
                ask=1.50,  # 50% spread - way above 10% limit
                last=1.25,
                volume=100,
                open_interest=500,
                delta=-0.25,
            ),
            OptionContract(
                symbol="SPY240215P00465000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=465.0,
                option_type="put",
                bid=0.50,
                ask=1.00,  # 100% spread
                last=0.75,
                volume=80,
                open_interest=400,
                delta=-0.15,
            ),
        ]

        chain = OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[valid_expiration],
            contracts=wide_spread_contracts,
        )

        screener = OptionsScreener()
        opportunities = screener.screen_chain(chain, high_iv_metrics)

        # Should return empty - all contracts filtered by bid-ask spread
        assert len(opportunities) == 0

    def test_zero_bid_filtered(self, valid_expiration, high_iv_metrics):
        """Test contracts with zero bid are filtered."""
        from core.analysis.screener import OptionsScreener

        # Create chain with zero bid
        zero_bid_contracts = [
            OptionContract(
                symbol="SPY240215P00470000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=470.0,
                option_type="put",
                bid=0.0,  # Zero bid
                ask=0.05,
                last=0.02,
                volume=100,
                open_interest=500,
                delta=-0.25,
            ),
            OptionContract(
                symbol="SPY240215P00465000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=465.0,
                option_type="put",
                bid=0.0,
                ask=0.03,
                last=0.01,
                volume=80,
                open_interest=400,
                delta=-0.15,
            ),
        ]

        chain = OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[valid_expiration],
            contracts=zero_bid_contracts,
        )

        screener = OptionsScreener()
        opportunities = screener.screen_chain(chain, high_iv_metrics)

        # Should return empty - zero bid filtered
        assert len(opportunities) == 0

    def test_delta_range_filter(self, valid_expiration, high_iv_metrics):
        """Test contracts outside delta range are filtered."""
        from core.analysis.screener import OptionsScreener

        # Create chain with delta outside 0.20-0.30 range
        wrong_delta_contracts = [
            OptionContract(
                symbol="SPY240215P00470000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=470.0,
                option_type="put",
                bid=1.20,
                ask=1.30,
                last=1.25,
                volume=100,
                open_interest=500,
                delta=-0.10,  # Too low (below 0.20)
            ),
            OptionContract(
                symbol="SPY240215P00465000",
                underlying="SPY",
                expiration=valid_expiration,
                strike=465.0,
                option_type="put",
                bid=0.70,
                ask=0.80,
                last=0.75,
                volume=80,
                open_interest=400,
                delta=-0.05,  # Too low
            ),
        ]

        chain = OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[valid_expiration],
            contracts=wrong_delta_contracts,
        )

        screener = OptionsScreener()
        opportunities = screener.screen_chain(chain, high_iv_metrics)

        # Should return empty - deltas outside range
        assert len(opportunities) == 0

    def test_empty_chain_handled(self, high_iv_metrics):
        """Test empty options chain is handled gracefully."""
        from core.analysis.screener import OptionsScreener

        empty_chain = OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[],
            contracts=[],
        )

        screener = OptionsScreener()
        opportunities = screener.screen_chain(empty_chain, high_iv_metrics)

        assert len(opportunities) == 0

    def test_dte_range_filter(self, high_iv_metrics):
        """Test expirations outside DTE range are filtered."""
        from core.analysis.screener import OptionsScreener

        # Expiration too soon (20 DTE, below 30)
        too_soon = (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
        # Expiration too far (60 DTE, above 45)
        too_far = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")

        contracts = [
            OptionContract(
                symbol="SPY240215P00470000",
                underlying="SPY",
                expiration=too_soon,
                strike=470.0,
                option_type="put",
                bid=1.20,
                ask=1.30,
                last=1.25,
                volume=100,
                open_interest=500,
                delta=-0.25,
            ),
            OptionContract(
                symbol="SPY240215P00465000",
                underlying="SPY",
                expiration=too_far,
                strike=465.0,
                option_type="put",
                bid=0.70,
                ask=0.80,
                last=0.75,
                volume=80,
                open_interest=400,
                delta=-0.15,
            ),
        ]

        chain = OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[too_soon, too_far],
            contracts=contracts,
        )

        screener = OptionsScreener()
        opportunities = screener.screen_chain(chain, high_iv_metrics)

        # Should return empty - all expirations outside DTE range
        assert len(opportunities) == 0


class TestScreenerScoring:
    """Test spread scoring calculations."""

    def test_score_components(self):
        """Test individual score components."""
        from core.analysis.screener import OptionsScreener, ScoredSpread
        from core.types import CreditSpread, SpreadType, OptionContract as CoreOptionContract

        # Verify scoring math
        # IV score = iv_percentile / 100
        # Delta score = 1 - abs(abs(delta) - 0.25) * 4 (peaks at 0.25)
        # Credit score = min(credit/width, 0.5) * 2
        # EV score = max(0, ev) / (width * 100)

        iv_metrics = IVMetrics(
            current_iv=0.22,
            iv_rank=65.0,
            iv_percentile=60.0,
            iv_high=0.30,
            iv_low=0.15,
        )

        # Create a spread with known values
        short_contract = CoreOptionContract(
            symbol="SPY240215P00470000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=470.0,
            option_type="put",
            bid=1.20,
            ask=1.30,
            last=1.25,
            volume=100,
            open_interest=500,
            implied_volatility=0.22,
        )
        long_contract = CoreOptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.70,
            ask=0.80,
            last=0.75,
            volume=80,
            open_interest=400,
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

        screener = OptionsScreener()
        scored = screener._score_spread(spread, iv_metrics, 0.25)  # Delta 0.25

        # Verify score is a valid number between 0 and 1
        assert 0 <= scored.score <= 1
        assert scored.probability_otm == 0.75  # 1 - 0.25

    def test_scoring_edge_cases(self):
        """Test scoring with edge case values."""
        from core.analysis.screener import OptionsScreener
        from core.types import CreditSpread, SpreadType, OptionContract as CoreOptionContract

        iv_metrics = IVMetrics(
            current_iv=0.22,
            iv_rank=0.0,  # Edge: zero IV rank
            iv_percentile=0.0,  # Edge: zero percentile
            iv_high=0.22,
            iv_low=0.22,  # Same as current - edge case
        )

        short_contract = CoreOptionContract(
            symbol="SPY240215P00470000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=470.0,
            option_type="put",
            bid=0.01,
            ask=0.02,  # Very small values
            last=0.015,
            volume=100,
            open_interest=500,
            implied_volatility=0.10,
        )
        long_contract = CoreOptionContract(
            symbol="SPY240215P00465000",
            underlying="SPY",
            expiration="2024-02-15",
            strike=465.0,
            option_type="put",
            bid=0.005,
            ask=0.01,
            last=0.007,
            volume=80,
            open_interest=400,
            implied_volatility=0.08,
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

        screener = OptionsScreener()
        # Should not raise even with edge case values
        scored = screener._score_spread(spread, iv_metrics, 0.50)

        assert scored.score >= 0
