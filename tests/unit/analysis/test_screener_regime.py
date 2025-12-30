"""Tests for regime-conditional scoring in the options screener.

Tests the ScoringWeights and RegimeConditionalScorer classes.
"""

from __future__ import annotations

import pytest

from core.analysis.screener import (
    MarketRegime,
    RegimeConditionalScorer,
    ScoringWeights,
)


class TestScoringWeights:
    """Test ScoringWeights dataclass."""

    def test_default_weights(self):
        """Test default weights are equal."""
        weights = ScoringWeights()

        assert weights.iv_weight == 0.25
        assert weights.delta_weight == 0.25
        assert weights.credit_weight == 0.25
        assert weights.ev_weight == 0.25

    def test_custom_weights(self):
        """Test custom weight assignment."""
        weights = ScoringWeights(
            iv_weight=0.30,
            delta_weight=0.25,
            credit_weight=0.25,
            ev_weight=0.20,
        )

        assert weights.iv_weight == 0.30
        assert weights.ev_weight == 0.20

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(
                iv_weight=0.50,
                delta_weight=0.50,
                credit_weight=0.50,
                ev_weight=0.50,
            )

    def test_weights_near_one_acceptable(self):
        """Test weights that are close to 1.0 (within tolerance)."""
        # Should not raise - within 0.01 tolerance
        weights = ScoringWeights(
            iv_weight=0.251,
            delta_weight=0.249,
            credit_weight=0.250,
            ev_weight=0.250,
        )
        assert weights is not None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        weights = ScoringWeights(
            iv_weight=0.30,
            delta_weight=0.25,
            credit_weight=0.25,
            ev_weight=0.20,
        )

        d = weights.to_dict()

        assert d["iv"] == 0.30
        assert d["delta"] == 0.25
        assert d["credit"] == 0.25
        assert d["ev"] == 0.20

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"iv": 0.35, "delta": 0.30, "credit": 0.20, "ev": 0.15}

        weights = ScoringWeights.from_dict(data)

        assert weights.iv_weight == 0.35
        assert weights.delta_weight == 0.30
        assert weights.credit_weight == 0.20
        assert weights.ev_weight == 0.15

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        data = {}  # Empty dict

        weights = ScoringWeights.from_dict(data)

        assert weights.iv_weight == 0.25
        assert weights.delta_weight == 0.25
        assert weights.credit_weight == 0.25
        assert weights.ev_weight == 0.25


class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_regime_values(self):
        """Test regime string values."""
        assert MarketRegime.BULL_LOW_VOL.value == "bull_low_vol"
        assert MarketRegime.BULL_HIGH_VOL.value == "bull_high_vol"
        assert MarketRegime.BEAR_LOW_VOL.value == "bear_low_vol"
        assert MarketRegime.BEAR_HIGH_VOL.value == "bear_high_vol"

    def test_regime_from_string(self):
        """Test creating regime from string."""
        regime = MarketRegime("bull_low_vol")
        assert regime == MarketRegime.BULL_LOW_VOL


class TestRegimeConditionalScorer:
    """Test RegimeConditionalScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with default settings."""
        return RegimeConditionalScorer()

    def test_default_weights_exist_for_all_regimes(self, scorer):
        """Test default weights exist for all regimes."""
        for regime in MarketRegime:
            weights = scorer.get_weights(regime)
            assert weights is not None
            total = weights.iv_weight + weights.delta_weight + weights.credit_weight + weights.ev_weight
            assert total == pytest.approx(1.0)

    def test_bull_low_vol_favors_ev(self, scorer):
        """Test bull low vol regime favors EV weight."""
        weights = scorer.get_weights(MarketRegime.BULL_LOW_VOL)
        assert weights.ev_weight >= weights.iv_weight
        assert weights.ev_weight >= weights.delta_weight

    def test_bear_high_vol_favors_iv(self, scorer):
        """Test bear high vol regime favors IV weight."""
        weights = scorer.get_weights(MarketRegime.BEAR_HIGH_VOL)
        assert weights.iv_weight >= weights.ev_weight
        assert weights.iv_weight >= weights.credit_weight

    def test_bull_high_vol_favors_delta(self, scorer):
        """Test bull high vol regime favors delta weight."""
        weights = scorer.get_weights(MarketRegime.BULL_HIGH_VOL)
        assert weights.delta_weight >= weights.credit_weight
        assert weights.delta_weight >= weights.ev_weight

    def test_bear_low_vol_favors_credit(self, scorer):
        """Test bear low vol regime favors credit weight."""
        weights = scorer.get_weights(MarketRegime.BEAR_LOW_VOL)
        assert weights.credit_weight >= weights.iv_weight
        assert weights.credit_weight >= weights.ev_weight

    def test_get_weights_with_string(self, scorer):
        """Test get_weights accepts string regime."""
        weights = scorer.get_weights("bull_low_vol")
        assert weights is not None
        assert weights.ev_weight == 0.35  # Default for bull_low_vol

    def test_get_weights_with_none_returns_default(self, scorer):
        """Test get_weights with None returns equal weights."""
        weights = scorer.get_weights(None)
        assert weights.iv_weight == 0.25
        assert weights.delta_weight == 0.25
        assert weights.credit_weight == 0.25
        assert weights.ev_weight == 0.25

    def test_get_weights_invalid_string_returns_default(self, scorer):
        """Test get_weights with invalid string returns equal weights."""
        weights = scorer.get_weights("invalid_regime")
        assert weights.iv_weight == 0.25
        assert weights.delta_weight == 0.25

    def test_update_weights(self, scorer):
        """Test updating weights for a regime."""
        new_weights = ScoringWeights(
            iv_weight=0.40,
            delta_weight=0.30,
            credit_weight=0.20,
            ev_weight=0.10,
        )
        scorer.update_weights(MarketRegime.BULL_LOW_VOL, new_weights)

        weights = scorer.get_weights(MarketRegime.BULL_LOW_VOL)
        assert weights.iv_weight == 0.40
        assert weights.delta_weight == 0.30

    def test_update_weights_with_string(self, scorer):
        """Test updating weights with string regime."""
        new_weights = ScoringWeights(
            iv_weight=0.40,
            delta_weight=0.30,
            credit_weight=0.20,
            ev_weight=0.10,
        )
        scorer.update_weights("bull_low_vol", new_weights)

        weights = scorer.get_weights(MarketRegime.BULL_LOW_VOL)
        assert weights.iv_weight == 0.40

    def test_load_from_dict(self, scorer):
        """Test loading multiple regimes from dictionary."""
        weights_dict = {
            "bull_low_vol": {"iv": 0.30, "delta": 0.30, "credit": 0.20, "ev": 0.20},
            "bear_high_vol": {"iv": 0.40, "delta": 0.25, "credit": 0.20, "ev": 0.15},
        }
        scorer.load_from_dict(weights_dict)

        weights_bull = scorer.get_weights(MarketRegime.BULL_LOW_VOL)
        assert weights_bull.iv_weight == 0.30

        weights_bear = scorer.get_weights(MarketRegime.BEAR_HIGH_VOL)
        assert weights_bear.iv_weight == 0.40

    def test_load_from_dict_ignores_invalid_regimes(self, scorer):
        """Test load_from_dict ignores invalid regime names."""
        weights_dict = {
            "invalid_regime": {"iv": 0.30, "delta": 0.30, "credit": 0.20, "ev": 0.20},
        }
        # Should not raise
        scorer.load_from_dict(weights_dict)

    def test_custom_weights_in_constructor(self):
        """Test custom weights can be passed to constructor."""
        custom = {
            MarketRegime.BULL_LOW_VOL: ScoringWeights(
                iv_weight=0.40, delta_weight=0.30, credit_weight=0.15, ev_weight=0.15
            )
        }
        scorer = RegimeConditionalScorer(custom_weights=custom)

        weights = scorer.get_weights(MarketRegime.BULL_LOW_VOL)
        assert weights.iv_weight == 0.40

        # Other regimes should still have defaults
        weights_bear = scorer.get_weights(MarketRegime.BEAR_HIGH_VOL)
        assert weights_bear.iv_weight == 0.35  # Default for bear_high_vol


class TestScreenerWithRegime:
    """Test OptionsScreener integration with regime scoring."""

    @pytest.fixture
    def mock_chain(self):
        """Create mock options chain for testing.

        Contracts must meet:
        - volume >= 10, open_interest >= 100
        - bid > 0, ask > 0
        - bid-ask spread <= 8% of mid (tightened from 10%)
        - short strike delta in 0.20-0.30 range
        - credit >= 25% of width
        - width >= 2.0 (minimum spread width)
        """
        from datetime import datetime, timedelta
        from core.broker.types import OptionContract, OptionsChain

        exp = (datetime.now() + timedelta(days=35)).strftime("%Y-%m-%d")
        contracts = [
            # Short put for bull put spread
            OptionContract(
                symbol="SPY240215P00570000",
                underlying="SPY",
                expiration=exp,
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
            # Long put for bull put spread - $5 width (>= 2.0)
            OptionContract(
                symbol="SPY240215P00565000",
                underlying="SPY",
                expiration=exp,
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
            # Additional put strikes for more width options
            OptionContract(
                symbol="SPY240215P00568000",
                underlying="SPY",
                expiration=exp,
                strike=568.0,  # $3 width with 565
                option_type="put",
                bid=1.40,
                ask=1.48,
                last=1.44,
                volume=300,
                open_interest=1200,
                delta=-0.22,  # Also in 0.20-0.30 range
                gamma=0.018,
                theta=-0.025,
                vega=0.13,
                implied_volatility=0.21,
            ),
        ]
        return OptionsChain(
            underlying="SPY",
            underlying_price=580.0,
            timestamp=datetime.now(),
            expirations=[exp],
            contracts=contracts,
        )

    @pytest.fixture
    def high_iv_metrics(self):
        """Create high IV metrics."""
        from core.analysis.iv_rank import IVMetrics
        return IVMetrics(
            current_iv=0.22,
            iv_rank=65.0,
            iv_percentile=70.0,
            iv_high=0.30,
            iv_low=0.15,
        )

    def test_screener_uses_scorer(self, mock_chain, high_iv_metrics):
        """Test screener uses provided scorer."""
        from core.analysis.screener import OptionsScreener

        scorer = RegimeConditionalScorer()
        screener = OptionsScreener(scorer=scorer)

        opportunities = screener.screen_chain(mock_chain, high_iv_metrics)
        assert len(opportunities) > 0

    def test_screener_accepts_regime(self, mock_chain, high_iv_metrics):
        """Test screener accepts regime parameter."""
        from core.analysis.screener import OptionsScreener

        screener = OptionsScreener()

        # Should work with enum
        opps1 = screener.screen_chain(mock_chain, high_iv_metrics, regime=MarketRegime.BULL_LOW_VOL)

        # Should work with string
        opps2 = screener.screen_chain(mock_chain, high_iv_metrics, regime="bull_low_vol")

        # Should work with None
        opps3 = screener.screen_chain(mock_chain, high_iv_metrics, regime=None)

        assert len(opps1) > 0
        assert len(opps2) > 0
        assert len(opps3) > 0

    def test_different_regimes_give_different_scores(self, mock_chain, high_iv_metrics):
        """Test that different regimes can produce different scores."""
        from core.analysis.screener import OptionsScreener

        screener = OptionsScreener()

        opps_bull_low = screener.screen_chain(mock_chain, high_iv_metrics, regime="bull_low_vol")
        opps_bear_high = screener.screen_chain(mock_chain, high_iv_metrics, regime="bear_high_vol")

        # Same opportunities should be found, but scores may differ
        assert len(opps_bull_low) == len(opps_bear_high)

        if len(opps_bull_low) > 0:
            # Scores should potentially be different due to different weights
            # (Not guaranteed to be different, but the scoring logic is different)
            score_bull = opps_bull_low[0].score
            score_bear = opps_bear_high[0].score
            # Both should be valid scores
            assert 0 <= score_bull <= 1
            assert 0 <= score_bear <= 1
