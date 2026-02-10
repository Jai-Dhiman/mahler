from __future__ import annotations

"""Options screener for finding credit spread opportunities.

Includes regime-conditional scoring for adaptive weight selection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.analysis.greeks import days_to_expiry, years_to_expiry
from core.analysis.greeks_vollib import calculate_greeks_vollib
from core.analysis.iv_rank import IVMetrics
from core.broker.types import OptionContract, OptionsChain
from core.types import CreditSpread, Greeks, SpreadType
from core.types import OptionContract as CoreOptionContract


class MarketRegime(str, Enum):
    """Market regime types for conditional scoring."""

    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"


@dataclass
class ScoringWeights:
    """Weights for scoring spread opportunities.

    All weights must sum to 1.0.
    """

    iv_weight: float = 0.25
    delta_weight: float = 0.25
    credit_weight: float = 0.25
    ev_weight: float = 0.25

    def __post_init__(self):
        total = self.iv_weight + self.delta_weight + self.credit_weight + self.ev_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iv": self.iv_weight,
            "delta": self.delta_weight,
            "credit": self.credit_weight,
            "ev": self.ev_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScoringWeights:
        """Create from dictionary."""
        return cls(
            iv_weight=data.get("iv", 0.25),
            delta_weight=data.get("delta", 0.25),
            credit_weight=data.get("credit", 0.25),
            ev_weight=data.get("ev", 0.25),
        )


class RegimeConditionalScorer:
    """Provides regime-specific scoring weights.

    Research shows different factors matter more in different regimes:
    - Bull Low Vol: EV-focused (trend following works, let winners run)
    - Bull High Vol: Delta-focused (protection matters, manage risk)
    - Bear Low Vol: Credit-focused (premium capture in range-bound markets)
    - Bear High Vol: IV-focused (sell high IV, tight delta control)
    """

    DEFAULT_WEIGHTS: dict[MarketRegime, ScoringWeights] = {
        MarketRegime.BULL_LOW_VOL: ScoringWeights(
            iv_weight=0.20,
            delta_weight=0.20,
            credit_weight=0.25,
            ev_weight=0.35,
        ),
        MarketRegime.BULL_HIGH_VOL: ScoringWeights(
            iv_weight=0.25,
            delta_weight=0.35,
            credit_weight=0.20,
            ev_weight=0.20,
        ),
        MarketRegime.BEAR_LOW_VOL: ScoringWeights(
            iv_weight=0.20,
            delta_weight=0.25,
            credit_weight=0.35,
            ev_weight=0.20,
        ),
        MarketRegime.BEAR_HIGH_VOL: ScoringWeights(
            iv_weight=0.35,
            delta_weight=0.30,
            credit_weight=0.20,
            ev_weight=0.15,
        ),
    }

    def __init__(
        self,
        custom_weights: dict[MarketRegime, ScoringWeights] | None = None,
    ):
        """Initialize with optional custom weights.

        Args:
            custom_weights: Override weights for specific regimes
        """
        self.weights = {**self.DEFAULT_WEIGHTS}
        if custom_weights:
            self.weights.update(custom_weights)

    def get_weights(self, regime: MarketRegime | str | None) -> ScoringWeights:
        """Get scoring weights for a regime.

        Args:
            regime: Market regime (enum, string, or None for default)

        Returns:
            ScoringWeights for the regime
        """
        if regime is None:
            return ScoringWeights()  # Default equal weights

        # Convert string to enum if needed
        if isinstance(regime, str):
            try:
                regime = MarketRegime(regime.lower())
            except ValueError:
                return ScoringWeights()

        return self.weights.get(regime, ScoringWeights())

    def update_weights(
        self,
        regime: MarketRegime | str,
        weights: ScoringWeights,
    ) -> None:
        """Update weights for a regime (from optimization).

        Args:
            regime: Market regime to update
            weights: New weights to use
        """
        if isinstance(regime, str):
            regime = MarketRegime(regime.lower())
        self.weights[regime] = weights

    def load_from_dict(self, weights_dict: dict[str, dict]) -> None:
        """Load weights from a dictionary (e.g., from KV cache).

        Args:
            weights_dict: Dict of regime_str -> weight_values
        """
        for regime_str, weight_values in weights_dict.items():
            try:
                regime = MarketRegime(regime_str.lower())
                self.weights[regime] = ScoringWeights.from_dict(weight_values)
            except (ValueError, KeyError):
                continue


@dataclass
class ScreenerConfig:
    """Configuration for the options screener."""

    # DTE range
    min_dte: int = 30
    max_dte: int = 45

    # Delta range for short strike
    # Backtest validated (2007-2025, QQQ): 0.05-0.15 range has +59% CAGR,
    # higher win rate (69.9% vs 67.9%), better profit factor (6.10 vs 3.16)
    min_delta: float = 0.05
    max_delta: float = 0.15

    # IV requirements
    # Research: IV Percentile is more reliable than IV Rank because it considers
    # all trading days over the past year, not just 52-week high/low extremes
    # IV filter removed based on backtest: +59% CAGR improvement by trading
    # in all IV environments. Strategy thrives in high-IV bear markets.
    min_iv_percentile: float = 0.0  # IV filter removed per backtest validation
    min_iv_rank: float = 50.0  # Keep as secondary signal for AI context

    # Minimum credit (as percentage of width)
    min_credit_pct: float = 0.12  # 12% of spread width (lowered for backtest-validated delta range)

    # Spread width range
    # Note: $2+ wide spreads have significantly better liquidity and fill rates
    # $1 wide spreads often have wide bid-ask spreads and poor fills
    min_width: float = 2.0  # Increased from 1.0 for better liquidity
    max_width: float = 10.0

    # Liquidity filters
    min_open_interest: int = 100
    min_volume: int = 10
    max_bid_ask_spread_pct: float = 0.08  # 8% of mid price (tightened from 10%)


@dataclass
class ScoredSpread:
    """A credit spread with a score for ranking."""

    spread: CreditSpread
    score: float
    iv_rank: float
    expected_value: float
    probability_otm: float


class OptionsScreener:
    """Screens options chains for credit spread opportunities.

    Supports regime-conditional scoring for adaptive opportunity ranking.
    """

    # Target underlyings per PRD
    UNDERLYINGS = ["SPY", "QQQ", "IWM"]

    def __init__(
        self,
        config: ScreenerConfig | None = None,
        scorer: RegimeConditionalScorer | None = None,
    ):
        self.config = config or ScreenerConfig()
        self.scorer = scorer or RegimeConditionalScorer()

    def screen_chain(
        self,
        chain: OptionsChain,
        iv_metrics: IVMetrics,
        regime: MarketRegime | str | None = None,
    ) -> list[ScoredSpread]:
        """Screen an options chain for credit spread opportunities.

        Args:
            chain: Options chain data from broker
            iv_metrics: IV metrics for the underlying
            regime: Current market regime for conditional scoring (optional)

        Returns:
            List of scored spreads, sorted by score descending
        """
        # Use IV Percentile as primary filter (more reliable than IV Rank)
        # IV Percentile considers all 252 trading days, not just extremes
        if iv_metrics.iv_percentile < self.config.min_iv_percentile:
            return []

        # Get regime-specific weights
        weights = self.scorer.get_weights(regime)

        opportunities = []

        # Filter expirations to DTE range
        valid_expirations = [
            exp
            for exp in chain.expirations
            if self.config.min_dte <= days_to_expiry(exp) <= self.config.max_dte
        ]

        for expiration in valid_expirations:
            # Find bull put spreads (bullish/neutral)
            put_opportunities = self._find_bull_put_spreads(
                chain, expiration, iv_metrics, weights
            )
            opportunities.extend(put_opportunities)

            # Find bear call spreads (bearish/neutral)
            call_opportunities = self._find_bear_call_spreads(
                chain, expiration, iv_metrics, weights
            )
            opportunities.extend(call_opportunities)

        # Sort by score descending
        opportunities.sort(key=lambda x: x.score, reverse=True)

        return opportunities

    def _find_bull_put_spreads(
        self,
        chain: OptionsChain,
        expiration: str,
        iv_metrics: IVMetrics,
        weights: ScoringWeights,
    ) -> list[ScoredSpread]:
        """Find bull put spread opportunities (sell higher put, buy lower put)."""
        puts = chain.get_puts(expiration)
        puts = self._filter_for_liquidity(puts)

        if len(puts) < 2:
            return []

        # Sort by strike descending
        puts.sort(key=lambda x: x.strike, reverse=True)

        opportunities = []
        tte = years_to_expiry(expiration)

        for i, short_put in enumerate(puts):
            # Check short strike delta
            short_delta = self._get_delta(
                short_put, chain.underlying_price, tte, iv_metrics.current_iv, "put"
            )
            if not (self.config.min_delta <= abs(short_delta) <= self.config.max_delta):
                continue

            # Find long put candidates (lower strikes)
            for long_put in puts[i + 1 :]:
                width = short_put.strike - long_put.strike
                if not (self.config.min_width <= width <= self.config.max_width):
                    continue

                spread = self._build_spread(
                    chain.underlying,
                    SpreadType.BULL_PUT,
                    short_put,
                    long_put,
                    expiration,
                )

                if spread.credit <= 0:
                    continue

                # Check minimum credit
                credit_pct = spread.credit / width
                if credit_pct < self.config.min_credit_pct:
                    continue

                # Score the spread with regime-specific weights
                scored = self._score_spread(spread, iv_metrics, abs(short_delta), weights)
                opportunities.append(scored)

        return opportunities

    def _find_bear_call_spreads(
        self,
        chain: OptionsChain,
        expiration: str,
        iv_metrics: IVMetrics,
        weights: ScoringWeights,
    ) -> list[ScoredSpread]:
        """Find bear call spread opportunities (sell lower call, buy higher call)."""
        calls = chain.get_calls(expiration)
        calls = self._filter_for_liquidity(calls)

        if len(calls) < 2:
            return []

        # Sort by strike ascending
        calls.sort(key=lambda x: x.strike)

        opportunities = []
        tte = years_to_expiry(expiration)

        for i, short_call in enumerate(calls):
            # Check short strike delta
            short_delta = self._get_delta(
                short_call, chain.underlying_price, tte, iv_metrics.current_iv, "call"
            )
            if not (self.config.min_delta <= abs(short_delta) <= self.config.max_delta):
                continue

            # Find long call candidates (higher strikes)
            for long_call in calls[i + 1 :]:
                width = long_call.strike - short_call.strike
                if not (self.config.min_width <= width <= self.config.max_width):
                    continue

                spread = self._build_spread(
                    chain.underlying,
                    SpreadType.BEAR_CALL,
                    short_call,
                    long_call,
                    expiration,
                )

                if spread.credit <= 0:
                    continue

                # Check minimum credit
                credit_pct = spread.credit / width
                if credit_pct < self.config.min_credit_pct:
                    continue

                # Score the spread with regime-specific weights
                scored = self._score_spread(spread, iv_metrics, abs(short_delta), weights)
                opportunities.append(scored)

        return opportunities

    def _filter_for_liquidity(self, contracts: list[OptionContract]) -> list[OptionContract]:
        """Filter contracts for minimum liquidity."""
        filtered = []
        for c in contracts:
            if c.open_interest < self.config.min_open_interest:
                continue
            if c.volume < self.config.min_volume:
                continue
            if c.bid <= 0 or c.ask <= 0:
                continue

            # Check bid-ask spread
            mid = (c.bid + c.ask) / 2
            spread_pct = (c.ask - c.bid) / mid if mid > 0 else 1.0
            if spread_pct > self.config.max_bid_ask_spread_pct:
                continue

            filtered.append(c)

        return filtered

    def _get_delta(
        self,
        contract: OptionContract,
        spot: float,
        tte: float,
        iv: float,
        option_type: str,
    ) -> float:
        """Get delta for a contract (from broker or calculated).

        Uses vollib for accurate Greeks calculation when broker data unavailable.
        """
        if contract.delta is not None:
            return contract.delta

        # Calculate using vollib if not provided by broker
        greeks = calculate_greeks_vollib(
            spot=spot,
            strike=contract.strike,
            time_to_expiry=tte,
            volatility=iv,
            option_type=option_type,
        )
        return greeks.delta

    def _build_spread(
        self,
        underlying: str,
        spread_type: SpreadType,
        short_contract: OptionContract,
        long_contract: OptionContract,
        expiration: str,
    ) -> CreditSpread:
        """Build a CreditSpread from broker contracts."""
        # Convert to core types
        short_core = CoreOptionContract(
            symbol=short_contract.symbol,
            underlying=underlying,
            expiration=expiration,
            strike=short_contract.strike,
            option_type=short_contract.option_type,
            bid=short_contract.bid,
            ask=short_contract.ask,
            last=short_contract.last,
            volume=short_contract.volume,
            open_interest=short_contract.open_interest,
            implied_volatility=short_contract.implied_volatility or 0.0,
            greeks=Greeks(
                delta=short_contract.delta or 0.0,
                gamma=short_contract.gamma or 0.0,
                theta=short_contract.theta or 0.0,
                vega=short_contract.vega or 0.0,
            )
            if short_contract.delta
            else None,
        )

        long_core = CoreOptionContract(
            symbol=long_contract.symbol,
            underlying=underlying,
            expiration=expiration,
            strike=long_contract.strike,
            option_type=long_contract.option_type,
            bid=long_contract.bid,
            ask=long_contract.ask,
            last=long_contract.last,
            volume=long_contract.volume,
            open_interest=long_contract.open_interest,
            implied_volatility=long_contract.implied_volatility or 0.0,
            greeks=Greeks(
                delta=long_contract.delta or 0.0,
                gamma=long_contract.gamma or 0.0,
                theta=long_contract.theta or 0.0,
                vega=long_contract.vega or 0.0,
            )
            if long_contract.delta
            else None,
        )

        return CreditSpread(
            underlying=underlying,
            spread_type=spread_type,
            short_strike=short_contract.strike,
            long_strike=long_contract.strike,
            expiration=expiration,
            short_contract=short_core,
            long_contract=long_core,
        )

    def _score_spread(
        self,
        spread: CreditSpread,
        iv_metrics: IVMetrics,
        short_delta: float,
        weights: ScoringWeights,
    ) -> ScoredSpread:
        """Score a spread for ranking using regime-conditional weights.

        Score factors:
        - Higher IV percentile = better premium environment
        - Delta closer to 0.25 (sweet spot) = better probability
        - Higher credit/width ratio = better risk/reward
        - Expected value (credit * prob_win - max_loss * prob_loss)

        Args:
            spread: The credit spread to score
            iv_metrics: IV metrics for context
            short_delta: Delta of the short strike
            weights: Regime-specific scoring weights
        """
        # Probability of expiring OTM (rough estimate from delta)
        prob_otm = 1 - abs(short_delta)

        # Expected value per spread
        credit = spread.credit * 100  # Per contract
        max_loss = spread.max_loss
        expected_value = (credit * prob_otm) - (max_loss * (1 - prob_otm))

        # Score components (normalized)
        # Use IV percentile for scoring (more stable than IV rank)
        iv_score = iv_metrics.iv_percentile / 100  # 0-1
        # Peak at 0.10 (center of 0.05-0.15 range), multiplier 10 for appropriate dropoff
        # Clamp to 0-1 range to handle edge cases with out-of-range deltas
        delta_score = max(0, min(1, 1 - abs(abs(short_delta) - 0.10) * 10))
        credit_score = min(spread.credit / spread.width, 0.5) * 2  # 0-1
        ev_score = max(0, expected_value) / (spread.width * 100)  # Normalized by width

        # Weighted average using regime-specific weights
        score = (
            iv_score * weights.iv_weight
            + delta_score * weights.delta_weight
            + credit_score * weights.credit_weight
            + ev_score * weights.ev_weight
        )

        return ScoredSpread(
            spread=spread,
            score=score,
            iv_rank=iv_metrics.iv_rank,  # Keep IV rank for display/AI context
            expected_value=expected_value,
            probability_otm=prob_otm,
        )
