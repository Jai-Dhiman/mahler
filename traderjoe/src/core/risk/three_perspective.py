"""Three-Perspective Risk Manager for V2 multi-agent system.

Provides VIX-weighted risk assessment from three perspectives:
- Aggressive: Uses full 2% per-trade limit
- Neutral: Uses 75% of limits
- Conservative: Uses 50% of limits

Two modes of operation:
1. Weighted Voting (default): Fast, deterministic VIX-based weighting
2. Agent-Based Deliberation: LLM-powered debate between risk perspectives

The weighted recommendation varies by market conditions:
- VIX > 30: Moderate-conservative weighting (0.15, 0.45, 0.40) — bear regimes profitable per backtest
- VIX > 20: Neutral-heavy weighting (0.2, 0.5, 0.3)
- VIX <= 20: Balanced weighting (0.3, 0.5, 0.2)

Agent-based deliberation is inspired by TradingAgents paper:
"Three risk perspectives (aggressive, neutral, conservative) monitor
portfolio exposure and adjust strategies through natural language debate."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient
    from core.risk.position_sizer import PositionSizer, PositionSizeResult
    from core.types import CreditSpread, Position


class RiskPerspective(str, Enum):
    """Risk perspective types."""

    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    CONSERVATIVE = "conservative"


@dataclass
class PerspectiveAssessment:
    """Assessment from a single risk perspective."""

    perspective: RiskPerspective
    recommended_contracts: int
    position_size_multiplier: float  # 0.0-1.0
    key_factors: list[str]
    recommendation: Literal["enter", "skip", "reduce_size"]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "perspective": self.perspective.value,
            "recommended_contracts": self.recommended_contracts,
            "position_size_multiplier": self.position_size_multiplier,
            "key_factors": self.key_factors,
            "recommendation": self.recommendation,
        }


@dataclass
class ThreePerspectiveResult:
    """Result of three-perspective risk assessment."""

    aggressive: PerspectiveAssessment
    neutral: PerspectiveAssessment
    conservative: PerspectiveAssessment
    weighted_contracts: int
    weights_used: dict[RiskPerspective, float]
    vix_at_assessment: float
    deliberation_summary: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "aggressive": self.aggressive.to_dict(),
            "neutral": self.neutral.to_dict(),
            "conservative": self.conservative.to_dict(),
            "weighted_contracts": self.weighted_contracts,
            "weights_used": {k.value: v for k, v in self.weights_used.items()},
            "vix_at_assessment": self.vix_at_assessment,
            "deliberation_summary": self.deliberation_summary,
        }

    @property
    def consensus_recommendation(self) -> Literal["enter", "skip", "reduce_size"]:
        """Get the consensus recommendation based on weighted perspectives."""
        if self.weighted_contracts == 0:
            return "skip"
        elif self.weighted_contracts < self.aggressive.recommended_contracts:
            return "reduce_size"
        return "enter"


@dataclass
class ThreePerspectiveConfig:
    """Configuration for three-perspective risk assessment."""

    # Perspective multipliers for position size limits
    aggressive_multiplier: float = 1.0  # Full 2% per-trade
    neutral_multiplier: float = 0.75  # 75% of limits
    conservative_multiplier: float = 0.50  # 50% of limits

    # VIX thresholds for weight adjustment
    vix_high_threshold: float = 30.0  # Conservative-heavy above this
    vix_moderate_threshold: float = 20.0  # Neutral-heavy above this

    # Weight distributions
    # Format: [aggressive, neutral, conservative]
    # Backtest validated: bear regimes are profitable, less conservative in high VIX
    high_vix_weights: list[float] = field(
        default_factory=lambda: [0.15, 0.45, 0.40]
    )  # Less conservative-heavy (bear regimes profitable per backtest)
    moderate_vix_weights: list[float] = field(
        default_factory=lambda: [0.2, 0.5, 0.3]
    )  # Neutral-heavy
    low_vix_weights: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.2]
    )  # Balanced


class ThreePerspectiveRiskManager:
    """Manager for three-perspective risk assessment.

    Uses a PositionSizer to calculate base position sizes, then applies
    perspective-specific multipliers and VIX-weighted averaging to produce
    a final position size recommendation.
    """

    def __init__(
        self,
        sizer: PositionSizer,
        config: ThreePerspectiveConfig | None = None,
    ):
        """Initialize the three-perspective risk manager.

        Args:
            sizer: PositionSizer instance for base calculations
            config: Optional configuration overrides
        """
        self.sizer = sizer
        self.config = config or ThreePerspectiveConfig()

    def _get_weights_for_vix(self, vix: float) -> dict[RiskPerspective, float]:
        """Get perspective weights based on VIX level.

        Args:
            vix: Current VIX level

        Returns:
            Dictionary mapping perspectives to their weights
        """
        if vix > self.config.vix_high_threshold:
            weights = self.config.high_vix_weights
        elif vix > self.config.vix_moderate_threshold:
            weights = self.config.moderate_vix_weights
        else:
            weights = self.config.low_vix_weights

        return {
            RiskPerspective.AGGRESSIVE: weights[0],
            RiskPerspective.NEUTRAL: weights[1],
            RiskPerspective.CONSERVATIVE: weights[2],
        }

    def _assess_perspective(
        self,
        perspective: RiskPerspective,
        base_result: PositionSizeResult,
        multiplier: float,
        vix: float,
    ) -> PerspectiveAssessment:
        """Generate assessment from a single perspective.

        Args:
            perspective: The risk perspective to use
            base_result: Base position sizing result
            multiplier: Perspective-specific size multiplier
            vix: Current VIX level

        Returns:
            PerspectiveAssessment for this perspective
        """
        # Apply multiplier to contracts
        adjusted_contracts = max(0, int(base_result.contracts * multiplier))

        # Build key factors based on perspective
        key_factors = []

        if perspective == RiskPerspective.AGGRESSIVE:
            key_factors.append("Using full position sizing limits")
            if vix < 20:
                key_factors.append("Low VIX environment favorable for premium selling")
            if adjusted_contracts > 0:
                key_factors.append(f"Risk {base_result.risk_percent:.1%} of equity")
        elif perspective == RiskPerspective.NEUTRAL:
            key_factors.append("Using 75% of position sizing limits")
            key_factors.append("Balanced approach to risk/reward")
            if base_result.reason:
                key_factors.append(f"Constraint: {base_result.reason}")
        else:  # CONSERVATIVE
            key_factors.append("Using 50% of position sizing limits")
            if vix > 25:
                key_factors.append("Elevated VIX suggests caution")
            key_factors.append("Prioritizing capital preservation")

        # Determine recommendation
        if adjusted_contracts == 0:
            recommendation: Literal["enter", "skip", "reduce_size"] = "skip"
            key_factors.append("Position size constraints prevent entry")
        elif adjusted_contracts < base_result.contracts * 0.5:
            recommendation = "reduce_size"
            key_factors.append("Significant size reduction applied")
        else:
            recommendation = "enter"

        return PerspectiveAssessment(
            perspective=perspective,
            recommended_contracts=adjusted_contracts,
            position_size_multiplier=multiplier,
            key_factors=key_factors,
            recommendation=recommendation,
        )

    def assess(
        self,
        spread: CreditSpread,
        account_equity: float,
        current_positions: list[Position],
        current_vix: float,
        spread_vanna: float | None = None,
        spread_volga: float | None = None,
    ) -> ThreePerspectiveResult:
        """Perform three-perspective risk assessment.

        Args:
            spread: The credit spread being considered
            account_equity: Current account equity
            current_positions: List of open positions
            current_vix: Current VIX level
            spread_vanna: Optional spread vanna for second-order Greeks
            spread_volga: Optional spread volga for second-order Greeks

        Returns:
            ThreePerspectiveResult with assessments and weighted recommendation
        """
        # Get base position sizing (without VIX adjustment - we'll weight by VIX separately)
        base_result = self.sizer.calculate_size(
            spread=spread,
            account_equity=account_equity,
            current_positions=current_positions,
            current_vix=None,  # Don't apply VIX reduction at base level
            spread_vanna=spread_vanna,
            spread_volga=spread_volga,
        )

        # Assess from each perspective
        aggressive = self._assess_perspective(
            RiskPerspective.AGGRESSIVE,
            base_result,
            self.config.aggressive_multiplier,
            current_vix,
        )

        neutral = self._assess_perspective(
            RiskPerspective.NEUTRAL,
            base_result,
            self.config.neutral_multiplier,
            current_vix,
        )

        conservative = self._assess_perspective(
            RiskPerspective.CONSERVATIVE,
            base_result,
            self.config.conservative_multiplier,
            current_vix,
        )

        # Get VIX-based weights
        weights = self._get_weights_for_vix(current_vix)

        # Calculate weighted average contracts
        weighted_sum = (
            aggressive.recommended_contracts * weights[RiskPerspective.AGGRESSIVE]
            + neutral.recommended_contracts * weights[RiskPerspective.NEUTRAL]
            + conservative.recommended_contracts * weights[RiskPerspective.CONSERVATIVE]
        )
        weighted_contracts = max(0, round(weighted_sum))

        # Ensure at least 1 contract if any perspective recommends entry
        if weighted_contracts == 0:
            if any(
                a.recommendation == "enter"
                for a in [aggressive, neutral, conservative]
            ):
                # All perspectives agree on skip, so keep 0
                if all(
                    a.recommendation == "skip"
                    for a in [aggressive, neutral, conservative]
                ):
                    weighted_contracts = 0
                else:
                    # At least one perspective says enter, so round up to 1
                    weighted_contracts = 1

        # Build deliberation summary
        vix_regime = (
            "high (conservative-heavy)"
            if current_vix > self.config.vix_high_threshold
            else "moderate (neutral-heavy)"
            if current_vix > self.config.vix_moderate_threshold
            else "low (balanced)"
        )

        summary_parts = [
            f"VIX {current_vix:.1f} ({vix_regime})",
            f"Aggressive: {aggressive.recommended_contracts} contracts ({aggressive.recommendation})",
            f"Neutral: {neutral.recommended_contracts} contracts ({neutral.recommendation})",
            f"Conservative: {conservative.recommended_contracts} contracts ({conservative.recommendation})",
            f"Weighted result: {weighted_contracts} contracts",
        ]
        deliberation_summary = " | ".join(summary_parts)

        return ThreePerspectiveResult(
            aggressive=aggressive,
            neutral=neutral,
            conservative=conservative,
            weighted_contracts=weighted_contracts,
            weights_used=weights,
            vix_at_assessment=current_vix,
            deliberation_summary=deliberation_summary,
        )

    def should_respect_conservative_skip(
        self,
        result: ThreePerspectiveResult,
        vix_threshold: float = 30.0,
    ) -> bool:
        """Check if conservative perspective's skip should be respected.

        In high VIX environments, if the conservative perspective recommends
        skipping the trade, this should generally be respected.

        Args:
            result: Three-perspective assessment result
            vix_threshold: VIX level above which to respect conservative skip

        Returns:
            True if conservative skip should be respected
        """
        if result.vix_at_assessment <= vix_threshold:
            return False

        return result.conservative.recommendation == "skip"


# =============================================================================
# Agent-Based Risk Deliberation System
# =============================================================================

# Prompts for risk perspective agents

AGGRESSIVE_RISK_SYSTEM = """You are an aggressive risk assessor for an options trading system. Your perspective prioritizes opportunity over caution.

Your role:
1. Argue for larger position sizes when opportunity is good
2. Point out when risk is overstated or manageable
3. Highlight favorable risk/reward setups
4. Push back against excessive caution that limits upside

Key principles:
- Credit spreads have defined risk - max loss is known
- Premium selling works best with size when conditions are right
- Missing good opportunities has a cost too
- You can accept higher heat when thesis is strong

Be specific and quantitative. Cite the actual numbers."""

AGGRESSIVE_RISK_USER = """Assess this trade from an AGGRESSIVE risk perspective:

**Trade:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Current Portfolio:**
- Equity: ${equity:.2f}
- Portfolio Heat: {portfolio_heat:.1%}
- Daily P/L: ${daily_pnl:.2f}
- Position Count: {position_count}

**Risk Parameters:**
- Max Risk Per Trade: 2% (${max_risk:.2f})
- Position Sizer Recommendation: {base_contracts} contracts

**Market Context:**
- VIX: {vix:.1f}
- Regime: {regime}

Argue for the most aggressive reasonable position size.

Respond in JSON:
{{
    "recommended_contracts": 1-10,
    "rationale": "Why this size is appropriate",
    "key_arguments": ["argument 1", "argument 2"],
    "acknowledged_risks": ["risk 1", "risk 2"],
    "conviction": 0.0-1.0
}}"""

NEUTRAL_RISK_SYSTEM = """You are a neutral risk assessor for an options trading system. Your perspective balances opportunity and caution.

Your role:
1. Weigh risk and reward objectively
2. Consider both upside potential and downside risk
3. Recommend balanced position sizes
4. Neither overly aggressive nor excessively conservative

Key principles:
- Size should match conviction level
- Follow standard risk management (2% rule, 10% heat)
- Account for current portfolio exposure
- Consider market conditions (VIX regime)

Be specific and quantitative. Cite the actual numbers."""

NEUTRAL_RISK_USER = """Assess this trade from a NEUTRAL risk perspective:

**Trade:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Current Portfolio:**
- Equity: ${equity:.2f}
- Portfolio Heat: {portfolio_heat:.1%}
- Daily P/L: ${daily_pnl:.2f}
- Position Count: {position_count}

**Risk Parameters:**
- Max Risk Per Trade: 2% (${max_risk:.2f})
- Position Sizer Recommendation: {base_contracts} contracts

**Market Context:**
- VIX: {vix:.1f}
- Regime: {regime}

Provide a balanced risk assessment.

Respond in JSON:
{{
    "recommended_contracts": 1-10,
    "rationale": "Why this size is appropriate",
    "key_arguments": ["argument 1", "argument 2"],
    "acknowledged_risks": ["risk 1", "risk 2"],
    "conviction": 0.0-1.0
}}"""

CONSERVATIVE_RISK_SYSTEM = """You are a conservative risk assessor for an options trading system. Your perspective prioritizes capital preservation.

Your role:
1. Argue for smaller position sizes to limit risk
2. Highlight risks that others might overlook
3. Push back against excessive aggression
4. Prioritize survival over profit maximization

Key principles:
- Capital preservation is paramount
- Avoid ruin at all costs
- Better to miss opportunities than suffer large losses
- Size down when uncertain

Be specific and quantitative. Cite the actual numbers."""

CONSERVATIVE_RISK_USER = """Assess this trade from a CONSERVATIVE risk perspective:

**Trade:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Current Portfolio:**
- Equity: ${equity:.2f}
- Portfolio Heat: {portfolio_heat:.1%}
- Daily P/L: ${daily_pnl:.2f}
- Position Count: {position_count}

**Risk Parameters:**
- Max Risk Per Trade: 2% (${max_risk:.2f})
- Position Sizer Recommendation: {base_contracts} contracts

**Market Context:**
- VIX: {vix:.1f}
- Regime: {regime}

Argue for the most conservative reasonable position size.

Respond in JSON:
{{
    "recommended_contracts": 0-5,
    "rationale": "Why this size is appropriate",
    "key_arguments": ["argument 1", "argument 2"],
    "acknowledged_risks": ["risk 1", "risk 2"],
    "conviction": 0.0-1.0
}}"""

RISK_FACILITATOR_SYSTEM = """You are a risk deliberation facilitator synthesizing three risk perspectives into a final recommendation.

You receive assessments from:
1. Aggressive perspective - argues for larger size
2. Neutral perspective - balanced view
3. Conservative perspective - argues for smaller size

Your role:
1. Weigh the arguments from each perspective
2. Consider market conditions (VIX level)
3. Determine the most appropriate position size
4. Explain why you weighted perspectives as you did

Decision framework:
- High VIX (>30): Weight conservative more heavily
- Normal VIX (20-30): Weight neutral more heavily
- Low VIX (<20): Can weight aggressive slightly more

Be decisive. Pick a final number and justify it."""

RISK_FACILITATOR_USER = """Synthesize these risk assessments into a final recommendation:

**Trade Summary:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- DTE: {dte}

**Aggressive Assessment:**
- Contracts: {aggressive_contracts}
- Rationale: {aggressive_rationale}
- Key Arguments: {aggressive_arguments}
- Conviction: {aggressive_conviction:.0%}

**Neutral Assessment:**
- Contracts: {neutral_contracts}
- Rationale: {neutral_rationale}
- Key Arguments: {neutral_arguments}
- Conviction: {neutral_conviction:.0%}

**Conservative Assessment:**
- Contracts: {conservative_contracts}
- Rationale: {conservative_rationale}
- Key Arguments: {conservative_arguments}
- Conviction: {conservative_conviction:.0%}

**Market Context:**
- VIX: {vix:.1f}
- Portfolio Heat: {portfolio_heat:.1%}

Synthesize into a final recommendation.

Respond in JSON:
{{
    "final_contracts": 0-10,
    "prevailing_perspective": "aggressive|neutral|conservative",
    "weighting_rationale": "Why you weighted perspectives this way",
    "key_factors": ["factor 1", "factor 2"],
    "concerns": ["concern 1", "concern 2"],
    "confidence": 0.0-1.0
}}"""


@dataclass
class RiskAgentAssessment:
    """Assessment from a single risk perspective agent."""

    perspective: RiskPerspective
    recommended_contracts: int
    rationale: str
    key_arguments: list[str]
    acknowledged_risks: list[str]
    conviction: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "perspective": self.perspective.value,
            "recommended_contracts": self.recommended_contracts,
            "rationale": self.rationale,
            "key_arguments": self.key_arguments,
            "acknowledged_risks": self.acknowledged_risks,
            "conviction": self.conviction,
        }


@dataclass
class RiskDeliberationResult:
    """Result of agent-based risk deliberation."""

    aggressive: RiskAgentAssessment
    neutral: RiskAgentAssessment
    conservative: RiskAgentAssessment
    final_contracts: int
    prevailing_perspective: str
    weighting_rationale: str
    key_factors: list[str]
    concerns: list[str]
    confidence: float
    vix_at_deliberation: float
    rounds_conducted: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "aggressive": self.aggressive.to_dict(),
            "neutral": self.neutral.to_dict(),
            "conservative": self.conservative.to_dict(),
            "final_contracts": self.final_contracts,
            "prevailing_perspective": self.prevailing_perspective,
            "weighting_rationale": self.weighting_rationale,
            "key_factors": self.key_factors,
            "concerns": self.concerns,
            "confidence": self.confidence,
            "vix_at_deliberation": self.vix_at_deliberation,
            "rounds_conducted": self.rounds_conducted,
        }

    @property
    def recommendation(self) -> Literal["enter", "skip", "reduce_size"]:
        """Get recommendation based on final contracts."""
        if self.final_contracts == 0:
            return "skip"
        elif self.final_contracts < self.aggressive.recommended_contracts:
            return "reduce_size"
        return "enter"


class RiskDeliberationManager:
    """Manages LLM-powered risk deliberation between three perspectives.

    This is an alternative to the weighted-voting ThreePerspectiveRiskManager.
    It uses actual LLM agents to debate and reach consensus on position sizing.

    Inspired by TradingAgents paper's risk deliberation approach.
    """

    def __init__(
        self,
        claude: ClaudeClient,
        sizer: PositionSizer,
        max_rounds: int = 2,
    ):
        """Initialize the risk deliberation manager.

        Args:
            claude: Claude client for LLM calls
            sizer: Position sizer for base calculations
            max_rounds: Maximum deliberation rounds (default 2)
        """
        self.claude = claude
        self.sizer = sizer
        self.max_rounds = max_rounds

    async def deliberate(
        self,
        spread: CreditSpread,
        account_equity: float,
        current_positions: list[Position],
        current_vix: float,
        market_regime: str | None = None,
        daily_pnl: float = 0.0,
    ) -> RiskDeliberationResult:
        """Run agent-based risk deliberation.

        Args:
            spread: The credit spread being considered
            account_equity: Current account equity
            current_positions: List of open positions
            current_vix: Current VIX level
            market_regime: Current market regime
            daily_pnl: Today's P/L

        Returns:
            RiskDeliberationResult with final recommendation
        """
        # Get base position sizing as starting point
        base_result = self.sizer.calculate_size(
            spread=spread,
            account_equity=account_equity,
            current_positions=current_positions,
            current_vix=None,  # Don't apply VIX at base level
        )

        # Calculate context values
        portfolio_heat = self._calculate_portfolio_heat(current_positions, account_equity)
        max_risk = account_equity * 0.02
        dte = self._calculate_dte(spread.expiration)

        context = {
            "underlying": spread.underlying,
            "spread_type": spread.spread_type.value.replace("_", " ").title(),
            "credit": spread.credit,
            "max_loss": spread.max_loss / 100,
            "expiration": spread.expiration,
            "dte": dte,
            "equity": account_equity,
            "portfolio_heat": portfolio_heat,
            "daily_pnl": daily_pnl,
            "position_count": len(current_positions),
            "max_risk": max_risk,
            "base_contracts": base_result.contracts,
            "vix": current_vix,
            "regime": market_regime or "unknown",
        }

        # Run all three perspective assessments in parallel conceptually
        # (In practice, we run them sequentially for simplicity)
        aggressive = await self._get_aggressive_assessment(context)
        neutral = await self._get_neutral_assessment(context)
        conservative = await self._get_conservative_assessment(context)

        # Facilitator synthesizes
        final = await self._facilitate_deliberation(
            aggressive=aggressive,
            neutral=neutral,
            conservative=conservative,
            context=context,
        )

        return RiskDeliberationResult(
            aggressive=aggressive,
            neutral=neutral,
            conservative=conservative,
            final_contracts=final["final_contracts"],
            prevailing_perspective=final["prevailing_perspective"],
            weighting_rationale=final["weighting_rationale"],
            key_factors=final["key_factors"],
            concerns=final["concerns"],
            confidence=final["confidence"],
            vix_at_deliberation=current_vix,
            rounds_conducted=1,
        )

    async def _get_aggressive_assessment(
        self, context: dict[str, Any]
    ) -> RiskAgentAssessment:
        """Get aggressive perspective assessment."""
        prompt = AGGRESSIVE_RISK_USER.format(**context)

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            AGGRESSIVE_RISK_SYSTEM,
        )

        data = self.claude._parse_json_response(response)

        return RiskAgentAssessment(
            perspective=RiskPerspective.AGGRESSIVE,
            recommended_contracts=data.get("recommended_contracts", context["base_contracts"]),
            rationale=data.get("rationale", ""),
            key_arguments=data.get("key_arguments", []),
            acknowledged_risks=data.get("acknowledged_risks", []),
            conviction=data.get("conviction", 0.5),
        )

    async def _get_neutral_assessment(
        self, context: dict[str, Any]
    ) -> RiskAgentAssessment:
        """Get neutral perspective assessment."""
        prompt = NEUTRAL_RISK_USER.format(**context)

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            NEUTRAL_RISK_SYSTEM,
        )

        data = self.claude._parse_json_response(response)

        return RiskAgentAssessment(
            perspective=RiskPerspective.NEUTRAL,
            recommended_contracts=data.get("recommended_contracts", context["base_contracts"]),
            rationale=data.get("rationale", ""),
            key_arguments=data.get("key_arguments", []),
            acknowledged_risks=data.get("acknowledged_risks", []),
            conviction=data.get("conviction", 0.5),
        )

    async def _get_conservative_assessment(
        self, context: dict[str, Any]
    ) -> RiskAgentAssessment:
        """Get conservative perspective assessment."""
        prompt = CONSERVATIVE_RISK_USER.format(**context)

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            CONSERVATIVE_RISK_SYSTEM,
        )

        data = self.claude._parse_json_response(response)

        return RiskAgentAssessment(
            perspective=RiskPerspective.CONSERVATIVE,
            recommended_contracts=data.get("recommended_contracts", max(0, context["base_contracts"] // 2)),
            rationale=data.get("rationale", ""),
            key_arguments=data.get("key_arguments", []),
            acknowledged_risks=data.get("acknowledged_risks", []),
            conviction=data.get("conviction", 0.5),
        )

    async def _facilitate_deliberation(
        self,
        aggressive: RiskAgentAssessment,
        neutral: RiskAgentAssessment,
        conservative: RiskAgentAssessment,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Facilitator synthesizes the three perspectives."""
        prompt = RISK_FACILITATOR_USER.format(
            underlying=context["underlying"],
            spread_type=context["spread_type"],
            credit=context["credit"],
            max_loss=context["max_loss"],
            dte=context["dte"],
            aggressive_contracts=aggressive.recommended_contracts,
            aggressive_rationale=aggressive.rationale,
            aggressive_arguments=", ".join(aggressive.key_arguments[:2]),
            aggressive_conviction=aggressive.conviction,
            neutral_contracts=neutral.recommended_contracts,
            neutral_rationale=neutral.rationale,
            neutral_arguments=", ".join(neutral.key_arguments[:2]),
            neutral_conviction=neutral.conviction,
            conservative_contracts=conservative.recommended_contracts,
            conservative_rationale=conservative.rationale,
            conservative_arguments=", ".join(conservative.key_arguments[:2]),
            conservative_conviction=conservative.conviction,
            vix=context["vix"],
            portfolio_heat=context["portfolio_heat"],
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            RISK_FACILITATOR_SYSTEM,
        )

        data = self.claude._parse_json_response(response)

        # Apply hard constraints
        final_contracts = data.get("final_contracts", neutral.recommended_contracts)

        # Never exceed aggressive recommendation
        final_contracts = min(final_contracts, aggressive.recommended_contracts)

        # Never go below conservative (unless conservative is 0)
        if conservative.recommended_contracts > 0:
            final_contracts = max(final_contracts, conservative.recommended_contracts)

        data["final_contracts"] = final_contracts

        return data

    def _calculate_portfolio_heat(
        self,
        positions: list[Position],
        account_equity: float,
    ) -> float:
        """Calculate portfolio heat as percentage of equity."""
        if not positions or account_equity <= 0:
            return 0.0

        total_risk = sum(
            getattr(p, "max_loss", 0) * getattr(p, "contracts", 1)
            for p in positions
        )

        return total_risk / account_equity

    def _calculate_dte(self, expiration: str) -> int:
        """Calculate days to expiration."""
        exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        return (exp_date - datetime.now()).days
