"""Three-Perspective Risk Manager for V2 multi-agent system.

Provides VIX-weighted risk assessment from three perspectives:
- Aggressive: Uses full 2% per-trade limit
- Neutral: Uses 75% of limits
- Conservative: Uses 50% of limits

The weighted recommendation varies by market conditions:
- VIX > 30: Conservative-heavy weighting (0.1, 0.3, 0.6)
- VIX > 20: Neutral-heavy weighting (0.2, 0.5, 0.3)
- VIX <= 20: Balanced weighting (0.3, 0.5, 0.2)

This approach provides more nuanced position sizing than a single
perspective, helping avoid overconfidence in calm markets and
excessive caution in volatile markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
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
    high_vix_weights: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.6]
    )  # Conservative-heavy
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
