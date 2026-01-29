"""CVRF (Conceptual Verbal Reinforcement Factor) Learning Rate Implementation.

Inspired by the FINCON paper (arXiv:2407.06567):
"tau = overlap(decisions_k, decisions_k-1) - percentage overlap between
consecutive trading decision sequences"

The CVRF tau value is used to weight rule updates in the reflection engine:
- High tau (decisions similar) = stable beliefs, small updates
- Low tau (decisions different) = unstable beliefs, larger updates

This creates an adaptive learning rate that responds to decision stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.agents.decision import TradingDecision


@dataclass
class TradeDecisionRecord:
    """Simplified trade decision record for overlap calculation."""

    underlying: str
    action: str  # "enter" or "skip"
    confidence: float
    position_size: int
    timestamp: datetime

    @classmethod
    def from_trading_decision(
        cls,
        decision: TradingDecision,
        underlying: str,
        timestamp: datetime | None = None,
    ) -> TradeDecisionRecord:
        """Create from a TradingDecision object."""
        return cls(
            underlying=underlying,
            action=decision.decision,
            confidence=decision.confidence,
            position_size=decision.position_size,
            timestamp=timestamp or datetime.now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "underlying": self.underlying,
            "action": self.action,
            "confidence": self.confidence,
            "position_size": self.position_size,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeDecisionRecord:
        """Deserialize from dictionary."""
        return cls(
            underlying=data["underlying"],
            action=data["action"],
            confidence=data["confidence"],
            position_size=data["position_size"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class CVRFResult:
    """Result of CVRF tau calculation."""

    tau: float  # 0.0 to 1.0
    overlap_count: int
    total_count: int
    stability: str  # "stable", "moderate", "unstable"
    learning_rate_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tau": self.tau,
            "overlap_count": self.overlap_count,
            "total_count": self.total_count,
            "stability": self.stability,
            "learning_rate_multiplier": self.learning_rate_multiplier,
        }


def calculate_cvrf_tau(
    decisions_prev: list[TradeDecisionRecord],
    decisions_current: list[TradeDecisionRecord],
    confidence_tolerance: float = 0.1,
) -> CVRFResult:
    """Calculate CVRF learning rate as overlap percentage.

    From FINCON paper:
    tau = |intersection(prev, current)| / |union(prev, current)|

    Overlap considers:
    - Same underlying
    - Same action (enter/skip)
    - Similar confidence (+/- tolerance)

    Args:
        decisions_prev: Previous period's decisions
        decisions_current: Current period's decisions
        confidence_tolerance: Tolerance for confidence comparison (default 0.1)

    Returns:
        CVRFResult with tau value and derived metrics
    """
    if not decisions_prev or not decisions_current:
        return CVRFResult(
            tau=0.5,  # Default to moderate
            overlap_count=0,
            total_count=0,
            stability="moderate",
            learning_rate_multiplier=1.0,
        )

    # Build decision fingerprints for comparison
    prev_fingerprints = _build_fingerprints(decisions_prev)
    curr_fingerprints = _build_fingerprints(decisions_current)

    # Calculate intersection (matching decisions)
    intersection = 0
    for underlying, (action, conf) in curr_fingerprints.items():
        if underlying in prev_fingerprints:
            prev_action, prev_conf = prev_fingerprints[underlying]
            if action == prev_action:
                if abs(conf - prev_conf) <= confidence_tolerance:
                    intersection += 1

    # Calculate union size
    all_underlyings = set(prev_fingerprints.keys()) | set(curr_fingerprints.keys())
    union_size = len(all_underlyings)

    # Calculate tau
    tau = intersection / union_size if union_size > 0 else 0.5

    # Determine stability category and learning rate
    if tau >= 0.7:
        stability = "stable"
        learning_rate_multiplier = 0.5  # Small updates for stable beliefs
    elif tau >= 0.4:
        stability = "moderate"
        learning_rate_multiplier = 1.0  # Normal updates
    else:
        stability = "unstable"
        learning_rate_multiplier = 1.5  # Larger updates for changing beliefs

    return CVRFResult(
        tau=tau,
        overlap_count=intersection,
        total_count=union_size,
        stability=stability,
        learning_rate_multiplier=learning_rate_multiplier,
    )


def _build_fingerprints(
    decisions: list[TradeDecisionRecord],
) -> dict[str, tuple[str, float]]:
    """Build fingerprints for decision comparison.

    For each underlying, take the most recent decision.

    Returns:
        Dict mapping underlying -> (action, confidence)
    """
    fingerprints: dict[str, tuple[str, float]] = {}

    # Sort by timestamp (most recent last)
    sorted_decisions = sorted(decisions, key=lambda d: d.timestamp)

    for decision in sorted_decisions:
        # Most recent decision for each underlying wins
        fingerprints[decision.underlying] = (
            decision.action,
            decision.confidence,
        )

    return fingerprints


def calculate_rolling_tau(
    decision_history: list[list[TradeDecisionRecord]],
    window_size: int = 5,
) -> list[CVRFResult]:
    """Calculate rolling tau values over a history of decision periods.

    Args:
        decision_history: List of decision lists, ordered chronologically
        window_size: Number of periods to include in rolling calculation

    Returns:
        List of CVRFResult for each period (starting from period 2)
    """
    if len(decision_history) < 2:
        return []

    results = []

    for i in range(1, len(decision_history)):
        # Compare current period with previous
        prev_decisions = decision_history[i - 1]
        curr_decisions = decision_history[i]

        result = calculate_cvrf_tau(prev_decisions, curr_decisions)
        results.append(result)

    return results


def get_adaptive_learning_rate(
    base_learning_rate: float,
    tau_result: CVRFResult,
    min_rate: float = 0.1,
    max_rate: float = 2.0,
) -> float:
    """Get adaptive learning rate based on CVRF tau.

    The learning rate is adjusted based on decision stability:
    - Stable (high tau): Reduce learning rate to preserve good strategies
    - Unstable (low tau): Increase learning rate to adapt faster

    Args:
        base_learning_rate: Base learning rate before adjustment
        tau_result: CVRF calculation result
        min_rate: Minimum allowed learning rate
        max_rate: Maximum allowed learning rate

    Returns:
        Adjusted learning rate
    """
    adjusted_rate = base_learning_rate * tau_result.learning_rate_multiplier

    # Clamp to bounds
    return max(min_rate, min(max_rate, adjusted_rate))


class CVRFTracker:
    """Tracks CVRF tau over time for continuous learning.

    Maintains a rolling history of decisions and provides
    tau calculations for adaptive learning rate adjustment.
    """

    def __init__(
        self,
        max_periods: int = 30,
        decisions_per_period: int = 10,
    ):
        """Initialize the CVRF tracker.

        Args:
            max_periods: Maximum number of periods to retain
            decisions_per_period: Expected decisions per period (for grouping)
        """
        self.max_periods = max_periods
        self.decisions_per_period = decisions_per_period
        self._decision_history: list[list[TradeDecisionRecord]] = []
        self._current_period: list[TradeDecisionRecord] = []
        self._latest_tau: CVRFResult | None = None

    def add_decision(self, decision: TradeDecisionRecord) -> None:
        """Add a decision to the current period."""
        self._current_period.append(decision)

        # If period is full, commit it
        if len(self._current_period) >= self.decisions_per_period:
            self.commit_period()

    def commit_period(self) -> CVRFResult | None:
        """Commit current period and calculate tau.

        Returns:
            CVRFResult if there's a previous period to compare with
        """
        if not self._current_period:
            return None

        # Add current period to history
        self._decision_history.append(self._current_period)

        # Trim history if needed
        if len(self._decision_history) > self.max_periods:
            self._decision_history = self._decision_history[-self.max_periods:]

        # Calculate tau if we have at least 2 periods
        result = None
        if len(self._decision_history) >= 2:
            result = calculate_cvrf_tau(
                self._decision_history[-2],
                self._decision_history[-1],
            )
            self._latest_tau = result

        # Reset current period
        self._current_period = []

        return result

    @property
    def latest_tau(self) -> CVRFResult | None:
        """Get the most recent tau calculation."""
        return self._latest_tau

    @property
    def stability_trend(self) -> str:
        """Get overall stability trend based on recent tau values."""
        if len(self._decision_history) < 3:
            return "insufficient_data"

        # Calculate tau for last few periods
        recent_results = calculate_rolling_tau(
            self._decision_history[-5:] if len(self._decision_history) >= 5 else self._decision_history
        )

        if not recent_results:
            return "insufficient_data"

        avg_tau = sum(r.tau for r in recent_results) / len(recent_results)

        if avg_tau >= 0.7:
            return "consistently_stable"
        elif avg_tau >= 0.4:
            return "moderately_stable"
        else:
            return "volatile"

    def get_adaptive_rate(
        self,
        base_rate: float = 1.0,
    ) -> float:
        """Get adaptive learning rate based on current tau."""
        if self._latest_tau is None:
            return base_rate

        return get_adaptive_learning_rate(base_rate, self._latest_tau)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state."""
        return {
            "max_periods": self.max_periods,
            "decisions_per_period": self.decisions_per_period,
            "decision_history": [
                [d.to_dict() for d in period]
                for period in self._decision_history
            ],
            "current_period": [d.to_dict() for d in self._current_period],
            "latest_tau": self._latest_tau.to_dict() if self._latest_tau else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CVRFTracker:
        """Deserialize tracker state."""
        tracker = cls(
            max_periods=data.get("max_periods", 30),
            decisions_per_period=data.get("decisions_per_period", 10),
        )

        tracker._decision_history = [
            [TradeDecisionRecord.from_dict(d) for d in period]
            for period in data.get("decision_history", [])
        ]

        tracker._current_period = [
            TradeDecisionRecord.from_dict(d)
            for d in data.get("current_period", [])
        ]

        if data.get("latest_tau"):
            tracker._latest_tau = CVRFResult(**data["latest_tau"])

        return tracker
