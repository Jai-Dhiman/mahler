"""FinMem memory types and configuration.

Defines types for three-tier episodic memory with exponential decay scoring.

Reference: FinMem paper (https://arxiv.org/abs/2311.13743)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryLayer(Enum):
    """Three-tier memory hierarchy from FinMem.

    Each layer has different stability constants (Q) for decay:
    - SHALLOW: Recent memories, decay quickly (Q=14 days)
    - INTERMEDIATE: Consolidated memories, moderate decay (Q=60 days)
    - DEEP: Long-term memories, slow decay (Q=180 days)
    """

    SHALLOW = "shallow"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"


# Layer-specific stability constants (Q) in days for exponential decay
# S_Recency = exp(-delta / Q) where delta is days since entry
LAYER_STABILITY_CONSTANTS: dict[MemoryLayer, int] = {
    MemoryLayer.SHALLOW: 14,
    MemoryLayer.INTERMEDIATE: 60,
    MemoryLayer.DEEP: 180,
}


@dataclass
class MemoryScores:
    """Composite memory scoring components from FinMem.

    gamma = S_Recency + S_Relevancy + S_Importance
    """

    recency: float  # S_Recency: exponential decay based on time
    relevancy: float  # S_Relevancy: vector similarity score (from Vectorize)
    importance: float  # S_Importance: P&L impact + critical event bonus
    composite: float  # gamma: sum of all scores

    def __post_init__(self) -> None:
        """Validate scores are non-negative."""
        if self.recency < 0 or self.relevancy < 0 or self.importance < 0:
            raise ValueError("All score components must be non-negative")


@dataclass
class CognitiveSpanConfig:
    """Configuration for adaptive cognitive span (K) based on VIX.

    K determines how many memories to retrieve:
    - High volatility (VIX > 25): More context needed, K=10
    - Normal volatility (15-25): Default K=5
    - Low volatility (VIX < 15): Less noise, K=3
    """

    base_k: int = 5
    high_vix_k: int = 10
    low_vix_k: int = 3
    high_vix_threshold: float = 25.0
    low_vix_threshold: float = 15.0

    def get_k(self, vix: float | None) -> int:
        """Get cognitive span K based on current VIX level.

        Args:
            vix: Current VIX level, or None for default

        Returns:
            Number of memories to retrieve (K)
        """
        if vix is None:
            return self.base_k

        if vix > self.high_vix_threshold:
            return self.high_vix_k
        elif vix < self.low_vix_threshold:
            return self.low_vix_k
        else:
            return self.base_k


@dataclass
class PromotionThresholds:
    """Thresholds for memory promotion between layers.

    Promotion rules from FinMem:
    - shallow -> intermediate: access >= 3 + significant P&L
    - intermediate -> deep: access >= 5
    - Critical events fast-track to intermediate
    """

    shallow_to_intermediate_access: int = 3
    intermediate_to_deep_access: int = 5
    significant_pnl_percent: float = 5.0  # Abs P&L % to count as significant

    def should_promote_to_intermediate(
        self,
        access_count: int,
        pnl_percent: float | None,
        is_critical: bool,
    ) -> bool:
        """Check if a shallow memory should promote to intermediate.

        Args:
            access_count: Number of times memory was accessed
            pnl_percent: P&L percentage of the trade
            is_critical: Whether this is a critical event

        Returns:
            True if should promote to intermediate
        """
        # Critical events fast-track to intermediate
        if is_critical:
            return True

        # Regular promotion: access threshold + significant P&L
        if access_count >= self.shallow_to_intermediate_access:
            if pnl_percent is not None:
                return abs(pnl_percent) >= self.significant_pnl_percent
            # Allow promotion even without P&L if frequently accessed
            return access_count >= self.shallow_to_intermediate_access + 2

        return False

    def should_promote_to_deep(self, access_count: int) -> bool:
        """Check if an intermediate memory should promote to deep.

        Args:
            access_count: Number of times memory was accessed

        Returns:
            True if should promote to deep
        """
        return access_count >= self.intermediate_to_deep_access


# Critical event detection thresholds
CRITICAL_EVENT_PNL_MULTIPLIER = 2.0  # P&L > 2x average is critical
CRITICAL_EVENT_BONUS = 5.0  # +5 points for critical events
