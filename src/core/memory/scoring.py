"""Memory scoring using FinMem exponential decay.

Implements composite scoring: gamma = S_Recency + S_Relevancy + S_Importance

Reference: FinMem paper (https://arxiv.org/abs/2311.13743)
"""

from __future__ import annotations

import math
from datetime import datetime

from core.memory.types import (
    CRITICAL_EVENT_BONUS,
    LAYER_STABILITY_CONSTANTS,
    MemoryLayer,
    MemoryScores,
)


class MemoryScorer:
    """Calculates composite memory scores using FinMem methodology.

    The composite score (gamma) determines memory retrieval priority:
    gamma = S_Recency + S_Relevancy + S_Importance

    Where:
    - S_Recency = exp(-delta / Q) with layer-specific Q
    - S_Relevancy = vector similarity score (0-1, passed from Vectorize)
    - S_Importance = normalized P&L + critical event bonus (+5)
    """

    def __init__(
        self,
        max_pnl_for_normalization: float = 20.0,
        importance_weight: float = 1.0,
    ):
        """Initialize the scorer.

        Args:
            max_pnl_for_normalization: P&L % to use as max for normalization
            importance_weight: Weight multiplier for importance score
        """
        self.max_pnl_for_normalization = max_pnl_for_normalization
        self.importance_weight = importance_weight

    def calculate_recency_score(
        self,
        entry_date: str | datetime,
        layer: MemoryLayer | str,
        as_of: datetime | None = None,
    ) -> float:
        """Calculate recency score using exponential decay.

        S_Recency = exp(-delta / Q)

        Where:
        - delta = days since entry
        - Q = layer-specific stability constant

        Args:
            entry_date: Date the memory was created
            layer: Memory layer (shallow/intermediate/deep)
            as_of: Reference date for calculation (default: now)

        Returns:
            Recency score between 0 and 1
        """
        if as_of is None:
            as_of = datetime.now()

        # Parse entry_date if string
        if isinstance(entry_date, str):
            entry_date = datetime.fromisoformat(entry_date)

        # Parse layer if string
        if isinstance(layer, str):
            layer = MemoryLayer(layer)

        # Calculate days since entry
        delta = (as_of - entry_date).days
        if delta < 0:
            delta = 0

        # Get stability constant Q for this layer
        q = LAYER_STABILITY_CONSTANTS[layer]

        # Exponential decay: exp(-delta / Q)
        return math.exp(-delta / q)

    def calculate_importance_score(
        self,
        pnl_percent: float | None,
        is_critical: bool,
    ) -> float:
        """Calculate importance score from P&L and critical status.

        S_Importance = normalized_pnl + critical_bonus

        Args:
            pnl_percent: P&L percentage (can be positive or negative)
            is_critical: Whether this is a critical event

        Returns:
            Importance score (unbounded, typically 0-6)
        """
        score = 0.0

        # Add normalized P&L contribution (capped to [0, 1])
        if pnl_percent is not None:
            # Use absolute value - both big wins and big losses are important
            normalized = min(abs(pnl_percent) / self.max_pnl_for_normalization, 1.0)
            score += normalized * self.importance_weight

        # Add critical event bonus
        if is_critical:
            score += CRITICAL_EVENT_BONUS

        return score

    def calculate_composite_score(
        self,
        recency: float,
        relevancy: float,
        importance: float,
    ) -> MemoryScores:
        """Calculate composite score gamma.

        gamma = S_Recency + S_Relevancy + S_Importance

        Args:
            recency: Recency score (0-1)
            relevancy: Relevancy/similarity score (0-1)
            importance: Importance score (0+)

        Returns:
            MemoryScores with all components and composite
        """
        composite = recency + relevancy + importance

        return MemoryScores(
            recency=recency,
            relevancy=relevancy,
            importance=importance,
            composite=composite,
        )

    def score_memory(
        self,
        entry_date: str | datetime,
        layer: MemoryLayer | str,
        similarity_score: float,
        pnl_percent: float | None,
        is_critical: bool,
        as_of: datetime | None = None,
    ) -> MemoryScores:
        """Calculate complete composite score for a memory.

        Convenience method that calculates all components.

        Args:
            entry_date: Date the memory was created
            layer: Memory layer
            similarity_score: Vector similarity from Vectorize (0-1)
            pnl_percent: P&L percentage
            is_critical: Whether this is a critical event
            as_of: Reference date for recency calculation

        Returns:
            MemoryScores with all components
        """
        recency = self.calculate_recency_score(entry_date, layer, as_of)
        importance = self.calculate_importance_score(pnl_percent, is_critical)

        return self.calculate_composite_score(
            recency=recency,
            relevancy=similarity_score,
            importance=importance,
        )
