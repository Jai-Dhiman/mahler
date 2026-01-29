"""Memory consolidation for three-tier episodic memory.

Implements memory promotion between layers based on access patterns
and significance, following FinMem methodology.

Reference: FinMem paper (https://arxiv.org/abs/2311.13743)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from core.memory.types import (
    MemoryLayer,
    PromotionThresholds,
)

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Handles memory consolidation and promotion between layers.

    Promotion rules:
    - shallow -> intermediate: access >= 3 + significant P&L, or critical event
    - intermediate -> deep: access >= 5

    Memories are promoted during access or via periodic sweep.
    """

    def __init__(
        self,
        d1_binding: Any,
        thresholds: PromotionThresholds | None = None,
    ):
        """Initialize the consolidator.

        Args:
            d1_binding: D1 database binding
            thresholds: Custom promotion thresholds
        """
        self.db = d1_binding
        self.thresholds = thresholds or PromotionThresholds()

    async def check_and_promote(self, memory_id: str) -> MemoryLayer | None:
        """Check if a memory should be promoted and execute promotion.

        Args:
            memory_id: ID of the memory to check

        Returns:
            New layer if promoted, None if no promotion
        """
        # Fetch current memory state
        row = await self.db.prepare("""
            SELECT memory_layer, access_count, critical_event, pnl_percent
            FROM episodic_memory
            WHERE id = ?
        """).bind(memory_id).first()

        if not row:
            logger.warning(f"Memory not found for promotion check: {memory_id}")
            return None

        current_layer = MemoryLayer(row["memory_layer"])
        access_count = row["access_count"] or 0
        is_critical = bool(row["critical_event"])
        pnl_percent = row["pnl_percent"]

        # Check promotion based on current layer
        new_layer = self._check_promotion(
            current_layer=current_layer,
            access_count=access_count,
            pnl_percent=pnl_percent,
            is_critical=is_critical,
        )

        if new_layer and new_layer != current_layer:
            await self._execute_promotion(memory_id, current_layer, new_layer)
            return new_layer

        return None

    def _check_promotion(
        self,
        current_layer: MemoryLayer,
        access_count: int,
        pnl_percent: float | None,
        is_critical: bool,
    ) -> MemoryLayer | None:
        """Determine if and where a memory should be promoted.

        Args:
            current_layer: Current memory layer
            access_count: Number of accesses
            pnl_percent: P&L percentage
            is_critical: Whether it's a critical event

        Returns:
            Target layer if promotion warranted, None otherwise
        """
        if current_layer == MemoryLayer.SHALLOW:
            if self.thresholds.should_promote_to_intermediate(
                access_count=access_count,
                pnl_percent=pnl_percent,
                is_critical=is_critical,
            ):
                return MemoryLayer.INTERMEDIATE

        elif current_layer == MemoryLayer.INTERMEDIATE:
            if self.thresholds.should_promote_to_deep(access_count):
                return MemoryLayer.DEEP

        # DEEP is the final layer, no further promotion
        return None

    async def _execute_promotion(
        self,
        memory_id: str,
        from_layer: MemoryLayer,
        to_layer: MemoryLayer,
    ) -> None:
        """Execute a memory promotion.

        Args:
            memory_id: ID of memory to promote
            from_layer: Current layer
            to_layer: Target layer
        """
        now = datetime.now().isoformat()

        await self.db.prepare("""
            UPDATE episodic_memory
            SET memory_layer = ?,
                promoted_at = ?,
                promoted_from = ?,
                updated_at = datetime('now')
            WHERE id = ?
        """).bind(
            to_layer.value,
            now,
            from_layer.value,
            memory_id,
        ).run()

        logger.info(
            f"Promoted memory {memory_id} from {from_layer.value} to {to_layer.value}"
        )

    async def run_consolidation_sweep(
        self,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Run a consolidation sweep across all memories.

        Checks all memories for potential promotion. This should be
        run periodically (e.g., nightly) to consolidate memories.

        Args:
            batch_size: Number of memories to process per batch

        Returns:
            Dictionary with promotion counts per transition type
        """
        promotions = {
            "shallow_to_intermediate": 0,
            "intermediate_to_deep": 0,
        }

        # Get all non-deep memories that might need promotion
        offset = 0
        while True:
            rows = await self.db.prepare("""
                SELECT id, memory_layer, access_count, critical_event, pnl_percent
                FROM episodic_memory
                WHERE memory_layer != 'deep'
                ORDER BY access_count DESC
                LIMIT ? OFFSET ?
            """).bind(batch_size, offset).all()

            if not rows.results:
                break

            for row in rows.results:
                current_layer = MemoryLayer(row["memory_layer"])
                new_layer = self._check_promotion(
                    current_layer=current_layer,
                    access_count=row["access_count"] or 0,
                    pnl_percent=row["pnl_percent"],
                    is_critical=bool(row["critical_event"]),
                )

                if new_layer:
                    await self._execute_promotion(
                        memory_id=row["id"],
                        from_layer=current_layer,
                        to_layer=new_layer,
                    )

                    if new_layer == MemoryLayer.INTERMEDIATE:
                        promotions["shallow_to_intermediate"] += 1
                    elif new_layer == MemoryLayer.DEEP:
                        promotions["intermediate_to_deep"] += 1

            offset += batch_size

        logger.info(f"Consolidation sweep completed: {promotions}")
        return promotions

    async def increment_access_count(self, memory_id: str) -> MemoryLayer | None:
        """Increment access count and check for promotion.

        Called each time a memory is retrieved.

        Args:
            memory_id: ID of the accessed memory

        Returns:
            New layer if promoted, None otherwise
        """
        now = datetime.now().isoformat()

        # Increment access count and update last accessed time
        await self.db.prepare("""
            UPDATE episodic_memory
            SET access_count = access_count + 1,
                last_accessed_at = ?,
                updated_at = datetime('now')
            WHERE id = ?
        """).bind(now, memory_id).run()

        # Check if this triggers a promotion
        return await self.check_and_promote(memory_id)
