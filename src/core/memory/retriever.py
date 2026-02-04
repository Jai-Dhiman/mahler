"""Memory retriever combining episodic and semantic memory.

The retriever provides a unified interface for retrieving relevant
context from both episodic (past trades) and semantic (rules) memory.

Implements FinMem enhancements:
- Composite scoring (gamma = S_Recency + S_Relevancy + S_Importance)
- Adaptive cognitive span based on VIX
- Memory access tracking for consolidation

Reference: FinMem paper (https://arxiv.org/abs/2311.13743)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.vectorize import EpisodicMemoryStore, SimilarTradeResult

from core.db.d1 import js_to_python
from core.memory.consolidation import MemoryConsolidator
from core.memory.scoring import MemoryScorer
from core.memory.types import CognitiveSpanConfig

logger = logging.getLogger(__name__)


@dataclass
class SemanticRule:
    """A validated trading rule from semantic memory."""

    id: str
    rule_text: str
    rule_type: str  # "entry", "exit", "sizing", "regime"
    source: str  # "initial", "learned", "validated"
    applies_to_agent: str  # "all" or specific agent_id
    target_agents: str  # Comma-separated list or "all" for selective propagation
    supporting_trades: int
    opposing_trades: int
    p_value: float | None
    effect_size: float | None
    conditions: dict | None
    is_active: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "rule_text": self.rule_text,
            "rule_type": self.rule_type,
            "source": self.source,
            "applies_to_agent": self.applies_to_agent,
            "target_agents": self.target_agents,
            "supporting_trades": self.supporting_trades,
            "opposing_trades": self.opposing_trades,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "conditions": self.conditions,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticRule:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            rule_text=data["rule_text"],
            rule_type=data["rule_type"],
            source=data["source"],
            applies_to_agent=data.get("applies_to_agent", "all"),
            target_agents=data.get("target_agents", "all"),
            supporting_trades=data.get("supporting_trades", 0),
            opposing_trades=data.get("opposing_trades", 0),
            p_value=data.get("p_value"),
            effect_size=data.get("effect_size"),
            conditions=data.get("conditions"),
            is_active=data.get("is_active", True),
        )

    def applies_to_target(self, agent_id: str) -> bool:
        """Check if this rule targets a specific agent.

        Used for selective knowledge propagation per FINCON paper:
        insights flow hierarchically and only to relevant agents.
        """
        if self.target_agents == "all":
            return True

        # Parse comma-separated list
        target_list = [t.strip().lower() for t in self.target_agents.split(",")]
        return agent_id.lower() in target_list

    def applies_to(self, regime: str | None = None, iv_rank: float | None = None) -> bool:
        """Check if this rule applies given current conditions."""
        if not self.conditions:
            return True

        # Check regime condition
        if "regime" in self.conditions:
            allowed_regimes = self.conditions["regime"]
            if isinstance(allowed_regimes, list) and regime not in allowed_regimes:
                return False
            elif isinstance(allowed_regimes, str) and regime != allowed_regimes:
                return False

        # Check IV rank condition
        if "iv_rank_min" in self.conditions and iv_rank is not None:
            if iv_rank < self.conditions["iv_rank_min"]:
                return False

        if "iv_rank_max" in self.conditions and iv_rank is not None:
            if iv_rank > self.conditions["iv_rank_max"]:
                return False

        return True


@dataclass
class RetrievedContext:
    """Context retrieved from memory for agent use."""

    # Similar past trades
    similar_trades: list[dict]
    trade_lessons: list[str]

    # Applicable rules
    entry_rules: list[SemanticRule]
    exit_rules: list[SemanticRule]
    sizing_rules: list[SemanticRule]
    regime_rules: list[SemanticRule]

    # Summary for prompts
    context_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "similar_trades": self.similar_trades,
            "trade_lessons": self.trade_lessons,
            "entry_rules": [r.to_dict() for r in self.entry_rules],
            "exit_rules": [r.to_dict() for r in self.exit_rules],
            "sizing_rules": [r.to_dict() for r in self.sizing_rules],
            "regime_rules": [r.to_dict() for r in self.regime_rules],
            "context_summary": self.context_summary,
        }


class MemoryRetriever:
    """Retrieves relevant context from episodic and semantic memory.

    Combines:
    - Episodic memory: Similar past trades via vector search
    - Semantic memory: Validated rules from D1

    Provides unified context for agent decision-making.

    FinMem enhancements:
    - Uses composite scoring (gamma) to rank memories
    - Adapts cognitive span (K) based on VIX volatility
    - Tracks memory access for consolidation/promotion
    """

    def __init__(self, d1_binding: Any, episodic_store: EpisodicMemoryStore | None = None):
        """Initialize the memory retriever.

        Args:
            d1_binding: D1 database binding
            episodic_store: Optional episodic memory store for similar trades
        """
        self.db = d1_binding
        self.episodic_store = episodic_store

        # FinMem components
        self.scorer = MemoryScorer()
        self.consolidator = MemoryConsolidator(d1_binding)
        self.cognitive_config = CognitiveSpanConfig()

    async def retrieve_context(
        self,
        underlying: str,
        spread_type: str,
        market_regime: str | None = None,
        iv_rank: float | None = None,
        vix: float | None = None,
        agent_id: str | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant context for a trade decision.

        Uses FinMem methodology:
        1. Determine cognitive span K based on VIX
        2. Over-fetch from episodic store (K * 2)
        3. Apply composite scoring to results
        4. Sort by composite score, take top K
        5. Track access for memory consolidation

        Args:
            underlying: The underlying symbol
            spread_type: Type of spread
            market_regime: Current market regime
            iv_rank: Current IV rank
            vix: Current VIX level
            agent_id: Optional agent to filter rules for

        Returns:
            RetrievedContext with similar trades and applicable rules
        """
        # Get cognitive span K based on VIX
        k = self._get_cognitive_span(vix)
        logger.debug(f"Cognitive span K={k} for VIX={vix}")

        # Get similar trades from episodic memory
        similar_trades = []
        trade_lessons = []

        if self.episodic_store:
            try:
                # Over-fetch to allow for composite scoring and re-ranking
                # Use metadata filtering for market_regime and spread_type
                similar_results = await self.episodic_store.find_similar(
                    underlying=underlying,
                    spread_type=spread_type,
                    market_regime=market_regime,
                    iv_rank=iv_rank,
                    vix_at_entry=vix,
                    top_k=k * 2,  # Over-fetch for re-ranking
                    filter_spread_type=True,  # Use Vectorize metadata filter
                )
            except Exception as e:
                # Gracefully handle episodic memory errors - pipeline can run without it
                logger.warning(f"Episodic memory retrieval failed: {e}, continuing without similar trades")
                similar_results = []

            # Apply composite scoring and re-rank
            scored_results = await self._score_and_rank_results(similar_results)

            # Take top K after scoring
            for result, scores in scored_results[:k]:
                memory = result.memory

                # Track memory access for consolidation
                await self._on_memory_accessed(memory.id)

                trade_summary = {
                    "underlying": memory.underlying,
                    "spread_type": memory.spread_type,
                    "entry_date": memory.entry_date,
                    "market_regime": memory.market_regime,
                    "iv_rank": memory.iv_rank,
                    "similarity_score": result.similarity_score,
                    "match_reasons": result.match_reasons,
                    # FinMem scores
                    "composite_score": scores.composite,
                    "recency_score": scores.recency,
                    "importance_score": scores.importance,
                    "memory_layer": memory.memory_layer,
                    "is_critical": memory.critical_event,
                }

                # Add outcome if available
                if memory.actual_outcome:
                    trade_summary["outcome"] = memory.actual_outcome
                    trade_summary["was_profitable"] = memory.actual_outcome.get("profit_loss", 0) > 0

                similar_trades.append(trade_summary)

                # Extract lessons
                if memory.lesson_extracted:
                    trade_lessons.append(memory.lesson_extracted)

        # Get applicable rules from semantic memory
        all_rules = await self._get_active_rules(agent_id)

        # Filter rules by conditions
        entry_rules = []
        exit_rules = []
        sizing_rules = []
        regime_rules = []

        for rule in all_rules:
            if not rule.applies_to(regime=market_regime, iv_rank=iv_rank):
                continue

            if rule.rule_type == "entry":
                entry_rules.append(rule)
            elif rule.rule_type == "exit":
                exit_rules.append(rule)
            elif rule.rule_type == "sizing":
                sizing_rules.append(rule)
            elif rule.rule_type == "regime":
                regime_rules.append(rule)

        # Generate context summary
        context_summary = self._generate_summary(
            similar_trades=similar_trades,
            trade_lessons=trade_lessons,
            entry_rules=entry_rules,
            market_regime=market_regime,
        )

        return RetrievedContext(
            similar_trades=similar_trades,
            trade_lessons=trade_lessons,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            sizing_rules=sizing_rules,
            regime_rules=regime_rules,
            context_summary=context_summary,
        )

    async def _get_active_rules(self, agent_id: str | None = None) -> list[SemanticRule]:
        """Get all active rules from semantic memory."""
        if agent_id:
            rows = await self.db.prepare("""
                SELECT * FROM semantic_rules
                WHERE is_active = 1
                  AND (applies_to_agent = 'all' OR applies_to_agent = ?)
                ORDER BY supporting_trades DESC
            """).bind(agent_id).all()
        else:
            rows = await self.db.prepare("""
                SELECT * FROM semantic_rules
                WHERE is_active = 1
                ORDER BY supporting_trades DESC
            """).all()

        # Convert JsProxy results to Python dicts
        results = js_to_python(rows)
        result_rows = results.get("results", []) if isinstance(results, dict) else []

        rules = []
        for row in result_rows:
            rules.append(SemanticRule(
                id=row["id"],
                rule_text=row["rule_text"],
                rule_type=row["rule_type"],
                source=row["source"],
                applies_to_agent=row.get("applies_to_agent", "all"),
                target_agents=row.get("target_agents", "all"),
                supporting_trades=row.get("supporting_trades", 0),
                opposing_trades=row.get("opposing_trades", 0),
                p_value=row.get("p_value"),
                effect_size=row.get("effect_size"),
                conditions=json.loads(row.get("conditions")) if row.get("conditions") else None,
                is_active=bool(row.get("is_active", 1)),
            ))

        return rules

    async def get_rules_for_agent(
        self,
        agent_id: str,
        rule_type: str | None = None,
        market_regime: str | None = None,
        iv_rank: float | None = None,
    ) -> list[SemanticRule]:
        """Get rules applicable to a specific agent with filtering.

        This implements selective knowledge propagation from FINCON paper:
        "Selectively propagates insights back to relevant agents
        rather than broadcasting system-wide."

        Args:
            agent_id: ID of the agent requesting rules
            rule_type: Optional filter by rule type
            market_regime: Current market regime for condition filtering
            iv_rank: Current IV rank for condition filtering

        Returns:
            List of SemanticRule objects applicable to this agent
        """
        # Build query with target_agents filtering
        query = """
            SELECT * FROM semantic_rules
            WHERE is_active = 1
              AND (target_agents = 'all' OR target_agents LIKE ?)
        """
        params = [f"%{agent_id}%"]

        if rule_type:
            query += " AND rule_type = ?"
            params.append(rule_type)

        query += " ORDER BY supporting_trades DESC"

        rows = await self.db.prepare(query).bind(*params).all()

        rules = []
        for row in rows.results:
            rule = SemanticRule(
                id=row["id"],
                rule_text=row["rule_text"],
                rule_type=row["rule_type"],
                source=row["source"],
                applies_to_agent=row.get("applies_to_agent", "all"),
                target_agents=row.get("target_agents", "all"),
                supporting_trades=row.get("supporting_trades", 0),
                opposing_trades=row.get("opposing_trades", 0),
                p_value=row.get("p_value"),
                effect_size=row.get("effect_size"),
                conditions=json.loads(row.get("conditions")) if row.get("conditions") else None,
                is_active=bool(row.get("is_active", 1)),
            )

            # Double-check target agent matching (for precise filtering)
            if not rule.applies_to_target(agent_id):
                continue

            # Check condition filtering
            if not rule.applies_to(regime=market_regime, iv_rank=iv_rank):
                continue

            rules.append(rule)

        return rules

    async def add_rule(
        self,
        rule_text: str,
        rule_type: str,
        source: str = "learned",
        applies_to_agent: str = "all",
        target_agents: str = "all",
        conditions: dict | None = None,
    ) -> str:
        """Add a new rule to semantic memory.

        Args:
            rule_text: The rule text
            rule_type: Type of rule (entry, exit, sizing, regime)
            source: Source of the rule (initial, learned, validated)
            applies_to_agent: Which agent(s) this applies to (legacy field)
            target_agents: Comma-separated list of target agents for propagation
            conditions: Optional conditions for rule application

        Returns:
            ID of the created rule
        """
        import uuid
        rule_id = str(uuid.uuid4())

        await self.db.prepare("""
            INSERT INTO semantic_rules (
                id, rule_text, rule_type, source, applies_to_agent, target_agents, conditions
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """).bind(
            rule_id,
            rule_text,
            rule_type,
            source,
            applies_to_agent,
            target_agents,
            json.dumps(conditions) if conditions else None,
        ).run()

        return rule_id

    async def update_rule_target_agents(
        self,
        rule_id: str,
        target_agents: str,
    ) -> None:
        """Update the target agents for a rule.

        Used for selective knowledge propagation per FINCON paper.

        Args:
            rule_id: ID of the rule
            target_agents: Comma-separated list or "all"
        """
        await self.db.prepare("""
            UPDATE semantic_rules
            SET target_agents = ?,
                updated_at = datetime('now')
            WHERE id = ?
        """).bind(target_agents, rule_id).run()

    async def update_rule_stats(
        self,
        rule_id: str,
        supported: bool,
    ) -> None:
        """Update rule statistics after a trade.

        Args:
            rule_id: ID of the rule
            supported: Whether the trade outcome supported the rule
        """
        if supported:
            await self.db.prepare("""
                UPDATE semantic_rules
                SET supporting_trades = supporting_trades + 1,
                    updated_at = datetime('now')
                WHERE id = ?
            """).bind(rule_id).run()
        else:
            await self.db.prepare("""
                UPDATE semantic_rules
                SET opposing_trades = opposing_trades + 1,
                    updated_at = datetime('now')
                WHERE id = ?
            """).bind(rule_id).run()

    async def validate_rule(
        self,
        rule_id: str,
        p_value: float,
        effect_size: float,
    ) -> None:
        """Update rule with statistical validation results.

        Args:
            rule_id: ID of the rule
            p_value: Statistical significance
            effect_size: Practical significance
        """
        # Mark as validated if statistically significant
        source = "validated" if p_value < 0.05 else "learned"

        await self.db.prepare("""
            UPDATE semantic_rules
            SET source = ?,
                p_value = ?,
                effect_size = ?,
                last_validated = datetime('now'),
                validation_count = validation_count + 1,
                updated_at = datetime('now')
            WHERE id = ?
        """).bind(source, p_value, effect_size, rule_id).run()

    def _generate_summary(
        self,
        similar_trades: list[dict],
        trade_lessons: list[str],
        entry_rules: list[SemanticRule],
        market_regime: str | None,
    ) -> str:
        """Generate a text summary of retrieved context."""
        parts = []

        # Similar trades summary
        if similar_trades:
            win_count = sum(1 for t in similar_trades if t.get("was_profitable", False))
            total = len(similar_trades)
            parts.append(f"Found {total} similar past trades ({win_count} profitable)")

            # Top lesson
            if trade_lessons:
                parts.append(f"Key lesson: {trade_lessons[0]}")

        # Rules summary
        if entry_rules:
            validated_count = sum(1 for r in entry_rules if r.source == "validated")
            parts.append(f"{len(entry_rules)} applicable entry rules ({validated_count} validated)")

        # Regime context
        if market_regime:
            parts.append(f"Current regime: {market_regime.replace('_', ' ')}")

        return "; ".join(parts) if parts else "No relevant context found"

    # FinMem helper methods

    def _get_cognitive_span(self, vix: float | None) -> int:
        """Get cognitive span K based on VIX level.

        From FinMem: adjusts how many memories to retrieve based on
        market volatility:
        - High VIX (>25): More context needed, K=10
        - Normal VIX (15-25): Default K=5
        - Low VIX (<15): Less noise, K=3

        Args:
            vix: Current VIX level

        Returns:
            Number of memories to retrieve (K)
        """
        return self.cognitive_config.get_k(vix)

    async def _score_and_rank_results(
        self,
        results: list[SimilarTradeResult],
    ) -> list[tuple[SimilarTradeResult, Any]]:
        """Apply composite scoring and rank results.

        gamma = S_Recency + S_Relevancy + S_Importance

        Args:
            results: Similar trade results from vector search

        Returns:
            List of (result, scores) tuples sorted by composite score
        """
        from core.memory.types import MemoryScores

        scored = []
        for result in results:
            memory = result.memory

            scores = self.scorer.score_memory(
                entry_date=memory.entry_date,
                layer=memory.memory_layer,
                similarity_score=result.similarity_score,
                pnl_percent=memory.pnl_percent,
                is_critical=memory.critical_event,
            )

            scored.append((result, scores))

        # Sort by composite score descending
        scored.sort(key=lambda x: x[1].composite, reverse=True)

        return scored

    async def _on_memory_accessed(self, memory_id: str) -> None:
        """Handle memory access event.

        Increments access count and checks for promotion.

        Args:
            memory_id: ID of the accessed memory
        """
        new_layer = await self.consolidator.increment_access_count(memory_id)
        if new_layer:
            logger.info(f"Memory {memory_id} promoted to {new_layer.value}")

    def _recency_score(self, memory: Any) -> float:
        """Calculate recency score for a memory.

        Uses MemoryScorer with layer-specific decay.

        Args:
            memory: EpisodicMemoryRecord

        Returns:
            Recency score between 0 and 1
        """
        return self.scorer.calculate_recency_score(
            entry_date=memory.entry_date,
            layer=memory.memory_layer,
        )

    def _importance_score(self, memory: Any) -> float:
        """Calculate importance score for a memory.

        Args:
            memory: EpisodicMemoryRecord

        Returns:
            Importance score (0+)
        """
        return self.scorer.calculate_importance_score(
            pnl_percent=memory.pnl_percent,
            is_critical=memory.critical_event,
        )
