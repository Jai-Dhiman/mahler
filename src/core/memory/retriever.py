"""Memory retriever combining episodic and semantic memory.

The retriever provides a unified interface for retrieving relevant
context from both episodic (past trades) and semantic (rules) memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.vectorize import EpisodicMemoryStore


@dataclass
class SemanticRule:
    """A validated trading rule from semantic memory."""

    id: str
    rule_text: str
    rule_type: str  # "entry", "exit", "sizing", "regime"
    source: str  # "initial", "learned", "validated"
    applies_to_agent: str  # "all" or specific agent_id
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
            supporting_trades=data.get("supporting_trades", 0),
            opposing_trades=data.get("opposing_trades", 0),
            p_value=data.get("p_value"),
            effect_size=data.get("effect_size"),
            conditions=data.get("conditions"),
            is_active=data.get("is_active", True),
        )

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
    """

    def __init__(self, d1_binding: Any, episodic_store: EpisodicMemoryStore | None = None):
        """Initialize the memory retriever.

        Args:
            d1_binding: D1 database binding
            episodic_store: Optional episodic memory store for similar trades
        """
        self.db = d1_binding
        self.episodic_store = episodic_store

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
        # Get similar trades from episodic memory
        similar_trades = []
        trade_lessons = []

        if self.episodic_store:
            similar_results = await self.episodic_store.find_similar(
                underlying=underlying,
                spread_type=spread_type,
                market_regime=market_regime,
                iv_rank=iv_rank,
                vix_at_entry=vix,
                top_k=5,
            )

            for result in similar_results:
                memory = result.memory
                trade_summary = {
                    "underlying": memory.underlying,
                    "spread_type": memory.spread_type,
                    "entry_date": memory.entry_date,
                    "market_regime": memory.market_regime,
                    "iv_rank": memory.iv_rank,
                    "similarity_score": result.similarity_score,
                    "match_reasons": result.match_reasons,
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

        rules = []
        for row in rows.results:
            rules.append(SemanticRule(
                id=row["id"],
                rule_text=row["rule_text"],
                rule_type=row["rule_type"],
                source=row["source"],
                applies_to_agent=row.get("applies_to_agent", "all"),
                supporting_trades=row.get("supporting_trades", 0),
                opposing_trades=row.get("opposing_trades", 0),
                p_value=row.get("p_value"),
                effect_size=row.get("effect_size"),
                conditions=json.loads(row.get("conditions")) if row.get("conditions") else None,
                is_active=bool(row.get("is_active", 1)),
            ))

        return rules

    async def add_rule(
        self,
        rule_text: str,
        rule_type: str,
        source: str = "learned",
        applies_to_agent: str = "all",
        conditions: dict | None = None,
    ) -> str:
        """Add a new rule to semantic memory.

        Args:
            rule_text: The rule text
            rule_type: Type of rule (entry, exit, sizing, regime)
            source: Source of the rule (initial, learned, validated)
            applies_to_agent: Which agent(s) this applies to
            conditions: Optional conditions for rule application

        Returns:
            ID of the created rule
        """
        import uuid
        rule_id = str(uuid.uuid4())

        await self.db.prepare("""
            INSERT INTO semantic_rules (
                id, rule_text, rule_type, source, applies_to_agent, conditions
            ) VALUES (?, ?, ?, ?, ?, ?)
        """).bind(
            rule_id,
            rule_text,
            rule_type,
            source,
            applies_to_agent,
            json.dumps(conditions) if conditions else None,
        ).run()

        return rule_id

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
