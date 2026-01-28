"""Self-reflection engine for continuous learning.

The reflection engine:
1. Generates reflections after trades close
2. Compares predicted vs actual outcomes
3. Extracts candidate rules
4. Validates rules statistically
5. Promotes validated rules to semantic memory
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient
    from core.memory.retriever import MemoryRetriever
    from core.memory.vectorize import EpisodicMemoryStore


@dataclass
class TradeOutcome:
    """Actual outcome of a closed trade."""

    trade_id: str
    entry_date: str
    exit_date: str
    underlying: str
    spread_type: str
    entry_credit: float
    exit_debit: float
    profit_loss: float
    profit_loss_percent: float
    was_profitable: bool
    exit_reason: str  # "profit_target", "stop_loss", "time_exit", "manual"
    days_held: int


@dataclass
class PredictedOutcome:
    """Predicted outcome from the agent pipeline."""

    recommendation: str  # "enter", "skip", "reduce_size"
    confidence: float
    expected_profit_probability: float
    key_bull_points: list[str]
    key_bear_points: list[str]
    thesis: str


@dataclass
class TradeReflection:
    """Generated reflection on a closed trade."""

    trade_id: str
    memory_id: str | None

    # Prediction accuracy
    prediction_correct: bool
    confidence_calibration: str  # "overconfident", "calibrated", "underconfident"

    # Analysis
    what_worked: list[str]
    what_failed: list[str]
    market_behavior: str
    key_lesson: str

    # Rule candidates
    candidate_rules: list[dict]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trade_id": self.trade_id,
            "memory_id": self.memory_id,
            "prediction_correct": self.prediction_correct,
            "confidence_calibration": self.confidence_calibration,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "market_behavior": self.market_behavior,
            "key_lesson": self.key_lesson,
            "candidate_rules": self.candidate_rules,
        }


# Prompt for reflection generation
REFLECTION_SYSTEM = """You are a trading coach analyzing closed trades to extract lessons. Your goal is to identify specific, actionable insights that can improve future trading decisions.

Focus on:
1. Whether the prediction was accurate and why
2. What factors correctly predicted the outcome
3. What factors were misleading or wrong
4. What specific rule could have improved the decision

Be precise and quantitative. Avoid generic advice."""

REFLECTION_USER = """Analyze this closed trade and generate a reflection:

**Trade Details:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Entry Date: {entry_date}
- Exit Date: {exit_date}
- Days Held: {days_held}

**Financials:**
- Entry Credit: ${entry_credit:.2f}
- Exit Debit: ${exit_debit:.2f}
- P/L: ${profit_loss:.2f} ({profit_loss_percent:.1f}%)
- Exit Reason: {exit_reason}

**Original Prediction:**
- Recommendation: {recommendation}
- Confidence: {confidence:.0%}
- Expected Win Probability: {expected_probability:.0%}
- Thesis: {thesis}

**Bull Arguments:** {bull_points}
**Bear Arguments:** {bear_points}

**Market Context at Entry:**
- Regime: {market_regime}
- IV Rank: {iv_rank}%
- VIX: {vix}

Generate a reflection analyzing what happened.

Respond in this exact JSON format:
{{
    "prediction_correct": true|false,
    "confidence_calibration": "overconfident|calibrated|underconfident",
    "what_worked": ["factor 1", "factor 2"],
    "what_failed": ["factor 1", "factor 2"],
    "market_behavior": "Description of what the market did",
    "key_lesson": "One specific, actionable lesson",
    "candidate_rules": [
        {{
            "rule_text": "Specific rule text",
            "rule_type": "entry|exit|sizing|regime",
            "rationale": "Why this rule would help"
        }}
    ]
}}"""


class SelfReflectionEngine:
    """Engine for generating reflections and extracting rules from closed trades.

    The reflection process:
    1. Compare predicted outcome with actual outcome
    2. Generate AI reflection on the trade
    3. Extract candidate rules from the reflection
    4. Track rule performance over time
    5. Validate rules statistically when enough data exists
    6. Promote validated rules to semantic memory
    """

    # Minimum trades to validate a rule
    MIN_TRADES_FOR_VALIDATION = 10

    # P-value threshold for rule validation
    P_VALUE_THRESHOLD = 0.05

    def __init__(
        self,
        claude: ClaudeClient,
        memory_retriever: MemoryRetriever,
        episodic_store: EpisodicMemoryStore | None = None,
    ):
        """Initialize the reflection engine.

        Args:
            claude: Claude client for generating reflections
            memory_retriever: Memory retriever for rule management
            episodic_store: Optional episodic store for updating memories
        """
        self.claude = claude
        self.retriever = memory_retriever
        self.episodic = episodic_store

    async def generate_reflection(
        self,
        outcome: TradeOutcome,
        predicted: PredictedOutcome,
        memory_id: str | None = None,
        market_regime: str | None = None,
        iv_rank: float | None = None,
        vix: float | None = None,
    ) -> TradeReflection:
        """Generate a reflection for a closed trade.

        Args:
            outcome: Actual trade outcome
            predicted: Original prediction from agents
            memory_id: Optional episodic memory ID
            market_regime: Market regime at entry
            iv_rank: IV rank at entry
            vix: VIX at entry

        Returns:
            TradeReflection with analysis and candidate rules
        """
        # Format arguments for prompt
        bull_points = ", ".join(predicted.key_bull_points[:3]) if predicted.key_bull_points else "None recorded"
        bear_points = ", ".join(predicted.key_bear_points[:3]) if predicted.key_bear_points else "None recorded"

        prompt = REFLECTION_USER.format(
            underlying=outcome.underlying,
            spread_type=outcome.spread_type.replace("_", " ").title(),
            entry_date=outcome.entry_date,
            exit_date=outcome.exit_date,
            days_held=outcome.days_held,
            entry_credit=outcome.entry_credit,
            exit_debit=outcome.exit_debit,
            profit_loss=outcome.profit_loss,
            profit_loss_percent=outcome.profit_loss_percent,
            exit_reason=outcome.exit_reason.replace("_", " "),
            recommendation=predicted.recommendation,
            confidence=predicted.confidence,
            expected_probability=predicted.expected_profit_probability,
            thesis=predicted.thesis[:300] if predicted.thesis else "None",
            bull_points=bull_points,
            bear_points=bear_points,
            market_regime=market_regime or "unknown",
            iv_rank=f"{iv_rank:.1f}" if iv_rank else "N/A",
            vix=f"{vix:.1f}" if vix else "N/A",
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            REFLECTION_SYSTEM,
        )

        data = self.claude._parse_json_response(response)

        reflection = TradeReflection(
            trade_id=outcome.trade_id,
            memory_id=memory_id,
            prediction_correct=data.get("prediction_correct", False),
            confidence_calibration=data.get("confidence_calibration", "calibrated"),
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            market_behavior=data.get("market_behavior", ""),
            key_lesson=data.get("key_lesson", ""),
            candidate_rules=data.get("candidate_rules", []),
        )

        # Update episodic memory with reflection
        if self.episodic and memory_id:
            actual_outcome = {
                "profit_loss": outcome.profit_loss,
                "profit_loss_percent": outcome.profit_loss_percent,
                "was_profitable": outcome.was_profitable,
                "exit_reason": outcome.exit_reason,
                "days_held": outcome.days_held,
            }
            await self.episodic.update_actual_outcome(
                memory_id=memory_id,
                actual_outcome=actual_outcome,
                reflection=f"{reflection.market_behavior}. {reflection.key_lesson}",
                lesson=reflection.key_lesson,
            )

        return reflection

    async def process_candidate_rules(
        self,
        reflection: TradeReflection,
        outcome: TradeOutcome,
    ) -> list[str]:
        """Process candidate rules from a reflection.

        Args:
            reflection: The generated reflection
            outcome: The trade outcome

        Returns:
            List of rule IDs that were created or updated
        """
        rule_ids = []

        for candidate in reflection.candidate_rules:
            rule_text = candidate.get("rule_text")
            rule_type = candidate.get("rule_type", "entry")

            if not rule_text:
                continue

            # Check if similar rule already exists
            existing = await self._find_similar_rule(rule_text, rule_type)

            if existing:
                # Update existing rule statistics
                supported = outcome.was_profitable if reflection.prediction_correct else not outcome.was_profitable
                await self.retriever.update_rule_stats(existing["id"], supported=supported)
                rule_ids.append(existing["id"])

                # Check if rule should be validated
                total_trades = existing["supporting_trades"] + existing["opposing_trades"] + 1
                if total_trades >= self.MIN_TRADES_FOR_VALIDATION:
                    await self._validate_rule(existing["id"])
            else:
                # Create new rule
                rule_id = await self.retriever.add_rule(
                    rule_text=rule_text,
                    rule_type=rule_type,
                    source="learned",
                    applies_to_agent="all",
                    conditions=self._extract_conditions(reflection, outcome),
                )
                rule_ids.append(rule_id)

        return rule_ids

    async def _find_similar_rule(self, rule_text: str, rule_type: str) -> dict | None:
        """Find an existing rule similar to the given text."""
        # Simple text matching for now - could use embeddings for better matching
        rows = await self.retriever.db.prepare("""
            SELECT * FROM semantic_rules
            WHERE rule_type = ?
              AND is_active = 1
            ORDER BY supporting_trades DESC
            LIMIT 20
        """).bind(rule_type).all()

        # Normalize text for comparison
        rule_normalized = rule_text.lower().strip()

        for row in rows.results:
            existing_normalized = row["rule_text"].lower().strip()

            # Check for high similarity (simple substring match)
            if rule_normalized in existing_normalized or existing_normalized in rule_normalized:
                return dict(row)

            # Check word overlap
            rule_words = set(rule_normalized.split())
            existing_words = set(existing_normalized.split())
            overlap = len(rule_words & existing_words) / max(len(rule_words), len(existing_words))

            if overlap > 0.7:
                return dict(row)

        return None

    async def _validate_rule(self, rule_id: str) -> None:
        """Statistically validate a rule based on trade outcomes.

        Uses a simple binomial test to check if the rule's success rate
        is significantly better than random (50%).
        """
        row = await self.retriever.db.prepare("""
            SELECT * FROM semantic_rules WHERE id = ?
        """).bind(rule_id).first()

        if not row:
            return

        supporting = row["supporting_trades"]
        opposing = row["opposing_trades"]
        total = supporting + opposing

        if total < self.MIN_TRADES_FOR_VALIDATION:
            return

        # Calculate success rate
        success_rate = supporting / total

        # Simple binomial test approximation
        # Under null hypothesis (random), p = 0.5
        # z = (p - 0.5) / sqrt(0.25 / n)
        import math
        z = (success_rate - 0.5) / math.sqrt(0.25 / total)

        # Two-tailed p-value approximation using normal CDF
        # p-value = 2 * (1 - Phi(|z|))
        from core.analysis.greeks import norm_cdf
        p_value = 2 * (1 - norm_cdf(abs(z)))

        # Effect size (Cohen's h for proportions)
        # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        effect_size = 2 * math.asin(math.sqrt(success_rate)) - 2 * math.asin(math.sqrt(0.5))

        await self.retriever.validate_rule(rule_id, p_value, effect_size)

    def _extract_conditions(self, reflection: TradeReflection, outcome: TradeOutcome) -> dict | None:
        """Extract conditions for rule application from context."""
        conditions = {}

        # Could extract regime, IV conditions, etc. from reflection
        # For now, return None to apply rule universally
        return conditions if conditions else None

    async def get_reflection_stats(self) -> dict[str, Any]:
        """Get statistics about reflections and rule validation."""
        # Count reflections
        rows = await self.retriever.db.prepare("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as with_outcomes,
                SUM(CASE WHEN reflection IS NOT NULL THEN 1 ELSE 0 END) as with_reflections
            FROM episodic_memory
        """).first()

        # Count rules
        rule_rows = await self.retriever.db.prepare("""
            SELECT
                source,
                COUNT(*) as count,
                AVG(supporting_trades) as avg_support
            FROM semantic_rules
            WHERE is_active = 1
            GROUP BY source
        """).all()

        rule_stats = {}
        for row in rule_rows.results:
            rule_stats[row["source"]] = {
                "count": row["count"],
                "avg_support": row["avg_support"],
            }

        return {
            "total_memories": rows["total"] if rows else 0,
            "with_outcomes": rows["with_outcomes"] if rows else 0,
            "with_reflections": rows["with_reflections"] if rows else 0,
            "rules_by_source": rule_stats,
        }
