"""Reflection module for continuous learning from trade outcomes.

The reflection engine analyzes closed trades to:
- Compare predictions with actual outcomes
- Generate reflections explaining what happened
- Extract candidate rules for future decisions
- Validate rules statistically over time

Usage:
    from core.reflection import (
        SelfReflectionEngine,
        TradeOutcome,
        PredictedOutcome,
        TradeReflection,
    )

    engine = SelfReflectionEngine(claude, memory_retriever, episodic_store)

    reflection = await engine.generate_reflection(
        outcome=trade_outcome,
        predicted=predicted_outcome,
        memory_id=memory_id,
    )

    rule_ids = await engine.process_candidate_rules(reflection, trade_outcome)
"""

from core.reflection.engine import (
    PredictedOutcome,
    SelfReflectionEngine,
    TradeOutcome,
    TradeReflection,
)

__all__ = [
    "PredictedOutcome",
    "SelfReflectionEngine",
    "TradeOutcome",
    "TradeReflection",
]
