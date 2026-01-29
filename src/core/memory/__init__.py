"""Memory module for the V2 multi-agent system.

Provides layered memory:
- Working Memory: Session-scoped transient state
- Episodic Memory: Vector-searchable past trade records
- Semantic Memory: Validated trading rules

FinMem enhancements (https://arxiv.org/abs/2311.13743):
- Three-tier episodic memory (shallow/intermediate/deep)
- Exponential decay scoring with layer-specific stability
- Critical event detection and fast-tracking
- Adaptive cognitive span based on VIX
- Memory consolidation and promotion

Usage:
    from core.memory import (
        WorkingMemory,
        EpisodicMemoryStore,
        MemoryRetriever,
        MemoryScorer,
        MemoryConsolidator,
        MemoryLayer,
        CognitiveSpanConfig,
    )

    # Session memory
    working = WorkingMemory(session_id="scan_123", scan_type="morning")
    working.set_context(context)
    working.add_analyst_message(message)

    # Long-term memory (requires Cloudflare bindings)
    episodic = EpisodicMemoryStore(vectorize, ai, db)
    memory_id = await episodic.store_memory(...)
    similar = await episodic.find_similar(underlying="SPY", ...)

    # Unified retrieval with FinMem scoring
    retriever = MemoryRetriever(db, episodic)
    context = await retriever.retrieve_context(underlying="SPY", vix=25.0, ...)

    # Manual scoring
    scorer = MemoryScorer()
    scores = scorer.score_memory(entry_date, layer, similarity, pnl, is_critical)

    # Consolidation sweep
    consolidator = MemoryConsolidator(db)
    promotions = await consolidator.run_consolidation_sweep()
"""

from core.memory.working import (
    SessionState,
    WorkingMemory,
)
from core.memory.vectorize import (
    EpisodicMemoryRecord,
    EpisodicMemoryStore,
    SimilarTradeResult,
)
from core.memory.retriever import (
    MemoryRetriever,
    RetrievedContext,
    SemanticRule,
)
from core.memory.types import (
    MemoryLayer,
    MemoryScores,
    CognitiveSpanConfig,
    PromotionThresholds,
    LAYER_STABILITY_CONSTANTS,
    CRITICAL_EVENT_BONUS,
    CRITICAL_EVENT_PNL_MULTIPLIER,
)
from core.memory.scoring import MemoryScorer
from core.memory.consolidation import MemoryConsolidator

__all__ = [
    # Working Memory
    "SessionState",
    "WorkingMemory",
    # Episodic Memory
    "EpisodicMemoryRecord",
    "EpisodicMemoryStore",
    "SimilarTradeResult",
    # Retriever
    "MemoryRetriever",
    "RetrievedContext",
    "SemanticRule",
    # FinMem Types
    "MemoryLayer",
    "MemoryScores",
    "CognitiveSpanConfig",
    "PromotionThresholds",
    "LAYER_STABILITY_CONSTANTS",
    "CRITICAL_EVENT_BONUS",
    "CRITICAL_EVENT_PNL_MULTIPLIER",
    # FinMem Components
    "MemoryScorer",
    "MemoryConsolidator",
]
