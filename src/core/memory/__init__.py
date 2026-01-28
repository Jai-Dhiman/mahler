"""Memory module for the V2 multi-agent system.

Provides layered memory:
- Working Memory: Session-scoped transient state
- Episodic Memory: Vector-searchable past trade records
- Semantic Memory: Validated trading rules

Usage:
    from core.memory import (
        WorkingMemory,
        EpisodicMemoryStore,
        MemoryRetriever,
    )

    # Session memory
    working = WorkingMemory(session_id="scan_123", scan_type="morning")
    working.set_context(context)
    working.add_analyst_message(message)

    # Long-term memory (requires Cloudflare bindings)
    episodic = EpisodicMemoryStore(vectorize, ai, db)
    memory_id = await episodic.store_memory(...)
    similar = await episodic.find_similar(underlying="SPY", ...)

    # Unified retrieval
    retriever = MemoryRetriever(db, episodic)
    context = await retriever.retrieve_context(underlying="SPY", ...)
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
]
