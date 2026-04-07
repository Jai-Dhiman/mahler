"""Episodic memory store using Cloudflare Vectorize.

Provides vector-based similarity search for retrieving similar past trades
to inform current trading decisions.

Vectorize Optimizations:
- Namespace partitioning: Vectors are stored in per-underlying namespaces
  (spy-trades, qqq-trades, iwm-trades) to reduce search space
- Metadata filtering: market_regime, spread_type, win fields are indexed
  for efficient pre-filtering

Setup (run once with wrangler):
    wrangler vectorize create-metadata-index mahler-episodic --property-name=market_regime --type=string
    wrangler vectorize create-metadata-index mahler-episodic --property-name=spread_type --type=string
    wrangler vectorize create-metadata-index mahler-episodic --property-name=win --type=boolean
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.agents.base import AgentMessage


@dataclass
class EpisodicMemoryRecord:
    """A single episodic memory record."""

    id: str
    trade_id: str | None
    entry_date: str
    underlying: str
    spread_type: str
    short_strike: float
    long_strike: float
    expiration: str

    # Agent outputs
    analyst_outputs: list[dict]
    debate_transcript: list[dict]
    debate_outcome: dict | None

    # Outcomes
    predicted_outcome: dict | None
    actual_outcome: dict | None

    # Learning
    reflection: str | None
    lesson_extracted: str | None
    embedding_id: str | None

    # Context
    market_regime: str | None
    iv_rank: float | None
    vix_at_entry: float | None

    created_at: datetime

    # FinMem fields
    memory_layer: str = "shallow"
    access_count: int = 0
    last_accessed_at: datetime | None = None
    critical_event: bool = False
    critical_event_reason: str | None = None
    pnl_dollars: float | None = None
    pnl_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "trade_id": self.trade_id,
            "entry_date": self.entry_date,
            "underlying": self.underlying,
            "spread_type": self.spread_type,
            "short_strike": self.short_strike,
            "long_strike": self.long_strike,
            "expiration": self.expiration,
            "analyst_outputs": self.analyst_outputs,
            "debate_transcript": self.debate_transcript,
            "debate_outcome": self.debate_outcome,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "reflection": self.reflection,
            "lesson_extracted": self.lesson_extracted,
            "embedding_id": self.embedding_id,
            "market_regime": self.market_regime,
            "iv_rank": self.iv_rank,
            "vix_at_entry": self.vix_at_entry,
            "created_at": self.created_at.isoformat(),
            # FinMem fields
            "memory_layer": self.memory_layer,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "critical_event": self.critical_event,
            "critical_event_reason": self.critical_event_reason,
            "pnl_dollars": self.pnl_dollars,
            "pnl_percent": self.pnl_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemoryRecord:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            trade_id=data.get("trade_id"),
            entry_date=data["entry_date"],
            underlying=data["underlying"],
            spread_type=data["spread_type"],
            short_strike=data["short_strike"],
            long_strike=data["long_strike"],
            expiration=data["expiration"],
            analyst_outputs=data.get("analyst_outputs", []),
            debate_transcript=data.get("debate_transcript", []),
            debate_outcome=data.get("debate_outcome"),
            predicted_outcome=data.get("predicted_outcome"),
            actual_outcome=data.get("actual_outcome"),
            reflection=data.get("reflection"),
            lesson_extracted=data.get("lesson_extracted"),
            embedding_id=data.get("embedding_id"),
            market_regime=data.get("market_regime"),
            iv_rank=data.get("iv_rank"),
            vix_at_entry=data.get("vix_at_entry"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            # FinMem fields
            memory_layer=data.get("memory_layer", "shallow"),
            access_count=data.get("access_count", 0),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]) if data.get("last_accessed_at") else None,
            critical_event=data.get("critical_event", False),
            critical_event_reason=data.get("critical_event_reason"),
            pnl_dollars=data.get("pnl_dollars"),
            pnl_percent=data.get("pnl_percent"),
        )


@dataclass
class SimilarTradeResult:
    """Result from similarity search."""

    memory: EpisodicMemoryRecord
    similarity_score: float
    match_reasons: list[str]


class EpisodicMemoryStore:
    """Store for episodic memories with vector similarity search.

    Uses Cloudflare Vectorize for embedding storage and similarity search,
    with D1 for metadata storage.

    The embedding is generated from a text representation of the trade context
    including underlying, regime, IV conditions, and outcome.
    """

    # Embedding model: Workers AI bge-small-en-v1.5 produces 384-dim vectors
    EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5"
    EMBEDDING_DIM = 384

    def __init__(self, vectorize_binding: Any, ai_binding: Any, d1_binding: Any):
        """Initialize the episodic memory store.

        Args:
            vectorize_binding: Cloudflare Vectorize binding (EPISODIC_MEMORY)
            ai_binding: Cloudflare AI binding for embeddings
            d1_binding: D1 database binding for metadata
        """
        self.vectorize = vectorize_binding
        self.ai = ai_binding
        self.db = d1_binding

    async def store_memory(
        self,
        trade_id: str | None,
        underlying: str,
        spread_type: str,
        short_strike: float,
        long_strike: float,
        expiration: str,
        analyst_messages: list[AgentMessage],
        debate_messages: list[AgentMessage],
        synthesis_message: AgentMessage | None,
        market_regime: str | None = None,
        iv_rank: float | None = None,
        vix_at_entry: float | None = None,
        predicted_outcome: dict | None = None,
        pnl_dollars: float | None = None,
        pnl_percent: float | None = None,
    ) -> str:
        """Store a new episodic memory.

        Args:
            trade_id: Optional trade ID if trade was executed
            underlying: The underlying symbol
            spread_type: Type of spread (bull_put, bear_call)
            short_strike: Short strike price
            long_strike: Long strike price
            expiration: Expiration date
            analyst_messages: Messages from analyst agents
            debate_messages: Messages from debate
            synthesis_message: Final synthesis from facilitator
            market_regime: Current market regime
            iv_rank: IV rank at entry
            vix_at_entry: VIX level at entry
            predicted_outcome: Predicted trade outcome
            pnl_dollars: P&L in dollars (if trade closed)
            pnl_percent: P&L percentage (if trade closed)

        Returns:
            ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        entry_date = datetime.now().strftime("%Y-%m-%d")

        # Convert messages to dicts
        analyst_outputs = [m.to_dict() for m in analyst_messages]
        debate_transcript = [m.to_dict() for m in debate_messages]
        debate_outcome = synthesis_message.to_dict() if synthesis_message else None

        # Detect critical event
        is_critical, critical_reason = await self._detect_critical_event(
            pnl_dollars=pnl_dollars,
            pnl_percent=pnl_percent,
            predicted_outcome=predicted_outcome,
        )

        # Critical events start in intermediate layer (fast-track)
        memory_layer = "intermediate" if is_critical else "shallow"

        # Generate embedding text
        embedding_text = self._generate_embedding_text(
            underlying=underlying,
            spread_type=spread_type,
            market_regime=market_regime,
            iv_rank=iv_rank,
            vix_at_entry=vix_at_entry,
            analyst_messages=analyst_messages,
            synthesis_message=synthesis_message,
        )

        # Generate embedding using Workers AI
        embedding = await self._generate_embedding(embedding_text)

        # Store embedding in Vectorize with FinMem metadata
        # Use namespace partitioning by underlying for query optimization
        embedding_id = f"episodic_{memory_id}"
        namespace = f"{underlying.lower()}-trades"

        await self.vectorize.upsert(
            [
                {
                    "id": embedding_id,
                    "values": embedding,
                    "metadata": {
                        "memory_id": memory_id,
                        "underlying": underlying,
                        "spread_type": spread_type,
                        "market_regime": market_regime or "unknown",
                        "entry_date": entry_date,
                        "memory_layer": memory_layer,
                        "critical_event": is_critical,
                        "win": None,  # Updated after trade closes
                    },
                }
            ],
            namespace=namespace,
        )

        # Store metadata in D1 with FinMem fields
        await self.db.prepare("""
            INSERT INTO episodic_memory (
                id, trade_id, entry_date, underlying, spread_type,
                short_strike, long_strike, expiration,
                analyst_outputs, debate_transcript, debate_outcome,
                predicted_outcome, embedding_id,
                market_regime, iv_rank, vix_at_entry,
                memory_layer, critical_event, critical_event_reason,
                pnl_dollars, pnl_percent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """).bind(
            memory_id,
            trade_id,
            entry_date,
            underlying,
            spread_type,
            short_strike,
            long_strike,
            expiration,
            json.dumps(analyst_outputs),
            json.dumps(debate_transcript),
            json.dumps(debate_outcome) if debate_outcome else None,
            json.dumps(predicted_outcome) if predicted_outcome else None,
            embedding_id,
            market_regime,
            iv_rank,
            vix_at_entry,
            memory_layer,
            1 if is_critical else 0,
            critical_reason,
            pnl_dollars,
            pnl_percent,
        ).run()

        return memory_id

    async def find_similar(
        self,
        underlying: str,
        spread_type: str,
        market_regime: str | None = None,
        iv_rank: float | None = None,
        vix_at_entry: float | None = None,
        top_k: int = 5,
        layer_filter: list[str] | None = None,
        include_critical_only: bool = False,
        filter_to_wins: bool = False,
        filter_spread_type: bool = False,
    ) -> list[SimilarTradeResult]:
        """Find similar past trades using vector similarity.

        Uses Cloudflare Vectorize with namespace partitioning (per-underlying)
        and metadata filtering for optimized queries.

        Args:
            underlying: The underlying symbol
            spread_type: Type of spread
            market_regime: Current market regime for metadata filter
            iv_rank: Current IV rank (used in query text)
            vix_at_entry: Current VIX (used in query text)
            top_k: Number of results to return
            layer_filter: Optional list of layers to include (e.g., ["intermediate", "deep"])
            include_critical_only: If True, only return critical events
            filter_to_wins: If True, only return winning trades (for learning from success)
            filter_spread_type: If True, filter by spread_type in metadata

        Returns:
            List of similar trade results with similarity scores
        """
        # Generate query embedding
        query_text = self._generate_query_text(
            underlying=underlying,
            spread_type=spread_type,
            market_regime=market_regime,
            iv_rank=iv_rank,
            vix_at_entry=vix_at_entry,
        )

        embedding = await self._generate_embedding(query_text)

        # Build Vectorize query options with metadata filtering
        # Use returnMetadata="indexed" for faster queries (only indexed fields returned)
        query_options: dict[str, Any] = {
            "topK": top_k * 2,  # Over-fetch for post-filtering
            "returnMetadata": "indexed",
        }

        # Build metadata filter conditions
        # These leverage Cloudflare Vectorize metadata indexes for efficient filtering
        filter_conditions: dict[str, Any] = {}

        if market_regime:
            filter_conditions["market_regime"] = {"$eq": market_regime}
        if filter_spread_type:
            filter_conditions["spread_type"] = {"$eq": spread_type}
        if layer_filter:
            filter_conditions["memory_layer"] = {"$in": layer_filter}
        if include_critical_only:
            filter_conditions["critical_event"] = {"$eq": True}
        if filter_to_wins:
            filter_conditions["win"] = {"$eq": True}

        if filter_conditions:
            query_options["filter"] = filter_conditions

        # Query Vectorize with namespace partitioning by underlying
        # This significantly reduces search space for per-underlying queries
        namespace = f"{underlying.lower()}-trades"
        results = await self.vectorize.query(embedding, query_options, namespace=namespace)

        similar_trades = []
        for match in results.matches:
            memory_id = match.metadata.get("memory_id")
            if not memory_id:
                continue

            # Fetch full memory from D1
            row = await self.db.prepare("""
                SELECT * FROM episodic_memory WHERE id = ?
            """).bind(memory_id).first()

            if not row:
                continue

            memory = self._row_to_record(row)
            match_reasons = self._explain_match(memory, underlying, spread_type, market_regime)

            similar_trades.append(SimilarTradeResult(
                memory=memory,
                similarity_score=match.score,
                match_reasons=match_reasons,
            ))

            # Limit to requested top_k after filtering
            if len(similar_trades) >= top_k:
                break

        return similar_trades

    async def update_actual_outcome(
        self,
        memory_id: str,
        actual_outcome: dict,
        reflection: str | None = None,
        lesson: str | None = None,
    ) -> None:
        """Update a memory with actual outcome after trade closes.

        Args:
            memory_id: ID of the memory to update
            actual_outcome: Actual trade outcome
            reflection: AI-generated reflection
            lesson: Extracted lesson
        """
        await self.db.prepare("""
            UPDATE episodic_memory
            SET actual_outcome = ?,
                reflection = ?,
                lesson_extracted = ?,
                updated_at = datetime('now')
            WHERE id = ?
        """).bind(
            json.dumps(actual_outcome),
            reflection,
            lesson,
            memory_id,
        ).run()

        # Also update win/loss metadata in Vectorize for filtering
        # Determine win/loss from actual_outcome
        profit_loss = actual_outcome.get("profit_loss", 0)
        win = profit_loss > 0

        # Get the underlying to determine namespace
        row = await self.db.prepare("""
            SELECT underlying, embedding_id FROM episodic_memory WHERE id = ?
        """).bind(memory_id).first()

        if row and row.get("embedding_id"):
            await self.update_outcome_metadata(
                embedding_id=row["embedding_id"],
                underlying=row["underlying"],
                win=win,
            )

    async def update_outcome_metadata(
        self,
        embedding_id: str,
        underlying: str,
        win: bool,
    ) -> None:
        """Update vector metadata after trade closes.

        Note: Cloudflare Vectorize requires re-upserting to update metadata.
        We fetch the existing vector and re-upsert with updated metadata.

        Args:
            embedding_id: ID of the embedding in Vectorize
            underlying: Underlying symbol (for namespace)
            win: Whether the trade was profitable
        """
        namespace = f"{underlying.lower()}-trades"

        # Fetch existing vector by ID
        existing = await self.vectorize.getByIds([embedding_id], namespace=namespace)

        if not existing or not existing.vectors:
            return

        vector = existing.vectors[0]
        updated_metadata = {**vector.metadata, "win": win}

        # Re-upsert with updated metadata
        await self.vectorize.upsert(
            [
                {
                    "id": embedding_id,
                    "values": vector.values,
                    "metadata": updated_metadata,
                }
            ],
            namespace=namespace,
        )

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Workers AI."""
        # Workers AI expects text as an array for embedding models
        response = await self.ai.run(self.EMBEDDING_MODEL, {"text": [text]})
        return response.data[0]

    def _generate_embedding_text(
        self,
        underlying: str,
        spread_type: str,
        market_regime: str | None,
        iv_rank: float | None,
        vix_at_entry: float | None,
        analyst_messages: list[AgentMessage],
        synthesis_message: AgentMessage | None,
    ) -> str:
        """Generate text for embedding from trade context."""
        parts = [
            f"Underlying: {underlying}",
            f"Strategy: {spread_type.replace('_', ' ')}",
        ]

        if market_regime:
            parts.append(f"Market regime: {market_regime.replace('_', ' ')}")

        if iv_rank is not None:
            iv_level = "high" if iv_rank > 70 else "elevated" if iv_rank > 50 else "normal" if iv_rank > 30 else "low"
            parts.append(f"IV environment: {iv_level} ({iv_rank:.0f}%)")

        if vix_at_entry is not None:
            vix_level = "extreme" if vix_at_entry > 40 else "elevated" if vix_at_entry > 25 else "normal" if vix_at_entry > 15 else "low"
            parts.append(f"VIX: {vix_level} ({vix_at_entry:.1f})")

        # Add key analyst observations
        for msg in analyst_messages[:2]:  # Limit to prevent embedding bloat
            if msg.structured_data:
                if "iv_signal" in msg.structured_data:
                    parts.append(f"IV signal: {msg.structured_data['iv_signal']}")
                if "trend" in msg.structured_data:
                    parts.append(f"Trend: {msg.structured_data['trend']}")
                if "regime_assessment" in msg.structured_data:
                    parts.append(f"Macro: {msg.structured_data['regime_assessment']}")

        # Add synthesis if available
        if synthesis_message:
            parts.append(f"Outcome: {synthesis_message.content[:200]}")

        return ". ".join(parts)

    def _generate_query_text(
        self,
        underlying: str,
        spread_type: str,
        market_regime: str | None,
        iv_rank: float | None,
        vix_at_entry: float | None,
    ) -> str:
        """Generate query text for similarity search."""
        parts = [
            f"Underlying: {underlying}",
            f"Strategy: {spread_type.replace('_', ' ')}",
        ]

        if market_regime:
            parts.append(f"Market regime: {market_regime.replace('_', ' ')}")

        if iv_rank is not None:
            iv_level = "high" if iv_rank > 70 else "elevated" if iv_rank > 50 else "normal" if iv_rank > 30 else "low"
            parts.append(f"IV environment: {iv_level}")

        if vix_at_entry is not None:
            vix_level = "extreme" if vix_at_entry > 40 else "elevated" if vix_at_entry > 25 else "normal" if vix_at_entry > 15 else "low"
            parts.append(f"VIX: {vix_level}")

        return ". ".join(parts)

    def _row_to_record(self, row: dict) -> EpisodicMemoryRecord:
        """Convert a D1 row to an EpisodicMemoryRecord."""
        return EpisodicMemoryRecord(
            id=row["id"],
            trade_id=row.get("trade_id"),
            entry_date=row["entry_date"],
            underlying=row["underlying"],
            spread_type=row["spread_type"],
            short_strike=row["short_strike"],
            long_strike=row["long_strike"],
            expiration=row["expiration"],
            analyst_outputs=json.loads(row.get("analyst_outputs") or "[]"),
            debate_transcript=json.loads(row.get("debate_transcript") or "[]"),
            debate_outcome=json.loads(row.get("debate_outcome")) if row.get("debate_outcome") else None,
            predicted_outcome=json.loads(row.get("predicted_outcome")) if row.get("predicted_outcome") else None,
            actual_outcome=json.loads(row.get("actual_outcome")) if row.get("actual_outcome") else None,
            reflection=row.get("reflection"),
            lesson_extracted=row.get("lesson_extracted"),
            embedding_id=row.get("embedding_id"),
            market_regime=row.get("market_regime"),
            iv_rank=row.get("iv_rank"),
            vix_at_entry=row.get("vix_at_entry"),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(),
            # FinMem fields
            memory_layer=row.get("memory_layer", "shallow"),
            access_count=row.get("access_count", 0),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row.get("last_accessed_at") else None,
            critical_event=bool(row.get("critical_event", 0)),
            critical_event_reason=row.get("critical_event_reason"),
            pnl_dollars=row.get("pnl_dollars"),
            pnl_percent=row.get("pnl_percent"),
        )

    async def _detect_critical_event(
        self,
        pnl_dollars: float | None,
        pnl_percent: float | None,
        predicted_outcome: dict | None,
    ) -> tuple[bool, str | None]:
        """Detect if this memory represents a critical event.

        Critical events from FinMem:
        1. Large P&L impact: |pnl_dollars| > 2 * average
        2. Correct against consensus: predicted "skip" but profitable

        Args:
            pnl_dollars: P&L in dollars
            pnl_percent: P&L percentage
            predicted_outcome: Predicted outcome dict

        Returns:
            Tuple of (is_critical, reason)
        """
        # Criterion 1: Large P&L impact (absolute threshold until we have average)
        if pnl_dollars is not None and abs(pnl_dollars) > 500:
            return True, f"large_pnl_impact: ${pnl_dollars:.2f}"

        if pnl_percent is not None and abs(pnl_percent) > 10:
            return True, f"large_pnl_percent: {pnl_percent:.1f}%"

        # Criterion 2: Correct against consensus (skip prediction that was profitable)
        if predicted_outcome:
            recommendation = predicted_outcome.get("recommendation", "").lower()
            confidence = predicted_outcome.get("confidence", 0)

            # If we skipped with high confidence but it would have been profitable
            if recommendation == "skip" and confidence > 0.7 and pnl_percent is not None and pnl_percent > 5:
                return True, "correct_against_consensus: skipped_but_profitable"

            # If we entered with high confidence but lost significantly
            if recommendation in ("enter", "execute") and confidence > 0.7 and pnl_percent is not None and pnl_percent < -5:
                return True, "learning_opportunity: entered_but_significant_loss"

        return False, None

    async def get_average_pnl_dollars(self) -> float:
        """Get the average absolute P&L dollars across all memories.

        Used for dynamic critical event detection threshold.

        Returns:
            Average absolute P&L in dollars
        """
        row = await self.db.prepare("""
            SELECT AVG(ABS(pnl_dollars)) as avg_pnl
            FROM episodic_memory
            WHERE pnl_dollars IS NOT NULL
        """).first()

        if row and row.get("avg_pnl"):
            return float(row["avg_pnl"])
        return 250.0  # Default threshold if no data

    def _explain_match(
        self,
        memory: EpisodicMemoryRecord,
        underlying: str,
        spread_type: str,
        market_regime: str | None,
    ) -> list[str]:
        """Explain why this memory matched."""
        reasons = []

        if memory.underlying == underlying:
            reasons.append(f"Same underlying ({underlying})")
        if memory.spread_type == spread_type:
            reasons.append(f"Same strategy ({spread_type.replace('_', ' ')})")
        if market_regime and memory.market_regime == market_regime:
            reasons.append(f"Same market regime ({market_regime.replace('_', ' ')})")
        if memory.actual_outcome:
            outcome_type = "profitable" if memory.actual_outcome.get("profit_loss", 0) > 0 else "loss"
            reasons.append(f"Has {outcome_type} outcome data")

        return reasons
