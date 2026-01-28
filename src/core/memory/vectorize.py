"""Episodic memory store using Cloudflare Vectorize.

Provides vector-based similarity search for retrieving similar past trades
to inform current trading decisions.
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

        Returns:
            ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        entry_date = datetime.now().strftime("%Y-%m-%d")

        # Convert messages to dicts
        analyst_outputs = [m.to_dict() for m in analyst_messages]
        debate_transcript = [m.to_dict() for m in debate_messages]
        debate_outcome = synthesis_message.to_dict() if synthesis_message else None

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

        # Store embedding in Vectorize
        embedding_id = f"episodic_{memory_id}"
        await self.vectorize.upsert([
            {
                "id": embedding_id,
                "values": embedding,
                "metadata": {
                    "memory_id": memory_id,
                    "underlying": underlying,
                    "spread_type": spread_type,
                    "market_regime": market_regime or "unknown",
                    "entry_date": entry_date,
                },
            }
        ])

        # Store metadata in D1
        await self.db.prepare("""
            INSERT INTO episodic_memory (
                id, trade_id, entry_date, underlying, spread_type,
                short_strike, long_strike, expiration,
                analyst_outputs, debate_transcript, debate_outcome,
                predicted_outcome, embedding_id,
                market_regime, iv_rank, vix_at_entry
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> list[SimilarTradeResult]:
        """Find similar past trades using vector similarity.

        Args:
            underlying: The underlying symbol
            spread_type: Type of spread
            market_regime: Current market regime
            iv_rank: Current IV rank
            vix_at_entry: Current VIX
            top_k: Number of results to return

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

        # Query Vectorize
        results = await self.vectorize.query(embedding, {"topK": top_k})

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

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Workers AI."""
        response = await self.ai.run(self.EMBEDDING_MODEL, {"text": text})
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
        )

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
