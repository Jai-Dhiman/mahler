"""Working memory for agent session state.

Working memory holds transient state during a single analysis session,
including:
- Current agent context
- Intermediate results from agents
- Session-level caching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.agents.base import AgentContext, AgentMessage


@dataclass
class SessionState:
    """State for a single analysis session."""

    session_id: str
    started_at: datetime
    scan_type: str  # "morning", "midday", "afternoon"

    # Current context
    context: AgentContext | None = None

    # Accumulated messages
    analyst_messages: list[AgentMessage] = field(default_factory=list)
    debate_messages: list[AgentMessage] = field(default_factory=list)
    synthesis_message: AgentMessage | None = None
    decision_message: AgentMessage | None = None

    # Session metadata
    spreads_analyzed: int = 0
    recommendations_made: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "scan_type": self.scan_type,
            "analyst_messages": [m.to_dict() for m in self.analyst_messages],
            "debate_messages": [m.to_dict() for m in self.debate_messages],
            "synthesis_message": self.synthesis_message.to_dict() if self.synthesis_message else None,
            "decision_message": self.decision_message.to_dict() if self.decision_message else None,
            "spreads_analyzed": self.spreads_analyzed,
            "recommendations_made": self.recommendations_made,
        }


class WorkingMemory:
    """Session-scoped working memory for agents.

    Working memory is ephemeral and only persists for the duration
    of a single analysis session (e.g., one morning scan run).

    It provides:
    - Session state management
    - Intermediate result caching
    - Cross-agent communication buffer
    """

    def __init__(self, session_id: str, scan_type: str = "morning"):
        """Initialize working memory for a session.

        Args:
            session_id: Unique identifier for this session
            scan_type: Type of scan ("morning", "midday", "afternoon")
        """
        self.session = SessionState(
            session_id=session_id,
            started_at=datetime.now(),
            scan_type=scan_type,
        )

        # Cache for intermediate computations
        self._cache: dict[str, Any] = {}

        # Retrieved memories from episodic/semantic stores
        self._similar_trades: list[dict] = []
        self._active_rules: list[dict] = []

    def set_context(self, context: AgentContext) -> None:
        """Set the current analysis context."""
        self.session.context = context

    def add_analyst_message(self, message: AgentMessage) -> None:
        """Add an analyst message to the session."""
        self.session.analyst_messages.append(message)

    def add_debate_message(self, message: AgentMessage) -> None:
        """Add a debate message to the session."""
        self.session.debate_messages.append(message)

    def set_synthesis(self, message: AgentMessage) -> None:
        """Set the synthesis message."""
        self.session.synthesis_message = message

    def set_decision(self, message: AgentMessage) -> None:
        """Set the decision message."""
        self.session.decision_message = message

    def get_all_messages(self) -> list[AgentMessage]:
        """Get all messages in order."""
        messages = []
        messages.extend(self.session.analyst_messages)
        messages.extend(self.session.debate_messages)
        if self.session.synthesis_message:
            messages.append(self.session.synthesis_message)
        if self.session.decision_message:
            messages.append(self.session.decision_message)
        return messages

    def cache_value(self, key: str, value: Any) -> None:
        """Cache a computed value for later retrieval."""
        self._cache[key] = value

    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get a cached value."""
        return self._cache.get(key, default)

    def set_similar_trades(self, trades: list[dict]) -> None:
        """Set similar trades retrieved from episodic memory."""
        self._similar_trades = trades

    def get_similar_trades(self) -> list[dict]:
        """Get similar trades for context."""
        return self._similar_trades

    def set_active_rules(self, rules: list[dict]) -> None:
        """Set active rules retrieved from semantic memory."""
        self._active_rules = rules

    def get_active_rules(self) -> list[dict]:
        """Get active rules for context."""
        return self._active_rules

    def increment_analyzed(self) -> None:
        """Increment the count of spreads analyzed."""
        self.session.spreads_analyzed += 1

    def increment_recommendations(self) -> None:
        """Increment the count of recommendations made."""
        self.session.recommendations_made += 1

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the session."""
        return {
            "session_id": self.session.session_id,
            "scan_type": self.session.scan_type,
            "duration_seconds": (datetime.now() - self.session.started_at).total_seconds(),
            "spreads_analyzed": self.session.spreads_analyzed,
            "recommendations_made": self.session.recommendations_made,
            "analyst_messages": len(self.session.analyst_messages),
            "debate_rounds": len(self.session.debate_messages) // 2,
            "similar_trades_used": len(self._similar_trades),
            "rules_applied": len(self._active_rules),
        }
