"""Base agent abstractions for the multi-agent trading system.

This module provides the foundational types and abstract base class
for all agents in the V2 multi-agent architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient
    from core.ai.router import LLMRouter
    from core.analysis.iv_rank import IVMetrics, MeanReversionResult, TermStructureResult
    from core.analysis.screener import MarketRegime
    from core.types import CreditSpread, PlaybookRule, PortfolioGreeks, Position


class MessageType(str, Enum):
    """Types of messages agents can produce."""

    ANALYSIS = "analysis"  # Analyst output
    ARGUMENT = "argument"  # Debate round argument
    SYNTHESIS = "synthesis"  # Facilitator synthesis
    DECISION = "decision"  # Final trade decision


@dataclass
class ReasoningTrace:
    """ReAct-style reasoning trace for interpretability.

    Inspired by TradingAgents' use of ReAct prompting for reasoning traces.
    Captures the agent's thought process in a structured format.
    """

    observation: str  # What data am I seeing?
    thought: str  # What does this mean for the trade?
    action: str  # What should I recommend?
    reasoning: str  # Why is this the right action?

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "observation": self.observation,
            "thought": self.thought,
            "action": self.action,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningTrace:
        """Deserialize from dictionary."""
        return cls(
            observation=data.get("observation", ""),
            thought=data.get("thought", ""),
            action=data.get("action", ""),
            reasoning=data.get("reasoning", ""),
        )

    def to_string(self) -> str:
        """Format as human-readable string."""
        return (
            f"Observation: {self.observation}\n"
            f"Thought: {self.thought}\n"
            f"Action: {self.action}\n"
            f"Reasoning: {self.reasoning}"
        )


@dataclass
class AgentMessage:
    """A message produced by an agent.

    This is the standard communication format between agents in the pipeline.
    Each agent produces one or more messages that can be consumed by downstream agents.

    V2: Added raw_thinking field for Chain-of-Thought capture per TradingGroup paper.
    """

    agent_id: str
    timestamp: datetime
    message_type: MessageType
    content: str  # Human-readable analysis/argument
    structured_data: dict[str, Any] | None = None  # Machine-readable output
    confidence: float = 0.5  # 0.0 to 1.0
    reasoning_trace: ReasoningTrace | None = None  # ReAct trace for interpretability
    raw_thinking: str | None = None  # V2: Raw CoT from extended thinking

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.value,
            "content": self.content,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace.to_dict() if self.reasoning_trace else None,
            "raw_thinking": self.raw_thinking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        """Deserialize from dictionary."""
        reasoning_trace = None
        if data.get("reasoning_trace"):
            reasoning_trace = ReasoningTrace.from_dict(data["reasoning_trace"])

        return cls(
            agent_id=data["agent_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            structured_data=data.get("structured_data"),
            confidence=data.get("confidence", 0.5),
            reasoning_trace=reasoning_trace,
            raw_thinking=data.get("raw_thinking"),
        )


@dataclass
class MarketData:
    """Current market data for analysis context."""

    underlying: str
    underlying_price: float
    current_vix: float | None = None
    vix_3m: float | None = None  # For term structure
    iv_metrics: IVMetrics | None = None
    term_structure: TermStructureResult | None = None
    mean_reversion: MeanReversionResult | None = None
    regime: MarketRegime | str | None = None
    regime_probability: float | None = None
    price_bars: list[dict] | None = None  # Historical OHLCV bars


@dataclass
class PortfolioContext:
    """Current portfolio state for risk analysis."""

    positions: list[Position] = field(default_factory=list)
    portfolio_greeks: PortfolioGreeks | None = None
    account_equity: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0


@dataclass
class AgentContext:
    """Context provided to agents for analysis.

    This is the complete context an agent needs to perform its analysis.
    Different agents will use different parts of this context.
    """

    # The trade opportunity being evaluated
    spread: CreditSpread

    # Market data
    market_data: MarketData

    # Portfolio state
    portfolio: PortfolioContext

    # Historical context
    playbook_rules: list[PlaybookRule] = field(default_factory=list)
    similar_trades: list[dict] = field(default_factory=list)  # From episodic memory

    # Previous agent outputs in the pipeline (for debate rounds)
    prior_messages: list[AgentMessage] = field(default_factory=list)

    # Metadata
    scan_timestamp: datetime = field(default_factory=datetime.now)
    scan_type: Literal["morning", "midday", "afternoon"] = "morning"

    def get_analyst_messages(self) -> list[AgentMessage]:
        """Get all analyst messages from prior messages."""
        return [m for m in self.prior_messages if m.message_type == MessageType.ANALYSIS]

    def get_debate_messages(self) -> list[AgentMessage]:
        """Get all debate argument messages from prior messages."""
        return [m for m in self.prior_messages if m.message_type == MessageType.ARGUMENT]

    def get_last_opponent_message(self, my_agent_id: str) -> AgentMessage | None:
        """Get the last argument from the opposing side in a debate."""
        debate_msgs = self.get_debate_messages()
        for msg in reversed(debate_msgs):
            if msg.agent_id != my_agent_id:
                return msg
        return None


class BaseAgent(ABC):
    """Abstract base class for all agents in the trading system.

    Each agent implements the analyze() method to process context and
    produce structured output. Agents are designed to be stateless -
    all necessary context is provided via AgentContext.

    Supports two initialization modes:
    1. Direct ClaudeClient: Traditional single-model approach
    2. LLMRouter: Multi-model routing based on agent type
    """

    def __init__(
        self,
        claude: ClaudeClient | None = None,
        agent_id: str | None = None,
        router: LLMRouter | None = None,
    ):
        """Initialize the agent.

        Args:
            claude: Claude API client for LLM calls (optional if router provided)
            agent_id: Unique identifier for this agent instance
            router: LLM router for multi-model support (optional)
        """
        self.agent_id = agent_id or self.__class__.__name__

        # Support both direct client and router modes
        if router is not None:
            self.router = router
            self.claude = router.get_client(self.agent_id)
        elif claude is not None:
            self.claude = claude
            self.router = None
        else:
            raise ValueError("Either claude or router must be provided")

    @property
    @abstractmethod
    def role(self) -> str:
        """Human-readable description of this agent's role."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining this agent's behavior."""
        pass

    @abstractmethod
    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Perform analysis and produce a message.

        Args:
            context: Full context for analysis

        Returns:
            AgentMessage with analysis results
        """
        pass

    def _create_message(
        self,
        message_type: MessageType,
        content: str,
        structured_data: dict[str, Any] | None = None,
        confidence: float = 0.5,
        reasoning_trace: ReasoningTrace | None = None,
    ) -> AgentMessage:
        """Helper to create a properly formatted AgentMessage."""
        # Extract reasoning trace from structured data if present and not provided
        if reasoning_trace is None and structured_data:
            trace_data = structured_data.get("reasoning_trace")
            if trace_data and isinstance(trace_data, dict):
                reasoning_trace = ReasoningTrace.from_dict(trace_data)

        return AgentMessage(
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            message_type=message_type,
            content=content,
            structured_data=structured_data,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
        )

    def _extract_reasoning_trace(
        self,
        structured_data: dict[str, Any] | None,
    ) -> ReasoningTrace | None:
        """Extract ReAct reasoning trace from structured data."""
        if not structured_data:
            return None

        # Check for explicit trace field
        if "reasoning_trace" in structured_data:
            trace_data = structured_data["reasoning_trace"]
            if isinstance(trace_data, dict):
                return ReasoningTrace.from_dict(trace_data)

        # Try to construct from observation/thought/action/reasoning fields
        if all(k in structured_data for k in ["observation", "thought", "action", "reasoning"]):
            return ReasoningTrace(
                observation=structured_data["observation"],
                thought=structured_data["thought"],
                action=structured_data["action"],
                reasoning=structured_data["reasoning"],
            )

        return None


class AnalystAgent(BaseAgent):
    """Base class for analyst agents that produce ANALYSIS messages.

    Analyst agents examine specific data domains (IV, technicals, macro, greeks)
    and produce structured analysis that feeds into the debate layer.
    """

    @abstractmethod
    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Produce analysis message for this domain."""
        pass


class DebateAgent(BaseAgent):
    """Base class for debate agents (bull/bear researchers).

    Debate agents argue for or against a trade based on analyst outputs.
    They engage in multi-round debates with opposing perspectives.
    """

    @property
    @abstractmethod
    def perspective(self) -> Literal["bull", "bear"]:
        """This agent's perspective in the debate."""
        pass

    @abstractmethod
    async def argue(self, context: AgentContext, round_number: int) -> AgentMessage:
        """Produce an argument for this debate round.

        Args:
            context: Analysis context including prior arguments
            round_number: Current round (1-indexed)

        Returns:
            AgentMessage with ARGUMENT type
        """
        pass

    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Default analyze delegates to argue with round 1."""
        return await self.argue(context, round_number=1)


class SynthesisAgent(BaseAgent):
    """Base class for synthesis agents (facilitator, decision maker).

    Synthesis agents combine inputs from multiple sources to produce
    final conclusions or decisions.
    """

    @abstractmethod
    async def synthesize(
        self, context: AgentContext, analyst_outputs: list[AgentMessage]
    ) -> AgentMessage:
        """Synthesize multiple inputs into a conclusion.

        Args:
            context: Analysis context
            analyst_outputs: Messages from analyst agents

        Returns:
            AgentMessage with SYNTHESIS or DECISION type
        """
        pass

    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Default analyze extracts analyst messages and synthesizes."""
        analyst_outputs = context.get_analyst_messages()
        return await self.synthesize(context, analyst_outputs)
