"""Debate facilitator agent for synthesizing trading decisions.

The facilitator weighs bull and bear arguments objectively and produces
a final recommendation for trade entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from core.agents.base import (
    AgentContext,
    AgentMessage,
    MessageType,
    SynthesisAgent,
)
from core.ai.prompts import (
    FACILITATOR_SYSTEM,
    FACILITATOR_USER,
)

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient


class DebatePerspective(str, Enum):
    """Which side won the debate."""

    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class DebateOutcome:
    """Structured outcome of the debate process.

    This captures the facilitator's synthesis of the bull/bear debate
    and provides the final trading recommendation.
    """

    winning_perspective: DebatePerspective
    confidence: float
    key_bull_points: list[str]
    key_bear_points: list[str]
    deciding_factors: list[str]
    consensus_reached: bool
    recommendation: Literal["enter", "skip", "reduce_size"]
    position_size_multiplier: float  # 0.0 to 1.0
    thesis: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "winning_perspective": self.winning_perspective.value,
            "confidence": self.confidence,
            "key_bull_points": self.key_bull_points,
            "key_bear_points": self.key_bear_points,
            "deciding_factors": self.deciding_factors,
            "consensus_reached": self.consensus_reached,
            "recommendation": self.recommendation,
            "position_size_multiplier": self.position_size_multiplier,
            "thesis": self.thesis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebateOutcome:
        """Deserialize from dictionary."""
        return cls(
            winning_perspective=DebatePerspective(data.get("winning_perspective", "neutral")),
            confidence=data.get("confidence", 0.5),
            key_bull_points=data.get("key_bull_points", []),
            key_bear_points=data.get("key_bear_points", []),
            deciding_factors=data.get("deciding_factors", []),
            consensus_reached=data.get("consensus_reached", False),
            recommendation=data.get("recommendation", "skip"),
            position_size_multiplier=data.get("position_size_multiplier", 0.0),
            thesis=data.get("thesis", ""),
        )


class DebateFacilitator(SynthesisAgent):
    """Facilitator that synthesizes debate arguments into a final decision.

    The facilitator:
    1. Reviews all analyst reports
    2. Weighs bull and bear arguments objectively
    3. Identifies the strongest points on each side
    4. Determines a final recommendation
    5. Sets position size based on confidence
    """

    def __init__(self, claude: ClaudeClient):
        super().__init__(claude, agent_id="facilitator")

    @property
    def role(self) -> str:
        return "Debate Facilitator synthesizing arguments into final decision"

    @property
    def system_prompt(self) -> str:
        return FACILITATOR_SYSTEM

    async def synthesize(
        self, context: AgentContext, analyst_outputs: list[AgentMessage]
    ) -> AgentMessage:
        """Synthesize debate and analyst outputs into final recommendation."""
        spread = context.spread

        # Get debate messages
        debate_msgs = context.get_debate_messages()

        # Separate bull and bear arguments
        bull_args = [m for m in debate_msgs if "bull" in m.agent_id.lower()]
        bear_args = [m for m in debate_msgs if "bear" in m.agent_id.lower()]

        # Format arguments for prompt
        bull_arguments = self._format_debate_arguments(bull_args)
        bear_arguments = self._format_debate_arguments(bear_args)
        analyst_summary = self._format_analyst_summary(analyst_outputs)

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        prompt = FACILITATOR_USER.format(
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            credit=spread.credit,
            max_loss=spread.max_loss / 100,
            expiration=spread.expiration,
            dte=dte,
            bull_arguments=bull_arguments,
            bear_arguments=bear_arguments,
            analyst_summary=analyst_summary,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        # Create DebateOutcome
        outcome = DebateOutcome(
            winning_perspective=DebatePerspective(data.get("winning_perspective", "neutral")),
            confidence=data.get("confidence", 0.5),
            key_bull_points=data.get("key_bull_points", []),
            key_bear_points=data.get("key_bear_points", []),
            deciding_factors=data.get("deciding_factors", []),
            consensus_reached=data.get("consensus_reached", False),
            recommendation=data.get("recommendation", "skip"),
            position_size_multiplier=data.get("position_size_multiplier", 0.0),
            thesis=data.get("thesis", ""),
        )

        return self._create_message(
            message_type=MessageType.SYNTHESIS,
            content=outcome.thesis,
            structured_data=outcome.to_dict(),
            confidence=outcome.confidence,
        )

    def _format_debate_arguments(self, messages: list[AgentMessage]) -> str:
        """Format debate arguments for the prompt."""
        if not messages:
            return "No arguments presented."

        formatted = []
        for i, msg in enumerate(messages, 1):
            round_info = f"Round {i}"

            # Extract key arguments from structured data
            key_args = []
            if msg.structured_data:
                key_args = msg.structured_data.get("key_arguments", [])

            args_str = "\n".join(f"  - {arg}" for arg in key_args) if key_args else msg.content

            formatted.append(f"**{round_info}** (conviction: {msg.confidence:.0%}):\n{args_str}")

        return "\n\n".join(formatted)

    def _format_analyst_summary(self, analyst_outputs: list[AgentMessage]) -> str:
        """Format analyst outputs for the prompt."""
        if not analyst_outputs:
            return "No analyst reports available."

        summaries = []
        for msg in analyst_outputs:
            agent_name = msg.agent_id.replace("_", " ").title()

            # Extract signal/assessment from structured data
            signal = ""
            if msg.structured_data:
                if "iv_signal" in msg.structured_data:
                    signal = f"IV: {msg.structured_data['iv_signal']}"
                elif "trend" in msg.structured_data:
                    signal = f"Trend: {msg.structured_data['trend']}"
                elif "regime_assessment" in msg.structured_data:
                    signal = f"Regime: {msg.structured_data['regime_assessment']}"
                elif "portfolio_fit" in msg.structured_data:
                    signal = f"Fit: {msg.structured_data['portfolio_fit']}"

            signal_str = f" [{signal}]" if signal else ""
            summaries.append(f"- {agent_name}{signal_str}: {msg.content[:100]}")

        return "\n".join(summaries)

    def extract_outcome(self, message: AgentMessage) -> DebateOutcome | None:
        """Extract DebateOutcome from a synthesis message."""
        if message.structured_data:
            return DebateOutcome.from_dict(message.structured_data)
        return None
