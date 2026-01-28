"""Agent orchestrator for managing the multi-agent analysis pipeline.

The orchestrator manages:
1. Parallel execution of analyst agents
2. Sequential debate rounds between bull/bear researchers
3. Final synthesis by the facilitator
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from core.agents.base import (
    AgentContext,
    AgentMessage,
    AnalystAgent,
    BaseAgent,
    DebateAgent,
    MarketData,
    MessageType,
    PortfolioContext,
    SynthesisAgent,
)

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient
    from core.analysis.iv_rank import IVMetrics, MeanReversionResult, TermStructureResult
    from core.types import CreditSpread, PlaybookRule, PortfolioGreeks, Position


class PipelineStage(str, Enum):
    """Stages in the analysis pipeline."""

    ANALYSTS = "analysts"
    DEBATE = "debate"
    SYNTHESIS = "synthesis"
    DECISION = "decision"


@dataclass
class DebateConfig:
    """Configuration for the debate process."""

    max_rounds: int = 3
    consensus_threshold: float = 0.7  # Confidence threshold for early consensus
    min_rounds: int = 1  # Always run at least this many rounds


@dataclass
class PipelineResult:
    """Result of running the full analysis pipeline."""

    # All messages from the pipeline
    analyst_messages: list[AgentMessage] = field(default_factory=list)
    debate_messages: list[AgentMessage] = field(default_factory=list)
    synthesis_message: AgentMessage | None = None
    decision_message: AgentMessage | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Outcome
    recommendation: str | None = None  # "enter", "skip", "reduce_size"
    confidence: float = 0.0
    thesis: str = ""

    @property
    def duration_ms(self) -> int | None:
        """Pipeline execution time in milliseconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "analyst_messages": [m.to_dict() for m in self.analyst_messages],
            "debate_messages": [m.to_dict() for m in self.debate_messages],
            "synthesis_message": self.synthesis_message.to_dict() if self.synthesis_message else None,
            "decision_message": self.decision_message.to_dict() if self.decision_message else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "thesis": self.thesis,
            "duration_ms": self.duration_ms,
        }


class AgentOrchestrator:
    """Orchestrates the multi-agent analysis pipeline.

    The pipeline runs in stages:
    1. Analysts (parallel): IV, Technical, Macro, Greeks analysts
    2. Debate (sequential): Bull vs Bear researchers, N rounds
    3. Synthesis: Facilitator combines all inputs
    4. Decision (optional): Final trade decision

    Usage:
        orchestrator = AgentOrchestrator(claude)
        orchestrator.register_analyst(IVAnalyst(claude))
        orchestrator.register_analyst(TechnicalAnalyst(claude))
        orchestrator.register_debater(BullResearcher(claude), "bull")
        orchestrator.register_debater(BearResearcher(claude), "bear")
        orchestrator.set_facilitator(DebateFacilitator(claude))

        result = await orchestrator.run_pipeline(context)
    """

    def __init__(
        self,
        claude: ClaudeClient,
        debate_config: DebateConfig | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            claude: Claude API client
            debate_config: Configuration for debate process
        """
        self.claude = claude
        self.debate_config = debate_config or DebateConfig()

        # Registered agents
        self._analysts: list[AnalystAgent] = []
        self._bull_debater: DebateAgent | None = None
        self._bear_debater: DebateAgent | None = None
        self._facilitator: SynthesisAgent | None = None
        self._decision_maker: SynthesisAgent | None = None

    def register_analyst(self, agent: AnalystAgent) -> None:
        """Register an analyst agent."""
        self._analysts.append(agent)

    def register_debater(self, agent: DebateAgent, perspective: str) -> None:
        """Register a debate agent."""
        if perspective == "bull":
            self._bull_debater = agent
        elif perspective == "bear":
            self._bear_debater = agent
        else:
            raise ValueError(f"Invalid perspective: {perspective}")

    def set_facilitator(self, agent: SynthesisAgent) -> None:
        """Set the debate facilitator."""
        self._facilitator = agent

    def set_decision_maker(self, agent: SynthesisAgent) -> None:
        """Set the decision maker agent."""
        self._decision_maker = agent

    async def run_pipeline(self, context: AgentContext) -> PipelineResult:
        """Run the full analysis pipeline.

        Args:
            context: Initial context with spread and market data

        Returns:
            PipelineResult with all messages and final recommendation
        """
        result = PipelineResult(started_at=datetime.now())

        # Stage 1: Run analysts in parallel
        result.analyst_messages = await self._run_analysts(context)

        # Add analyst outputs to context for debate
        context.prior_messages.extend(result.analyst_messages)

        # Stage 2: Run debate (if debaters registered)
        if self._bull_debater and self._bear_debater:
            result.debate_messages = await self._run_debate(context)
            context.prior_messages.extend(result.debate_messages)

        # Stage 3: Synthesis (if facilitator registered)
        if self._facilitator:
            result.synthesis_message = await self._facilitator.synthesize(
                context, result.analyst_messages
            )
            if result.synthesis_message:
                context.prior_messages.append(result.synthesis_message)

        # Stage 4: Decision (if decision maker registered)
        if self._decision_maker:
            result.decision_message = await self._decision_maker.synthesize(
                context, result.analyst_messages
            )

        # Extract final recommendation
        self._extract_recommendation(result)

        result.completed_at = datetime.now()
        return result

    async def _run_analysts(self, context: AgentContext) -> list[AgentMessage]:
        """Run all analyst agents in parallel."""
        if not self._analysts:
            return []

        # Run all analysts concurrently
        tasks = [analyst.analyze(context) for analyst in self._analysts]
        messages = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_messages = []
        for i, msg in enumerate(messages):
            if isinstance(msg, Exception):
                print(f"Analyst {self._analysts[i].agent_id} failed: {msg}")
            elif isinstance(msg, AgentMessage):
                valid_messages.append(msg)

        return valid_messages

    async def _run_debate(self, context: AgentContext) -> list[AgentMessage]:
        """Run debate rounds between bull and bear researchers.

        The debate alternates between bull and bear, with each side
        seeing the opponent's previous argument.
        """
        if not self._bull_debater or not self._bear_debater:
            return []

        debate_messages = []

        for round_num in range(1, self.debate_config.max_rounds + 1):
            # Bull argues first in each round
            bull_msg = await self._bull_debater.argue(context, round_num)
            debate_messages.append(bull_msg)
            context.prior_messages.append(bull_msg)

            # Bear responds
            bear_msg = await self._bear_debater.argue(context, round_num)
            debate_messages.append(bear_msg)
            context.prior_messages.append(bear_msg)

            # Check for early consensus (both sides high confidence, same direction)
            if round_num >= self.debate_config.min_rounds:
                if self._check_consensus(bull_msg, bear_msg):
                    print(f"Debate reached consensus at round {round_num}")
                    break

        return debate_messages

    def _check_consensus(self, bull_msg: AgentMessage, bear_msg: AgentMessage) -> bool:
        """Check if both sides have reached consensus."""
        threshold = self.debate_config.consensus_threshold

        # If both sides have high confidence and their structured data
        # indicates the same recommendation, we have consensus
        if bull_msg.confidence < threshold or bear_msg.confidence < threshold:
            return False

        bull_rec = bull_msg.structured_data.get("recommendation") if bull_msg.structured_data else None
        bear_rec = bear_msg.structured_data.get("recommendation") if bear_msg.structured_data else None

        return bull_rec == bear_rec and bull_rec is not None

    def _extract_recommendation(self, result: PipelineResult) -> None:
        """Extract final recommendation from pipeline result."""
        # Priority: decision > synthesis > debate consensus > analyst average
        if result.decision_message and result.decision_message.structured_data:
            data = result.decision_message.structured_data
            result.recommendation = data.get("recommendation", "skip")
            result.confidence = result.decision_message.confidence
            result.thesis = result.decision_message.content
        elif result.synthesis_message and result.synthesis_message.structured_data:
            data = result.synthesis_message.structured_data
            result.recommendation = data.get("recommendation", "skip")
            result.confidence = result.synthesis_message.confidence
            result.thesis = result.synthesis_message.content
        elif result.debate_messages:
            # Use last debate message
            last_msg = result.debate_messages[-1]
            if last_msg.structured_data:
                result.recommendation = last_msg.structured_data.get("recommendation", "skip")
            result.confidence = last_msg.confidence
            result.thesis = last_msg.content
        elif result.analyst_messages:
            # Average analyst confidence
            confidences = [m.confidence for m in result.analyst_messages]
            result.confidence = sum(confidences) / len(confidences)
            result.recommendation = "enter" if result.confidence > 0.5 else "skip"
            result.thesis = "; ".join(m.content[:100] for m in result.analyst_messages)


def build_agent_context(
    spread: CreditSpread,
    underlying_price: float,
    iv_metrics: IVMetrics | None = None,
    term_structure: TermStructureResult | None = None,
    mean_reversion: MeanReversionResult | None = None,
    regime: str | None = None,
    regime_probability: float | None = None,
    current_vix: float | None = None,
    vix_3m: float | None = None,
    price_bars: list[dict] | None = None,
    positions: list[Position] | None = None,
    portfolio_greeks: PortfolioGreeks | None = None,
    account_equity: float = 0.0,
    buying_power: float = 0.0,
    daily_pnl: float = 0.0,
    weekly_pnl: float = 0.0,
    playbook_rules: list[PlaybookRule] | None = None,
    similar_trades: list[dict] | None = None,
    scan_type: str = "morning",
) -> AgentContext:
    """Build an AgentContext from individual components.

    This is a convenience function for constructing the context
    object that gets passed to the agent pipeline.
    """
    market_data = MarketData(
        underlying=spread.underlying,
        underlying_price=underlying_price,
        current_vix=current_vix,
        vix_3m=vix_3m,
        iv_metrics=iv_metrics,
        term_structure=term_structure,
        mean_reversion=mean_reversion,
        regime=regime,
        regime_probability=regime_probability,
        price_bars=price_bars,
    )

    portfolio = PortfolioContext(
        positions=positions or [],
        portfolio_greeks=portfolio_greeks,
        account_equity=account_equity,
        buying_power=buying_power,
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
    )

    return AgentContext(
        spread=spread,
        market_data=market_data,
        portfolio=portfolio,
        playbook_rules=playbook_rules or [],
        similar_trades=similar_trades or [],
        scan_type=scan_type,
    )
