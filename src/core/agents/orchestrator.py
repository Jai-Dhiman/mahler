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
    """Stages in the analysis pipeline.

    V2 Pipeline Order:
    1. ANALYSTS - Parallel analysis (IV, Technical, Macro, Greeks)
    2. DEBATE - Bull vs Bear researchers, N rounds
    3. SYNTHESIS - Facilitator combines debate outcome
    4. RISK_DELIBERATION - Three-perspective risk debate (new)
    5. TRADER - Synthesizes into trading proposal (new)
    6. FUND_MANAGER - Final approval authority (new)
    """

    ANALYSTS = "analysts"
    DEBATE = "debate"
    SYNTHESIS = "synthesis"
    RISK_DELIBERATION = "risk_deliberation"  # V2: Agent-based risk debate
    TRADER = "trader"  # V2: Trading proposal synthesis
    FUND_MANAGER = "fund_manager"  # V2: Final approval
    DECISION = "decision"  # Legacy stage (maps to fund_manager)


@dataclass
class DebateConfig:
    """Configuration for the debate process."""

    max_rounds: int = 3
    consensus_threshold: float = 0.7  # Confidence threshold for early consensus
    min_rounds: int = 1  # Always run at least this many rounds

    # Dynamic termination settings
    enable_dynamic_termination: bool = True
    confidence_convergence_threshold: float = 0.15  # Gap below this = convergence
    novelty_threshold: int = 2  # Minimum new arguments to continue


@dataclass
class ConvergenceResult:
    """Result of convergence detection for debate termination."""

    has_converged: bool
    reason: str
    recommendation_aligned: bool = False
    confidence_converged: bool = False
    no_new_arguments: bool = False
    bull_recommendation: str | None = None
    bear_recommendation: str | None = None
    confidence_gap: float = 1.0


@dataclass
class PipelineResult:
    """Result of running the full analysis pipeline.

    V2 additions:
    - risk_deliberation_message: Three-perspective risk debate result
    - trader_message: Trading proposal from Trader Agent
    - fund_manager_message: Final decision from Fund Manager
    """

    # All messages from the pipeline
    analyst_messages: list[AgentMessage] = field(default_factory=list)
    debate_messages: list[AgentMessage] = field(default_factory=list)
    synthesis_message: AgentMessage | None = None

    # V2: New pipeline stages
    risk_deliberation_message: AgentMessage | None = None
    trader_message: AgentMessage | None = None
    fund_manager_message: AgentMessage | None = None

    # Legacy (maps to fund_manager_message for backward compatibility)
    decision_message: AgentMessage | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Outcome
    recommendation: str | None = None  # "enter", "skip", "reduce_size"
    confidence: float = 0.0
    thesis: str = ""
    final_contracts: int = 0

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
            "risk_deliberation_message": self.risk_deliberation_message.to_dict() if self.risk_deliberation_message else None,
            "trader_message": self.trader_message.to_dict() if self.trader_message else None,
            "fund_manager_message": self.fund_manager_message.to_dict() if self.fund_manager_message else None,
            "decision_message": self.decision_message.to_dict() if self.decision_message else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "thesis": self.thesis,
            "final_contracts": self.final_contracts,
            "duration_ms": self.duration_ms,
        }


class AgentOrchestrator:
    """Orchestrates the multi-agent analysis pipeline.

    V2 Pipeline (7-agent architecture inspired by TradingAgents paper):
    1. Analysts (parallel): IV, Technical, Macro, Greeks analysts
    2. Debate (sequential): Bull vs Bear researchers, N rounds
    3. Synthesis: Facilitator combines all inputs
    4. Risk Deliberation: Three-perspective risk debate (V2)
    5. Trader: Synthesizes into trading proposal (V2)
    6. Fund Manager: Final approval authority (V2)

    Usage:
        orchestrator = AgentOrchestrator(claude)
        orchestrator.register_analyst(IVAnalyst(claude))
        orchestrator.register_analyst(TechnicalAnalyst(claude))
        orchestrator.register_debater(BullResearcher(claude), "bull")
        orchestrator.register_debater(BearResearcher(claude), "bear")
        orchestrator.set_facilitator(DebateFacilitator(claude))
        orchestrator.set_trader(TraderAgent(claude))
        orchestrator.set_fund_manager(FundManagerAgent(claude))

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

        # V2: New agent roles
        self._risk_deliberation_manager: Any | None = None  # RiskDeliberationManager
        self._trader: SynthesisAgent | None = None
        self._fund_manager: SynthesisAgent | None = None

        # Legacy (backward compatibility)
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

    def set_risk_deliberation_manager(self, manager: Any) -> None:
        """Set the risk deliberation manager (RiskDeliberationManager)."""
        self._risk_deliberation_manager = manager

    def set_trader(self, agent: SynthesisAgent) -> None:
        """Set the trader agent."""
        self._trader = agent

    def set_fund_manager(self, agent: SynthesisAgent) -> None:
        """Set the fund manager agent."""
        self._fund_manager = agent

    def set_decision_maker(self, agent: SynthesisAgent) -> None:
        """Set the decision maker agent (legacy, maps to fund_manager)."""
        self._decision_maker = agent

    async def run_pipeline(
        self,
        context: AgentContext,
        current_vix: float | None = None,
    ) -> PipelineResult:
        """Run the full V2 analysis pipeline.

        Pipeline stages:
        1. Analysts (parallel) - IV, Technical, Macro, Greeks
        2. Debate - Bull vs Bear, N rounds with dynamic termination
        3. Facilitator synthesis - Debate outcome
        4. Risk deliberation - Three-perspective agent debate (V2)
        5. Trader proposal - Concrete trading parameters (V2)
        6. Fund Manager approval - Final decision (V2)

        Args:
            context: Initial context with spread and market data
            current_vix: Current VIX level for risk assessment

        Returns:
            PipelineResult with all messages and final recommendation
        """
        result = PipelineResult(started_at=datetime.now())

        # Get VIX from context if not provided
        if current_vix is None:
            current_vix = context.market_data.current_vix or 20.0

        # Stage 1: Run analysts in parallel
        result.analyst_messages = await self._run_analysts(context)

        # Add analyst outputs to context for debate
        context.prior_messages.extend(result.analyst_messages)

        # Stage 2: Run debate (if debaters registered)
        if self._bull_debater and self._bear_debater:
            result.debate_messages = await self._run_debate(context)
            context.prior_messages.extend(result.debate_messages)

        # Stage 3: Facilitator synthesis (if facilitator registered)
        if self._facilitator:
            result.synthesis_message = await self._facilitator.synthesize(
                context, result.analyst_messages
            )
            if result.synthesis_message:
                context.prior_messages.append(result.synthesis_message)

        # Stage 4: Risk deliberation (V2 - if manager registered)
        if self._risk_deliberation_manager is not None:
            risk_result = await self._run_risk_deliberation(
                context, current_vix
            )
            if risk_result:
                result.risk_deliberation_message = risk_result
                context.prior_messages.append(risk_result)

        # Stage 5: Trader proposal (V2 - if trader registered)
        if self._trader:
            result.trader_message = await self._trader.synthesize(
                context, result.analyst_messages
            )
            if result.trader_message:
                context.prior_messages.append(result.trader_message)

        # Stage 6: Fund Manager approval (V2 - if fund manager registered)
        if self._fund_manager:
            result.fund_manager_message = await self._fund_manager.synthesize(
                context, result.analyst_messages
            )
            # Map to decision_message for backward compatibility
            result.decision_message = result.fund_manager_message

        # Legacy fallback: use decision_maker if no V2 agents
        elif self._decision_maker:
            result.decision_message = await self._decision_maker.synthesize(
                context, result.analyst_messages
            )

        # Extract final recommendation
        self._extract_recommendation(result)

        result.completed_at = datetime.now()
        return result

    async def _run_risk_deliberation(
        self,
        context: AgentContext,
        current_vix: float,
    ) -> AgentMessage | None:
        """Run three-perspective risk deliberation.

        Args:
            context: Agent context
            current_vix: Current VIX level

        Returns:
            AgentMessage with risk deliberation result
        """
        if self._risk_deliberation_manager is None:
            return None

        # Get portfolio context
        portfolio = context.portfolio

        # Run deliberation
        result = await self._risk_deliberation_manager.deliberate(
            spread=context.spread,
            account_equity=portfolio.account_equity,
            current_positions=portfolio.positions,
            current_vix=current_vix,
            market_regime=context.market_data.regime,
            daily_pnl=portfolio.daily_pnl,
        )

        # Convert to AgentMessage
        return AgentMessage(
            agent_id="risk_deliberation",
            timestamp=datetime.now(),
            message_type=MessageType.SYNTHESIS,
            content=result.weighting_rationale,
            structured_data=result.to_dict(),
            confidence=result.confidence,
        )

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

        Supports dynamic termination when:
        - Both sides recommend same action
        - Confidence gap narrows below threshold
        - Arguments stop introducing new points
        """
        if not self._bull_debater or not self._bear_debater:
            return []

        debate_messages = []
        bull_messages: list[AgentMessage] = []
        bear_messages: list[AgentMessage] = []

        for round_num in range(1, self.debate_config.max_rounds + 1):
            # Bull argues first in each round
            bull_msg = await self._bull_debater.argue(context, round_num)
            debate_messages.append(bull_msg)
            bull_messages.append(bull_msg)
            context.prior_messages.append(bull_msg)

            # Bear responds
            bear_msg = await self._bear_debater.argue(context, round_num)
            debate_messages.append(bear_msg)
            bear_messages.append(bear_msg)
            context.prior_messages.append(bear_msg)

            # Check for early termination (after minimum rounds)
            if round_num >= self.debate_config.min_rounds:
                # Simple consensus check (backward compatible)
                if self._check_consensus(bull_msg, bear_msg):
                    print(f"Debate reached consensus at round {round_num}")
                    break

                # Dynamic termination (if enabled)
                if self.debate_config.enable_dynamic_termination:
                    convergence = self._detect_convergence(bull_messages, bear_messages)
                    if convergence.has_converged:
                        print(f"Debate terminated at round {round_num}: {convergence.reason}")
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

    def _detect_convergence(
        self,
        bull_msgs: list[AgentMessage],
        bear_msgs: list[AgentMessage],
    ) -> ConvergenceResult:
        """Detect if debate has converged for dynamic termination.

        Convergence is detected when:
        1. Both sides recommend the same action (enter/skip)
        2. Confidence gap narrows below threshold
        3. Arguments stop introducing new points (stalemate)

        Args:
            bull_msgs: All bull researcher messages so far
            bear_msgs: All bear researcher messages so far

        Returns:
            ConvergenceResult indicating if and why debate has converged
        """
        if not bull_msgs or not bear_msgs:
            return ConvergenceResult(
                has_converged=False,
                reason="Insufficient messages for convergence check",
            )

        # Get latest messages
        latest_bull = bull_msgs[-1]
        latest_bear = bear_msgs[-1]

        # Extract recommendations
        bull_rec = None
        bear_rec = None
        if latest_bull.structured_data:
            bull_rec = latest_bull.structured_data.get("recommendation")
        if latest_bear.structured_data:
            bear_rec = latest_bear.structured_data.get("recommendation")

        # Check 1: Recommendation alignment
        recommendation_aligned = False
        if bull_rec and bear_rec:
            # Normalize recommendations
            bull_action = "enter" if bull_rec in ("enter", "reduce_size") else "skip"
            bear_action = "skip" if bear_rec in ("skip", "reduce_size") else "enter"

            # If both agree on core action, that's alignment
            # Note: bull_action is naturally "enter" and bear_action is naturally "skip"
            # True alignment is when they both say the same thing
            recommendation_aligned = bull_rec == bear_rec

        # Check 2: Confidence convergence
        confidence_gap = abs(latest_bull.confidence - latest_bear.confidence)
        confidence_converged = confidence_gap < self.debate_config.confidence_convergence_threshold

        # Check 3: Argument novelty (check if arguments repeat)
        no_new_arguments = self._check_argument_novelty(bull_msgs, bear_msgs)

        # Determine if converged and why
        has_converged = False
        reason = ""

        if recommendation_aligned:
            has_converged = True
            reason = f"Both sides recommend {bull_rec}"
        elif confidence_converged and len(bull_msgs) >= 2:
            has_converged = True
            reason = f"Confidence gap narrowed to {confidence_gap:.1%}"
        elif no_new_arguments and len(bull_msgs) >= 2:
            has_converged = True
            reason = "Arguments have stopped introducing new points (stalemate)"

        return ConvergenceResult(
            has_converged=has_converged,
            reason=reason,
            recommendation_aligned=recommendation_aligned,
            confidence_converged=confidence_converged,
            no_new_arguments=no_new_arguments,
            bull_recommendation=bull_rec,
            bear_recommendation=bear_rec,
            confidence_gap=confidence_gap,
        )

    def _check_argument_novelty(
        self,
        bull_msgs: list[AgentMessage],
        bear_msgs: list[AgentMessage],
    ) -> bool:
        """Check if recent arguments introduce new points.

        Returns True if there are no significant new arguments (stalemate).
        """
        if len(bull_msgs) < 2 or len(bear_msgs) < 2:
            return False

        # Extract arguments from previous and current rounds
        prev_bull_args = set()
        curr_bull_args = set()
        prev_bear_args = set()
        curr_bear_args = set()

        if bull_msgs[-2].structured_data:
            prev_bull_args = set(bull_msgs[-2].structured_data.get("key_arguments", []))
        if bull_msgs[-1].structured_data:
            curr_bull_args = set(bull_msgs[-1].structured_data.get("key_arguments", []))

        if bear_msgs[-2].structured_data:
            prev_bear_args = set(bear_msgs[-2].structured_data.get("key_arguments", []))
        if bear_msgs[-1].structured_data:
            curr_bear_args = set(bear_msgs[-1].structured_data.get("key_arguments", []))

        # Count new arguments
        new_bull_args = len(curr_bull_args - prev_bull_args)
        new_bear_args = len(curr_bear_args - prev_bear_args)

        total_new = new_bull_args + new_bear_args

        return total_new < self.debate_config.novelty_threshold

    def _extract_recommendation(self, result: PipelineResult) -> None:
        """Extract final recommendation from pipeline result.

        V2 Priority order:
        1. Fund Manager decision (final authority)
        2. Trader proposal (if no fund manager)
        3. Decision message (legacy)
        4. Synthesis message
        5. Debate consensus
        6. Analyst average
        """
        # V2: Fund Manager has final authority
        if result.fund_manager_message and result.fund_manager_message.structured_data:
            data = result.fund_manager_message.structured_data
            decision = data.get("decision", "reject")

            # Map fund manager decisions to recommendations
            if decision == "approve":
                result.recommendation = "enter"
            elif decision == "modify":
                result.recommendation = "reduce_size"
            else:
                result.recommendation = "skip"

            result.confidence = result.fund_manager_message.confidence
            result.thesis = data.get("final_thesis", result.fund_manager_message.content)
            result.final_contracts = data.get("final_contracts", 0)
            return

        # V2: Trader proposal (if no fund manager)
        if result.trader_message and result.trader_message.structured_data:
            data = result.trader_message.structured_data
            result.recommendation = data.get("action", "skip")
            result.confidence = result.trader_message.confidence
            result.thesis = data.get("entry_rationale", result.trader_message.content)
            result.final_contracts = data.get("contracts", 0)
            return

        # Legacy: decision message
        if result.decision_message and result.decision_message.structured_data:
            data = result.decision_message.structured_data
            result.recommendation = data.get("recommendation") or data.get("decision", "skip")
            result.confidence = result.decision_message.confidence
            result.thesis = result.decision_message.content
            result.final_contracts = data.get("position_size", 0)
            return

        # Synthesis message
        if result.synthesis_message and result.synthesis_message.structured_data:
            data = result.synthesis_message.structured_data
            result.recommendation = data.get("recommendation", "skip")
            result.confidence = result.synthesis_message.confidence
            result.thesis = result.synthesis_message.content
            return

        # Debate consensus
        if result.debate_messages:
            last_msg = result.debate_messages[-1]
            if last_msg.structured_data:
                result.recommendation = last_msg.structured_data.get("recommendation", "skip")
            result.confidence = last_msg.confidence
            result.thesis = last_msg.content
            return

        # Analyst average
        if result.analyst_messages:
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
