"""Multi-agent trading system components.

This package provides the V2 multi-agent architecture for autonomous
trading decisions through analyst collaboration and debate.

Components:
- Base types: AgentMessage, AgentContext, BaseAgent
- Analysts: IVAnalyst, TechnicalAnalyst, MacroAnalyst, GreeksAnalyst
- Researchers: BullResearcher, BearResearcher (debate agents)
- Facilitator: DebateFacilitator (synthesis agent)
- Orchestrator: AgentOrchestrator for pipeline management

Usage:
    from core.agents import (
        AgentOrchestrator,
        IVAnalyst,
        TechnicalAnalyst,
        MacroAnalyst,
        GreeksAnalyst,
        BullResearcher,
        BearResearcher,
        DebateFacilitator,
        build_agent_context,
    )

    # Create orchestrator with Claude client
    orchestrator = AgentOrchestrator(claude)

    # Register analysts
    orchestrator.register_analyst(IVAnalyst(claude))
    orchestrator.register_analyst(TechnicalAnalyst(claude))
    orchestrator.register_analyst(MacroAnalyst(claude))
    orchestrator.register_analyst(GreeksAnalyst(claude))

    # Register debate agents
    orchestrator.register_debater(BullResearcher(claude), "bull")
    orchestrator.register_debater(BearResearcher(claude), "bear")
    orchestrator.set_facilitator(DebateFacilitator(claude))

    # Build context and run pipeline
    context = build_agent_context(spread, underlying_price, ...)
    result = await orchestrator.run_pipeline(context)
"""

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
from core.agents.orchestrator import (
    AgentOrchestrator,
    DebateConfig,
    PipelineResult,
    PipelineStage,
    build_agent_context,
)
from core.agents.analysts import (
    GreeksAnalyst,
    IVAnalyst,
    MacroAnalyst,
    TechnicalAnalyst,
)
from core.agents.researchers import (
    BearResearcher,
    BullResearcher,
)
from core.agents.facilitator import (
    DebateFacilitator,
    DebateOutcome,
    DebatePerspective,
)
from core.agents.decision import (
    RiskState,
    TradingDecision,
    TradingDecisionAgent,
)

__all__ = [
    # Base types
    "AgentContext",
    "AgentMessage",
    "AnalystAgent",
    "BaseAgent",
    "DebateAgent",
    "MarketData",
    "MessageType",
    "PortfolioContext",
    "SynthesisAgent",
    # Orchestrator
    "AgentOrchestrator",
    "DebateConfig",
    "PipelineResult",
    "PipelineStage",
    "build_agent_context",
    # Analysts
    "GreeksAnalyst",
    "IVAnalyst",
    "MacroAnalyst",
    "TechnicalAnalyst",
    # Researchers (Debate)
    "BearResearcher",
    "BullResearcher",
    # Facilitator
    "DebateFacilitator",
    "DebateOutcome",
    "DebatePerspective",
    # Decision Agent
    "RiskState",
    "TradingDecision",
    "TradingDecisionAgent",
]
