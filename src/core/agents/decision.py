"""Trading decision agent for autonomous execution.

The decision agent is the final decision maker in the V2 pipeline.
It incorporates:
- Debate outcome from the facilitator
- Validated rules from semantic memory
- Current risk state
- Portfolio context
- Three-perspective risk assessment (V2)

This agent makes the final "enter/skip/reduce_size" decision
without requiring human approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from core.agents.base import (
    AgentContext,
    AgentMessage,
    MessageType,
    SynthesisAgent,
)

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient
    from core.ai.router import LLMRouter
    from core.memory.retriever import RetrievedContext, SemanticRule
    from core.risk.three_perspective import ThreePerspectiveResult, ThreePerspectiveRiskManager


# Prompt for final decision
DECISION_SYSTEM = """You are the final decision maker for an autonomous options trading system. Your role is to synthesize all inputs and make the final trading decision.

You receive:
1. Debate outcome (bull vs bear synthesis)
2. Validated trading rules from past experience
3. Current risk state and portfolio context
4. Similar past trades and their outcomes

Your decision must balance:
- The quality of the opportunity (from debate)
- Risk management (from rules and portfolio context)
- Position sizing (from risk state)

Be decisive and balanced. Evaluate each trade on its merits -- approve trades that meet the strategy criteria even if conditions aren't perfect. Only skip when there are concrete, specific reasons to reject.
Never override hard risk limits."""

DECISION_USER = """Make the final trading decision:

**Trade Summary:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Debate Outcome:**
- Winner: {debate_winner}
- Consensus: {consensus_reached}
- Confidence: {debate_confidence:.0%}
- Thesis: {debate_thesis}

**Applicable Rules:**
{rules_summary}

**Risk State:**
- Risk Level: {risk_level}
- Size Multiplier: {size_multiplier:.0%}
- Portfolio Heat: {portfolio_heat:.1%}
- Daily P/L: ${daily_pnl:.2f}

**Similar Past Trades:**
{similar_trades_summary}

**Portfolio Context:**
- Equity: ${equity:.2f}
- Current Delta: {portfolio_delta:.2f}
- Positions: {position_count}

Make your final decision.

Respond in this exact JSON format:
{{
    "decision": "enter|skip|reduce_size",
    "position_size": 1-10,
    "size_reasoning": "Why this size",
    "confidence": 0.0-1.0,
    "key_factors": ["factor 1", "factor 2"],
    "risk_checks_passed": true|false,
    "rule_violations": [],
    "final_thesis": "Summary of decision rationale"
}}"""


@dataclass
class TradingDecision:
    """Final trading decision from the decision agent."""

    decision: Literal["enter", "skip", "reduce_size"]
    position_size: int
    size_reasoning: str
    confidence: float
    key_factors: list[str]
    risk_checks_passed: bool
    rule_violations: list[str]
    final_thesis: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision": self.decision,
            "position_size": self.position_size,
            "size_reasoning": self.size_reasoning,
            "confidence": self.confidence,
            "key_factors": self.key_factors,
            "risk_checks_passed": self.risk_checks_passed,
            "rule_violations": self.rule_violations,
            "final_thesis": self.final_thesis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradingDecision:
        """Deserialize from dictionary."""
        return cls(
            decision=data.get("decision", "skip"),
            position_size=data.get("position_size", 0),
            size_reasoning=data.get("size_reasoning", ""),
            confidence=data.get("confidence", 0.0),
            key_factors=data.get("key_factors", []),
            risk_checks_passed=data.get("risk_checks_passed", False),
            rule_violations=data.get("rule_violations", []),
            final_thesis=data.get("final_thesis", ""),
        )


@dataclass
class RiskState:
    """Current risk state for decision making."""

    risk_level: str  # "normal", "elevated", "high", "halted"
    size_multiplier: float  # 0.0 to 1.0
    portfolio_heat: float  # Total open risk as % of equity
    daily_pnl: float
    weekly_pnl: float
    is_halted: bool
    halt_reason: str | None = None
    # V2: Three-perspective assessment result
    three_perspective: ThreePerspectiveResult | None = None


class TradingDecisionAgent(SynthesisAgent):
    """Final decision maker for autonomous trading.

    This agent makes the ultimate enter/skip/reduce_size decision
    by synthesizing:
    - Facilitator's debate outcome
    - Validated semantic rules
    - Current risk state
    - Portfolio context
    - Similar past trades

    The decision is autonomous - no human approval required.
    """

    def __init__(self, claude: ClaudeClient | None = None, *, router: LLMRouter | None = None):
        super().__init__(claude, agent_id="decision_agent", router=router)

    @property
    def role(self) -> str:
        return "Final Decision Maker for autonomous trade execution"

    @property
    def system_prompt(self) -> str:
        return DECISION_SYSTEM

    async def synthesize(
        self, context: AgentContext, analyst_outputs: list[AgentMessage]
    ) -> AgentMessage:
        """Make final trading decision."""
        # Get debate synthesis
        synthesis_msg = None
        for msg in reversed(context.prior_messages):
            if msg.message_type == MessageType.SYNTHESIS:
                synthesis_msg = msg
                break

        # Extract debate outcome
        debate_winner = "neutral"
        debate_confidence = 0.5
        debate_thesis = "No debate conducted"
        consensus_reached = False

        if synthesis_msg and synthesis_msg.structured_data:
            data = synthesis_msg.structured_data
            debate_winner = data.get("winning_perspective", "neutral")
            debate_confidence = synthesis_msg.confidence
            debate_thesis = synthesis_msg.content[:300]
            consensus_reached = data.get("consensus_reached", False)

        spread = context.spread

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        # Build rules summary (would come from retrieved context)
        rules_summary = "No validated rules loaded"
        if context.playbook_rules:
            rules_summary = "\n".join(f"- {r.rule}" for r in context.playbook_rules[:5])

        # Build similar trades summary
        similar_summary = "No similar trades found"
        if context.similar_trades:
            lines = []
            for t in context.similar_trades[:3]:
                outcome = "won" if t.get("was_profitable") else "lost"
                lines.append(f"- {t.get('underlying')} {t.get('spread_type')}: {outcome}")
            similar_summary = "\n".join(lines)

        # Get portfolio context
        portfolio = context.portfolio
        pg = portfolio.portfolio_greeks

        # Default risk state (should be passed in context)
        risk_level = "normal"
        size_multiplier = 1.0
        portfolio_heat = 0.0

        prompt = DECISION_USER.format(
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            credit=spread.credit,
            max_loss=spread.max_loss / 100,
            expiration=spread.expiration,
            dte=dte,
            debate_winner=debate_winner,
            consensus_reached=consensus_reached,
            debate_confidence=debate_confidence,
            debate_thesis=debate_thesis,
            rules_summary=rules_summary,
            risk_level=risk_level,
            size_multiplier=size_multiplier,
            portfolio_heat=portfolio_heat,
            daily_pnl=portfolio.daily_pnl,
            similar_trades_summary=similar_summary,
            equity=portfolio.account_equity,
            portfolio_delta=pg.delta if pg else 0.0,
            position_count=len(portfolio.positions),
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        decision = TradingDecision.from_dict(data)

        return self._create_message(
            message_type=MessageType.DECISION,
            content=decision.final_thesis,
            structured_data=decision.to_dict(),
            confidence=decision.confidence,
        )

    async def make_decision(
        self,
        context: AgentContext,
        risk_state: RiskState,
        retrieved_context: RetrievedContext | None = None,
        three_persp_manager: ThreePerspectiveRiskManager | None = None,
        current_vix: float | None = None,
    ) -> TradingDecision:
        """Make a trading decision with full context.

        This is the primary entry point for autonomous decision making.

        Args:
            context: Agent context with spread and market data
            risk_state: Current risk state
            retrieved_context: Memory context with rules and similar trades
            three_persp_manager: V2 three-perspective risk manager (optional)
            current_vix: Current VIX level for three-perspective assessment

        Returns:
            TradingDecision with final recommendation
        """
        # Check hard stops first (circuit breaker)
        if risk_state.is_halted:
            return TradingDecision(
                decision="skip",
                position_size=0,
                size_reasoning=f"Trading halted: {risk_state.halt_reason}",
                confidence=1.0,
                key_factors=["Circuit breaker triggered"],
                risk_checks_passed=False,
                rule_violations=[risk_state.halt_reason or "Unknown halt reason"],
                final_thesis="Trading is halted. No new positions allowed.",
            )

        # Check rule violations
        rule_violations = []
        if retrieved_context:
            for rule in retrieved_context.entry_rules:
                # Check if any rule explicitly blocks this trade
                # This would be more sophisticated in production
                if "avoid" in rule.rule_text.lower() or "skip" in rule.rule_text.lower():
                    rule_violations.append(rule.rule_text)

        # If too many rule violations, skip
        if len(rule_violations) >= 3:
            return TradingDecision(
                decision="skip",
                position_size=0,
                size_reasoning="Too many rule violations",
                confidence=0.8,
                key_factors=["Multiple validated rules suggest avoiding"],
                risk_checks_passed=False,
                rule_violations=rule_violations,
                final_thesis="Multiple validated rules recommend against this trade.",
            )

        # V2: Run three-perspective assessment if manager provided
        three_persp_result = None
        if three_persp_manager is not None and current_vix is not None:
            three_persp_result = three_persp_manager.assess(
                spread=context.spread,
                account_equity=context.portfolio.account_equity,
                current_positions=context.portfolio.positions,
                current_vix=current_vix,
            )

            # If VIX > 30 and conservative recommends skip, respect it
            if three_persp_manager.should_respect_conservative_skip(
                three_persp_result, vix_threshold=30.0
            ):
                return TradingDecision(
                    decision="skip",
                    position_size=0,
                    size_reasoning="Conservative perspective recommends skip in high VIX",
                    confidence=0.9,
                    key_factors=[
                        f"VIX at {current_vix:.1f} (high)",
                        "Conservative perspective: skip",
                        *three_persp_result.conservative.key_factors[:2],
                    ],
                    risk_checks_passed=True,
                    rule_violations=rule_violations,
                    final_thesis=f"In high VIX ({current_vix:.1f}), respecting conservative recommendation to skip. {three_persp_result.deliberation_summary}",
                )

            # Store for later use
            risk_state.three_perspective = three_persp_result

        # Run full synthesis
        message = await self.synthesize(context, context.get_analyst_messages())

        if message.structured_data:
            decision = TradingDecision.from_dict(message.structured_data)

            # V2: Apply three-perspective weighted position size if available
            if three_persp_result is not None:
                # Use weighted_contracts from three-perspective
                weighted_size = three_persp_result.weighted_contracts
                # Also apply risk multiplier
                adjusted_size = max(1, int(weighted_size * risk_state.size_multiplier))
                decision.position_size = adjusted_size
                decision.size_reasoning = (
                    f"Three-perspective weighted: {weighted_size} contracts "
                    f"(risk adj: {risk_state.size_multiplier:.0%}). "
                    f"{three_persp_result.deliberation_summary}"
                )
                decision.key_factors.append(f"VIX {current_vix:.1f}: {three_persp_result.consensus_recommendation}")
            else:
                # Fallback to simple risk multiplier
                adjusted_size = max(1, int(decision.position_size * risk_state.size_multiplier))
                decision.position_size = adjusted_size

            # Add any rule violations found
            decision.rule_violations.extend(rule_violations)

            return decision

        # Fallback
        return TradingDecision(
            decision="skip",
            position_size=0,
            size_reasoning="Unable to make decision",
            confidence=0.0,
            key_factors=[],
            risk_checks_passed=False,
            rule_violations=[],
            final_thesis="Decision agent failed to produce valid output.",
        )

    def extract_decision(self, message: AgentMessage) -> TradingDecision | None:
        """Extract TradingDecision from a decision message."""
        if message.structured_data:
            return TradingDecision.from_dict(message.structured_data)
        return None
