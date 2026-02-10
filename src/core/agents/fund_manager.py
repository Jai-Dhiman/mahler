"""Fund Manager Agent for final trade decisions.

The Fund Manager is the final authority in the trading pipeline.
It evaluates the debate outcome directly, determines position sizing,
and makes the approve/reject/modify decision in a single step.

Combines the former Trader (proposal synthesis) and Fund Manager (approval)
roles into one agent to reduce LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    from core.risk.three_perspective import ThreePerspectiveResult


FUND_MANAGER_SYSTEM = """You are the Fund Manager with final authority over all trades. You directly evaluate debate outcomes and make trading decisions -- there is no intermediate trader.

Your responsibilities:
1. Evaluate the debate outcome (bull vs bear synthesis)
2. Determine whether to enter, skip, or modify the trade
3. Set position size based on conviction, risk, and portfolio context
4. Apply risk management constraints

Decision framework:
- APPROVE: Trade meets criteria, risk is acceptable, specify contract count
- MODIFY: Trade has merit but needs smaller size
- REJECT: Risk outweighs reward, skip entirely

Key principles:
- Preservation of capital is paramount
- When in doubt, reduce size rather than reject outright -- the system is paper trading and needs live experience to calibrate
- Size positions proportionally to confidence and debate outcome quality
- Consider portfolio-level impact, not just individual trade merit
- During elevated VIX, verify that position sizing accounts for higher volatility but do not reject solely due to VIX level
- When debate is contentious, proceed with moderate size rather than skipping -- paper trading benefits from live experience"""

FUND_MANAGER_USER = """Evaluate this trade opportunity and make a final decision:

**Trade Opportunity:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Short Strike: ${short_strike:.2f}
- Long Strike: ${long_strike:.2f}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Debate Outcome:**
- Winning Perspective: {winning_perspective}
- Debate Confidence: {debate_confidence:.0%}
- Consensus Reached: {consensus_reached}
- Key Bull Points: {bull_points}
- Key Bear Points: {bear_points}
- Deciding Factors: {deciding_factors}
- Facilitator Thesis: {facilitator_thesis}

**Risk Team Deliberation:**
{risk_deliberation}

**Portfolio Context:**
- Account Equity: ${equity:.2f}
- Available Buying Power: ${buying_power:.2f}
- Current Positions: {position_count}
- Portfolio Delta: {portfolio_delta:.2f}
- Portfolio Heat: {portfolio_heat:.1%}
- Daily P/L: ${daily_pnl:.2f}

**Risk Parameters:**
- Max Risk Per Trade: 2% of equity (${max_risk_amount:.2f})

**Market Context:**
- Current VIX: {current_vix}
- Market Regime: {market_regime}

Make your final decision as Fund Manager.

Respond in this exact JSON format:
{{
    "decision": "approve|modify|reject",
    "final_contracts": 0-10,
    "override_reason": "If modifying/rejecting, explain why",
    "confidence": 0.0-1.0,
    "key_factors": ["factor 1", "factor 2"],
    "risk_concerns": ["concern 1", "concern 2"],
    "approval_conditions": ["condition if any"],
    "final_thesis": "Summary of decision rationale"
}}"""


@dataclass
class FinalDecision:
    """Final trading decision from the Fund Manager."""

    decision: Literal["approve", "modify", "reject"]
    final_contracts: int
    override_reason: str
    confidence: float
    key_factors: list[str]
    risk_concerns: list[str]
    approval_conditions: list[str]
    final_thesis: str

    # Original proposal metadata
    original_contracts: int = 0
    original_action: str = "skip"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision": self.decision,
            "final_contracts": self.final_contracts,
            "override_reason": self.override_reason,
            "confidence": self.confidence,
            "key_factors": self.key_factors,
            "risk_concerns": self.risk_concerns,
            "approval_conditions": self.approval_conditions,
            "final_thesis": self.final_thesis,
            "original_contracts": self.original_contracts,
            "original_action": self.original_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalDecision:
        """Deserialize from dictionary."""
        return cls(
            decision=data.get("decision", "reject"),
            final_contracts=data.get("final_contracts", 0),
            override_reason=data.get("override_reason", ""),
            confidence=data.get("confidence", 0.0),
            key_factors=data.get("key_factors", []),
            risk_concerns=data.get("risk_concerns", []),
            approval_conditions=data.get("approval_conditions", []),
            final_thesis=data.get("final_thesis", ""),
            original_contracts=data.get("original_contracts", 0),
            original_action=data.get("original_action", "skip"),
        )

    @property
    def should_execute(self) -> bool:
        """Check if this decision approves trade execution."""
        return self.decision in ("approve", "modify") and self.final_contracts > 0

    @property
    def was_modified(self) -> bool:
        """Check if the fund manager modified the original proposal."""
        return self.decision == "modify" or self.final_contracts != self.original_contracts

    @property
    def was_rejected(self) -> bool:
        """Check if the fund manager rejected the trade."""
        return self.decision == "reject" or self.final_contracts == 0


class FundManagerAgent(SynthesisAgent):
    """Fund Manager Agent with final decision authority.

    Pipeline position: Final stage - after Facilitator and Risk Deliberation.
    Evaluates debate outcome directly and makes the trading decision in one step.

    The Fund Manager:
    1. Extracts debate outcome from facilitator synthesis
    2. Considers risk team deliberation
    3. Determines position size and action
    4. Makes final approve/modify/reject decision
    5. Applies hard override rules
    """

    def __init__(self, claude: ClaudeClient | None = None, *, router: LLMRouter | None = None):
        super().__init__(claude, agent_id="fund_manager", router=router)

    @property
    def role(self) -> str:
        return "Fund Manager with final trade decision authority"

    @property
    def system_prompt(self) -> str:
        return FUND_MANAGER_SYSTEM

    async def synthesize(
        self, context: AgentContext, analyst_outputs: list[AgentMessage]
    ) -> AgentMessage:
        """Evaluate debate outcome and make final decision."""
        debate_outcome = self._extract_debate_outcome(context)
        risk_deliberation = self._extract_risk_deliberation(context)

        decision = await self.review_and_decide(
            context=context,
            debate_outcome=debate_outcome,
            risk_deliberation=risk_deliberation,
        )

        return self._create_message(
            message_type=MessageType.DECISION,
            content=decision.final_thesis,
            structured_data=decision.to_dict(),
            confidence=decision.confidence,
        )

    async def review_and_decide(
        self,
        context: AgentContext,
        debate_outcome: dict[str, Any] | None = None,
        risk_deliberation: dict[str, Any] | None = None,
        current_vix: float | None = None,
    ) -> FinalDecision:
        """Evaluate debate outcome and risk deliberation, make final decision.

        Args:
            context: Agent context with spread and portfolio data
            debate_outcome: Structured debate outcome from facilitator
            risk_deliberation: Risk team deliberation result
            current_vix: Current VIX level

        Returns:
            FinalDecision with approval/modification/rejection
        """
        spread = context.spread
        portfolio = context.portfolio
        market = context.market_data

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        # Extract debate outcome if not provided
        if debate_outcome is None:
            debate_outcome = self._extract_debate_outcome(context)

        # Extract risk deliberation if not provided
        if risk_deliberation is None:
            risk_deliberation = self._extract_risk_deliberation(context)

        # Get VIX from context if not provided
        if current_vix is None:
            current_vix = market.current_vix or 20.0

        # Extract debate fields
        winning_perspective = debate_outcome.get("winning_perspective", "neutral")
        debate_confidence = debate_outcome.get("confidence", 0.5)
        consensus_reached = debate_outcome.get("consensus_reached", False)
        key_bull_points = debate_outcome.get("key_bull_points", [])
        key_bear_points = debate_outcome.get("key_bear_points", [])
        deciding_factors = debate_outcome.get("deciding_factors", [])
        facilitator_thesis = debate_outcome.get("thesis", "No thesis provided")

        # Calculate risk parameters
        max_risk_amount = portfolio.account_equity * 0.02
        portfolio_heat = self._calculate_portfolio_heat(portfolio)

        # Format risk deliberation for prompt
        risk_deliberation_str = self._format_risk_deliberation(risk_deliberation)

        prompt = FUND_MANAGER_USER.format(
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            short_strike=spread.short_strike,
            long_strike=spread.long_strike,
            credit=spread.credit,
            max_loss=spread.max_loss / 100,
            expiration=spread.expiration,
            dte=dte,
            winning_perspective=winning_perspective,
            debate_confidence=debate_confidence,
            consensus_reached=consensus_reached,
            bull_points=", ".join(key_bull_points[:3]) if key_bull_points else "None",
            bear_points=", ".join(key_bear_points[:3]) if key_bear_points else "None",
            deciding_factors=", ".join(deciding_factors[:3]) if deciding_factors else "None",
            facilitator_thesis=facilitator_thesis[:300],
            risk_deliberation=risk_deliberation_str,
            equity=portfolio.account_equity,
            buying_power=portfolio.buying_power,
            position_count=len(portfolio.positions),
            portfolio_delta=portfolio.portfolio_greeks.delta if portfolio.portfolio_greeks else 0.0,
            portfolio_heat=portfolio_heat,
            daily_pnl=portfolio.daily_pnl,
            max_risk_amount=max_risk_amount,
            current_vix=f"{current_vix:.1f}",
            market_regime=market.regime or "unknown",
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        decision = FinalDecision(
            decision=data.get("decision", "reject"),
            final_contracts=data.get("final_contracts", 0),
            override_reason=data.get("override_reason", ""),
            confidence=data.get("confidence", 0.0),
            key_factors=data.get("key_factors", []),
            risk_concerns=data.get("risk_concerns", []),
            approval_conditions=data.get("approval_conditions", []),
            final_thesis=data.get("final_thesis", ""),
        )

        # Apply hard override rules
        decision = self._apply_overrides(decision, portfolio, spread, current_vix)

        return decision

    def _extract_debate_outcome(self, context: AgentContext) -> dict[str, Any]:
        """Extract debate outcome from prior messages."""
        # Look for facilitator synthesis message
        for msg in reversed(context.prior_messages):
            if msg.message_type == MessageType.SYNTHESIS and msg.structured_data:
                return msg.structured_data

        # Fallback: construct from debate messages
        debate_msgs = context.get_debate_messages()
        if not debate_msgs:
            return {
                "winning_perspective": "neutral",
                "confidence": 0.5,
                "consensus_reached": False,
                "key_bull_points": [],
                "key_bear_points": [],
                "deciding_factors": [],
                "thesis": "No debate conducted",
            }

        # Analyze debate messages to determine outcome
        bull_confidence = 0.0
        bear_confidence = 0.0
        key_bull_points = []
        key_bear_points = []

        for msg in debate_msgs:
            if "bull" in msg.agent_id.lower():
                bull_confidence = max(bull_confidence, msg.confidence)
                if msg.structured_data:
                    key_bull_points.extend(msg.structured_data.get("key_arguments", []))
            elif "bear" in msg.agent_id.lower():
                bear_confidence = max(bear_confidence, msg.confidence)
                if msg.structured_data:
                    key_bear_points.extend(msg.structured_data.get("key_arguments", []))

        # Determine winner
        if bull_confidence > bear_confidence + 0.1:
            winning = "bull"
        elif bear_confidence > bull_confidence + 0.1:
            winning = "bear"
        else:
            winning = "neutral"

        return {
            "winning_perspective": winning,
            "confidence": max(bull_confidence, bear_confidence),
            "consensus_reached": abs(bull_confidence - bear_confidence) < 0.1,
            "key_bull_points": key_bull_points[:3],
            "key_bear_points": key_bear_points[:3],
            "deciding_factors": [],
            "thesis": "Outcome derived from debate messages",
        }

    def _extract_risk_deliberation(self, context: AgentContext) -> dict[str, Any] | None:
        """Extract risk deliberation from prior messages."""
        for msg in reversed(context.prior_messages):
            if "risk" in msg.agent_id.lower() and msg.structured_data:
                return msg.structured_data

        return None

    def _format_risk_deliberation(self, risk_deliberation: dict[str, Any] | None) -> str:
        """Format risk deliberation for the prompt."""
        if not risk_deliberation:
            return "No risk deliberation available."

        parts = []

        # Three-perspective result format
        if "aggressive" in risk_deliberation:
            agg = risk_deliberation.get("aggressive", {})
            neu = risk_deliberation.get("neutral", {})
            con = risk_deliberation.get("conservative", {})

            parts.append(f"- Aggressive: {agg.get('recommendation', 'N/A')} ({agg.get('recommended_contracts', 0)} contracts)")
            parts.append(f"- Neutral: {neu.get('recommendation', 'N/A')} ({neu.get('recommended_contracts', 0)} contracts)")
            parts.append(f"- Conservative: {con.get('recommendation', 'N/A')} ({con.get('recommended_contracts', 0)} contracts)")

            if "weighted_contracts" in risk_deliberation:
                parts.append(f"- Weighted Result: {risk_deliberation['weighted_contracts']} contracts")

            if "deliberation_summary" in risk_deliberation:
                parts.append(f"- Summary: {risk_deliberation['deliberation_summary']}")

        # Agent-based deliberation format
        elif "consensus" in risk_deliberation:
            parts.append(f"- Consensus: {risk_deliberation.get('consensus', 'No consensus')}")
            parts.append(f"- Recommended Size: {risk_deliberation.get('recommended_contracts', 0)} contracts")

            concerns = risk_deliberation.get("key_concerns", [])
            if concerns:
                parts.append(f"- Key Concerns: {', '.join(concerns[:3])}")

        return "\n".join(parts) if parts else "Risk assessment available but format unknown."

    def _calculate_portfolio_heat(self, portfolio) -> float:
        """Calculate current portfolio risk as percentage of equity."""
        if not portfolio.positions or portfolio.account_equity <= 0:
            return 0.0

        total_risk = sum(
            getattr(p, "max_loss", 0) * getattr(p, "contracts", 1)
            for p in portfolio.positions
        )

        return total_risk / portfolio.account_equity

    def _apply_overrides(
        self,
        decision: FinalDecision,
        portfolio,
        spread,
        current_vix: float,
    ) -> FinalDecision:
        """Apply hard override rules that cannot be bypassed.

        These are circuit-breaker level rules that the fund manager
        must respect regardless of the LLM's decision.
        """
        # Override 1: Reject if VIX > 40 (extreme fear)
        if current_vix > 40:
            if decision.decision != "reject":
                decision.decision = "reject"
                decision.final_contracts = 0
                decision.override_reason = f"VIX at {current_vix:.1f} exceeds extreme threshold (40)"
                decision.risk_concerns.append("Extreme market volatility - circuit breaker")

        # Override 2: Reject if portfolio heat > 12%
        portfolio_heat = self._calculate_portfolio_heat(portfolio)
        if portfolio_heat > 0.12:
            if decision.decision != "reject":
                decision.decision = "reject"
                decision.final_contracts = 0
                decision.override_reason = f"Portfolio heat at {portfolio_heat:.1%} exceeds 12% limit"
                decision.risk_concerns.append("Portfolio risk concentration - circuit breaker")

        # Override 3: Cap size at 2% risk per trade
        max_risk = portfolio.account_equity * 0.02
        max_contracts = int(max_risk / (spread.max_loss / 100)) if spread.max_loss > 0 else 0

        if decision.final_contracts > max_contracts:
            original = decision.final_contracts
            decision.final_contracts = max(0, max_contracts)
            if decision.final_contracts > 0:
                decision.decision = "modify"
                decision.override_reason = f"Reduced from {original} to {decision.final_contracts} contracts (2% risk limit)"
            else:
                decision.decision = "reject"
                decision.override_reason = "Cannot size position within 2% risk limit"

        # Override 4: Ensure approved trades have at least 1 contract
        if decision.decision == "approve" and decision.final_contracts < 1:
            decision.final_contracts = 1

        return decision

    def extract_decision(self, message: AgentMessage) -> FinalDecision | None:
        """Extract FinalDecision from an agent message."""
        if message.structured_data:
            return FinalDecision.from_dict(message.structured_data)
        return None
