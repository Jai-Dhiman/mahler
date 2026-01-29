"""Fund Manager Agent for final trade approval.

The Fund Manager is the final approval authority in the trading pipeline.
It reviews the Trader's proposal along with risk team deliberation,
and can approve, reject, or modify the proposed trade.

Inspired by TradingAgents paper: The Fund Manager reviews risk discussion
and approves/modifies trader decisions with final authority.
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
    from core.agents.trader import TradingProposal
    from core.risk.three_perspective import ThreePerspectiveResult


# Prompt for Fund Manager Agent
FUND_MANAGER_SYSTEM = """You are the Fund Manager with final approval authority over all trades. You are the last line of defense before capital is deployed.

Your responsibilities:
1. Review the Trader's proposal with a critical eye
2. Consider the risk team's deliberation and concerns
3. Make the final approve/reject/modify decision
4. Override the Trader if risk assessment demands it

Decision framework:
- APPROVE: Trade meets all criteria, risk is acceptable, proceed as proposed
- MODIFY: Trade has merit but needs adjustment (typically smaller size)
- REJECT: Risk outweighs reward, skip this trade entirely

Key principles:
- Preservation of capital is paramount
- When in doubt, reduce size or reject
- Override trader if risk deliberation raises serious concerns
- Consider portfolio-level impact, not just individual trade merit
- Be skeptical of high-conviction proposals during elevated VIX

You have override authority - the Trader's recommendation is advisory only."""

FUND_MANAGER_USER = """Review this trading proposal for final approval:

**Trade Opportunity:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)

**Trader's Proposal:**
- Action: {trader_action}
- Contracts: {trader_contracts}
- Entry Type: {entry_type}
- Confidence: {trader_confidence:.0%}
- Size Rationale: {size_rationale}
- Entry Rationale: {entry_rationale}
- Risk Assessment: {risk_assessment}
- Warnings: {trader_warnings}

**Risk Team Deliberation:**
{risk_deliberation}

**Portfolio Context:**
- Account Equity: ${equity:.2f}
- Current Positions: {position_count}
- Portfolio Heat: {portfolio_heat:.1%}
- Daily P/L: ${daily_pnl:.2f}

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
    """Fund Manager Agent with final approval authority.

    Pipeline position: Final stage - after Trader and Risk Deliberation

    The Fund Manager:
    1. Reviews Trader's proposal
    2. Considers risk team deliberation
    3. Makes final approve/modify/reject decision
    4. Has override authority for special cases
    """

    def __init__(self, claude: ClaudeClient):
        super().__init__(claude, agent_id="fund_manager")

    @property
    def role(self) -> str:
        return "Fund Manager with final trade approval authority"

    @property
    def system_prompt(self) -> str:
        return FUND_MANAGER_SYSTEM

    async def synthesize(
        self, context: AgentContext, analyst_outputs: list[AgentMessage]
    ) -> AgentMessage:
        """Review and make final decision on trading proposal."""
        # Extract trader proposal from prior messages
        trader_proposal = self._extract_trader_proposal(context)
        risk_deliberation = self._extract_risk_deliberation(context)

        decision = await self.review_and_approve(
            context=context,
            proposal=trader_proposal,
            risk_deliberation=risk_deliberation,
        )

        return self._create_message(
            message_type=MessageType.DECISION,
            content=decision.final_thesis,
            structured_data=decision.to_dict(),
            confidence=decision.confidence,
        )

    async def review_and_approve(
        self,
        context: AgentContext,
        proposal: dict[str, Any] | None = None,
        risk_deliberation: dict[str, Any] | None = None,
        current_vix: float | None = None,
    ) -> FinalDecision:
        """Review trader proposal and risk deliberation, make final decision.

        Args:
            context: Agent context with spread and portfolio data
            proposal: Trader's proposal (extracted from context if not provided)
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

        # Extract proposal if not provided
        if proposal is None:
            proposal = self._extract_trader_proposal(context)

        # Extract risk deliberation if not provided
        if risk_deliberation is None:
            risk_deliberation = self._extract_risk_deliberation(context)

        # Get VIX from context if not provided
        if current_vix is None:
            current_vix = market.current_vix or 20.0

        # Format risk deliberation for prompt
        risk_deliberation_str = self._format_risk_deliberation(risk_deliberation)

        # Calculate portfolio heat
        portfolio_heat = self._calculate_portfolio_heat(portfolio)

        prompt = FUND_MANAGER_USER.format(
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            credit=spread.credit,
            max_loss=spread.max_loss / 100,
            expiration=spread.expiration,
            dte=dte,
            trader_action=proposal.get("action", "skip"),
            trader_contracts=proposal.get("contracts", 0),
            entry_type=proposal.get("entry_type", "limit"),
            trader_confidence=proposal.get("confidence", 0.0),
            size_rationale=proposal.get("size_rationale", "Not provided"),
            entry_rationale=proposal.get("entry_rationale", "Not provided"),
            risk_assessment=proposal.get("risk_assessment", "Not provided"),
            trader_warnings=", ".join(proposal.get("warnings", [])) or "None",
            risk_deliberation=risk_deliberation_str,
            equity=portfolio.account_equity,
            position_count=len(portfolio.positions),
            portfolio_heat=portfolio_heat,
            daily_pnl=portfolio.daily_pnl,
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
            original_contracts=proposal.get("contracts", 0),
            original_action=proposal.get("action", "skip"),
        )

        # Apply hard override rules
        decision = self._apply_overrides(decision, portfolio, spread, current_vix)

        return decision

    def _extract_trader_proposal(self, context: AgentContext) -> dict[str, Any]:
        """Extract trader proposal from prior messages."""
        # Look for trader's synthesis message
        for msg in reversed(context.prior_messages):
            if msg.agent_id == "trader" and msg.structured_data:
                return msg.structured_data

        # Fallback: empty proposal
        return {
            "action": "skip",
            "contracts": 0,
            "entry_type": "limit",
            "confidence": 0.0,
            "size_rationale": "No proposal received",
            "entry_rationale": "No proposal received",
            "risk_assessment": "No proposal received",
            "warnings": ["No trader proposal found"],
        }

    def _extract_risk_deliberation(self, context: AgentContext) -> dict[str, Any] | None:
        """Extract risk deliberation from prior messages."""
        # Look for risk deliberation message
        for msg in reversed(context.prior_messages):
            if "risk" in msg.agent_id.lower() and msg.structured_data:
                # Could be from risk deliberation facilitator or three-perspective
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
