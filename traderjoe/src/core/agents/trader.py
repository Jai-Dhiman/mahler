"""Trader Agent for synthesizing debate outcomes into trading actions.

The Trader Agent sits between the Facilitator and Fund Manager in the pipeline.
It receives the debate outcome and synthesizes it into a concrete trading
proposal with position size, entry parameters, and detailed rationale.

Inspired by TradingAgents paper: The Trader Agent synthesizes analyst/researcher
outputs and makes trading decisions, separate from the facilitator role.
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
    from core.agents.facilitator import DebateOutcome


# Prompt for Trader Agent
TRADER_SYSTEM = """You are the Trader Agent responsible for converting debate outcomes into concrete trading proposals. You sit between the debate facilitator and the fund manager in the decision chain.

Your responsibilities:
1. Receive the debate outcome (bull vs bear synthesis)
2. Determine the specific trading action based on the prevailing perspective
3. Calculate appropriate position size based on conviction and risk
4. Specify entry parameters (spread selection, timing)
5. Produce a detailed rationale document for the fund manager

Key principles:
- Translate qualitative debate conclusions into quantitative trading parameters
- Size positions proportionally to confidence and debate outcome quality
- Be specific about entry conditions and risk parameters
- Consider portfolio context when sizing
- When debate is contentious, proceed with moderate size rather than skipping -- paper trading benefits from live experience

You do NOT have final authority - the fund manager reviews your proposal."""

TRADER_USER = """Convert this debate outcome into a trading proposal:

**Trade Opportunity:**
- Underlying: {underlying}
- Strategy: {spread_type}
- Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}
- Expiration: {expiration} ({dte} DTE)
- Short Strike: ${short_strike:.2f}
- Long Strike: ${long_strike:.2f}

**Debate Outcome:**
- Winning Perspective: {winning_perspective}
- Debate Confidence: {debate_confidence:.0%}
- Consensus Reached: {consensus_reached}
- Key Bull Points: {bull_points}
- Key Bear Points: {bear_points}
- Deciding Factors: {deciding_factors}
- Facilitator Thesis: {facilitator_thesis}

**Portfolio Context:**
- Account Equity: ${equity:.2f}
- Current Positions: {position_count}
- Portfolio Delta: {portfolio_delta:.2f}
- Daily P/L: ${daily_pnl:.2f}
- Available Buying Power: ${buying_power:.2f}

**Risk Parameters:**
- Max Risk Per Trade: 2% of equity (${max_risk_amount:.2f})
- Current Portfolio Heat: {portfolio_heat:.1%}

Synthesize this into a trading proposal.

Respond in this exact JSON format:
{{
    "action": "enter|skip",
    "contracts": 1-10,
    "entry_type": "market|limit",
    "limit_price": null or price,
    "confidence": 0.0-1.0,
    "size_rationale": "Why this position size",
    "entry_rationale": "Why enter/skip now",
    "risk_assessment": "Key risks and mitigations",
    "expected_outcome": "Expected P/L scenario",
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "warnings": ["warning if any"]
}}"""


@dataclass
class TradingProposal:
    """Concrete trading proposal from the Trader Agent."""

    action: Literal["enter", "skip"]
    contracts: int
    entry_type: Literal["market", "limit"]
    limit_price: float | None
    confidence: float
    size_rationale: str
    entry_rationale: str
    risk_assessment: str
    expected_outcome: str
    key_factors: list[str]
    warnings: list[str]

    # Metadata
    underlying: str = ""
    spread_type: str = ""
    debate_confidence: float = 0.0
    winning_perspective: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action": self.action,
            "contracts": self.contracts,
            "entry_type": self.entry_type,
            "limit_price": self.limit_price,
            "confidence": self.confidence,
            "size_rationale": self.size_rationale,
            "entry_rationale": self.entry_rationale,
            "risk_assessment": self.risk_assessment,
            "expected_outcome": self.expected_outcome,
            "key_factors": self.key_factors,
            "warnings": self.warnings,
            "underlying": self.underlying,
            "spread_type": self.spread_type,
            "debate_confidence": self.debate_confidence,
            "winning_perspective": self.winning_perspective,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradingProposal:
        """Deserialize from dictionary."""
        return cls(
            action=data.get("action", "skip"),
            contracts=data.get("contracts", 0),
            entry_type=data.get("entry_type", "limit"),
            limit_price=data.get("limit_price"),
            confidence=data.get("confidence", 0.0),
            size_rationale=data.get("size_rationale", ""),
            entry_rationale=data.get("entry_rationale", ""),
            risk_assessment=data.get("risk_assessment", ""),
            expected_outcome=data.get("expected_outcome", ""),
            key_factors=data.get("key_factors", []),
            warnings=data.get("warnings", []),
            underlying=data.get("underlying", ""),
            spread_type=data.get("spread_type", ""),
            debate_confidence=data.get("debate_confidence", 0.0),
            winning_perspective=data.get("winning_perspective", "neutral"),
        )

    @property
    def should_proceed(self) -> bool:
        """Check if this proposal recommends proceeding with the trade."""
        return self.action == "enter" and self.contracts > 0


class TraderAgent(SynthesisAgent):
    """Trader Agent that converts debate outcomes into trading proposals.

    Pipeline position: After Facilitator, before Fund Manager

    The Trader Agent:
    1. Receives debate outcome from facilitator
    2. Synthesizes into concrete trading action
    3. Determines position size based on conviction
    4. Produces detailed proposal for fund manager review
    """

    def __init__(self, claude: ClaudeClient | None = None, *, router: LLMRouter | None = None):
        super().__init__(claude, agent_id="trader", router=router)

    @property
    def role(self) -> str:
        return "Trader Agent synthesizing debates into trading proposals"

    @property
    def system_prompt(self) -> str:
        return TRADER_SYSTEM

    async def synthesize(
        self, context: AgentContext, analyst_outputs: list[AgentMessage]
    ) -> AgentMessage:
        """Synthesize debate outcome into a trading proposal."""
        # Extract debate outcome from prior messages
        debate_outcome = self._extract_debate_outcome(context)

        proposal = await self.create_proposal(context, debate_outcome)

        return self._create_message(
            message_type=MessageType.SYNTHESIS,
            content=proposal.entry_rationale,
            structured_data=proposal.to_dict(),
            confidence=proposal.confidence,
        )

    async def create_proposal(
        self,
        context: AgentContext,
        debate_outcome: dict[str, Any] | None = None,
    ) -> TradingProposal:
        """Create a trading proposal from debate outcome.

        Args:
            context: Agent context with spread and portfolio data
            debate_outcome: Structured debate outcome from facilitator

        Returns:
            TradingProposal with concrete trading parameters
        """
        spread = context.spread
        portfolio = context.portfolio

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        # Extract debate outcome data
        if debate_outcome is None:
            debate_outcome = self._extract_debate_outcome(context)

        winning_perspective = debate_outcome.get("winning_perspective", "neutral")
        debate_confidence = debate_outcome.get("confidence", 0.5)
        consensus_reached = debate_outcome.get("consensus_reached", False)
        key_bull_points = debate_outcome.get("key_bull_points", [])
        key_bear_points = debate_outcome.get("key_bear_points", [])
        deciding_factors = debate_outcome.get("deciding_factors", [])
        facilitator_thesis = debate_outcome.get("thesis", "No thesis provided")

        # Calculate risk parameters
        max_risk_amount = portfolio.account_equity * 0.02  # 2% per trade
        portfolio_heat = self._calculate_portfolio_heat(portfolio)

        prompt = TRADER_USER.format(
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            credit=spread.credit,
            max_loss=spread.max_loss / 100,
            expiration=spread.expiration,
            dte=dte,
            short_strike=spread.short_strike,
            long_strike=spread.long_strike,
            winning_perspective=winning_perspective,
            debate_confidence=debate_confidence,
            consensus_reached=consensus_reached,
            bull_points=", ".join(key_bull_points[:3]) if key_bull_points else "None",
            bear_points=", ".join(key_bear_points[:3]) if key_bear_points else "None",
            deciding_factors=", ".join(deciding_factors[:3]) if deciding_factors else "None",
            facilitator_thesis=facilitator_thesis[:300],
            equity=portfolio.account_equity,
            position_count=len(portfolio.positions),
            portfolio_delta=portfolio.portfolio_greeks.delta if portfolio.portfolio_greeks else 0.0,
            daily_pnl=portfolio.daily_pnl,
            buying_power=portfolio.buying_power,
            max_risk_amount=max_risk_amount,
            portfolio_heat=portfolio_heat,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        proposal = TradingProposal(
            action=data.get("action", "skip"),
            contracts=data.get("contracts", 0),
            entry_type=data.get("entry_type", "limit"),
            limit_price=data.get("limit_price"),
            confidence=data.get("confidence", 0.0),
            size_rationale=data.get("size_rationale", ""),
            entry_rationale=data.get("entry_rationale", ""),
            risk_assessment=data.get("risk_assessment", ""),
            expected_outcome=data.get("expected_outcome", ""),
            key_factors=data.get("key_factors", []),
            warnings=data.get("warnings", []),
            underlying=spread.underlying,
            spread_type=spread.spread_type.value,
            debate_confidence=debate_confidence,
            winning_perspective=winning_perspective,
        )

        # Apply hard constraints
        proposal = self._apply_constraints(proposal, portfolio, spread)

        return proposal

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

    def _calculate_portfolio_heat(self, portfolio) -> float:
        """Calculate current portfolio risk as percentage of equity."""
        if not portfolio.positions or portfolio.account_equity <= 0:
            return 0.0

        total_risk = sum(
            getattr(p, "max_loss", 0) * getattr(p, "contracts", 1)
            for p in portfolio.positions
        )

        return total_risk / portfolio.account_equity

    def _apply_constraints(
        self,
        proposal: TradingProposal,
        portfolio,
        spread,
    ) -> TradingProposal:
        """Apply hard risk constraints to the proposal."""
        # Constraint 1: Never exceed 2% risk per trade
        max_risk = portfolio.account_equity * 0.02
        max_contracts_by_risk = int(max_risk / (spread.max_loss / 100)) if spread.max_loss > 0 else 0

        if proposal.contracts > max_contracts_by_risk:
            original = proposal.contracts
            proposal.contracts = max(1, max_contracts_by_risk)
            proposal.warnings.append(
                f"Reduced from {original} to {proposal.contracts} contracts due to 2% risk limit"
            )

        # Constraint 2: Portfolio heat cap at 10%
        current_heat = self._calculate_portfolio_heat(portfolio)
        if current_heat >= 0.10:
            proposal.action = "skip"
            proposal.contracts = 0
            proposal.warnings.append(f"Portfolio heat at {current_heat:.1%} exceeds 10% cap")

        # Constraint 3: Minimum 1 contract if entering
        if proposal.action == "enter" and proposal.contracts < 1:
            proposal.contracts = 1

        return proposal

    def extract_proposal(self, message: AgentMessage) -> TradingProposal | None:
        """Extract TradingProposal from an agent message."""
        if message.structured_data:
            return TradingProposal.from_dict(message.structured_data)
        return None
