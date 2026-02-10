"""Debate researcher agents for the multi-agent trading system.

Bull and Bear researchers argue for and against trade entry based on
analyst reports. They engage in multi-round debates to surface the
strongest arguments on each side.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from core.agents.base import (
    AgentContext,
    AgentMessage,
    DebateAgent,
    MessageType,
)
from core.ai.prompts import (
    BEAR_RESEARCHER_SYSTEM,
    BEAR_RESEARCHER_USER,
    BULL_RESEARCHER_SYSTEM,
    BULL_RESEARCHER_USER,
)

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient
    from core.ai.router import LLMRouter


class BullResearcher(DebateAgent):
    """Researcher that argues FOR entering the trade.

    Highlights favorable signals from analyst reports and builds
    a compelling case for trade entry. Counters bear arguments
    in multi-round debates.
    """

    def __init__(self, claude: ClaudeClient | None = None, *, router: LLMRouter | None = None):
        super().__init__(claude, agent_id="bull_researcher", router=router)

    @property
    def role(self) -> str:
        return "Bull Researcher arguing for trade entry"

    @property
    def system_prompt(self) -> str:
        return BULL_RESEARCHER_SYSTEM

    @property
    def perspective(self) -> Literal["bull", "bear"]:
        return "bull"

    async def argue(self, context: AgentContext, round_number: int) -> AgentMessage:
        """Produce a bull argument for this debate round."""
        spread = context.spread

        # Summarize analyst reports
        analyst_summaries = self._summarize_analysts(context)

        # Get opponent's last argument (if any)
        opponent_msg = context.get_last_opponent_message(self.agent_id)
        opponent_argument = ""
        if opponent_msg and round_number > 1:
            opponent_argument = f"\n**Bear's Previous Argument:**\n{opponent_msg.content}"
            if opponent_msg.structured_data:
                bear_points = opponent_msg.structured_data.get("key_arguments", [])
                if bear_points:
                    opponent_argument += f"\nKey points: {', '.join(bear_points)}"

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        prompt = BULL_RESEARCHER_USER.format(
            round_number=round_number,
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            credit=spread.credit,
            max_loss=spread.max_loss / 100,  # Per spread
            expiration=spread.expiration,
            dte=dte,
            analyst_summaries=analyst_summaries,
            opponent_argument=opponent_argument,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        # Build content from key arguments
        key_args = data.get("key_arguments", [])
        content = "; ".join(key_args) if key_args else "Trade entry recommended"

        return self._create_message(
            message_type=MessageType.ARGUMENT,
            content=content,
            structured_data=data,
            confidence=data.get("conviction", 0.5),
        )

    def _summarize_analysts(self, context: AgentContext) -> str:
        """Build summary of analyst reports for the prompt."""
        analyst_msgs = context.get_analyst_messages()

        if not analyst_msgs:
            return "No analyst reports available."

        summaries = []
        for msg in analyst_msgs:
            agent_name = msg.agent_id.replace("_", " ").title()
            content = msg.content[:200] if len(msg.content) > 200 else msg.content

            # Extract key structured data
            details = []
            if msg.structured_data:
                # IV Analyst
                if "iv_signal" in msg.structured_data:
                    details.append(f"IV Signal: {msg.structured_data['iv_signal']}")
                # Technical Analyst
                if "trend" in msg.structured_data:
                    details.append(f"Trend: {msg.structured_data['trend']}")
                if "short_strike_assessment" in msg.structured_data:
                    details.append(f"Strike: {msg.structured_data['short_strike_assessment']}")
                # Macro Analyst
                if "regime_assessment" in msg.structured_data:
                    details.append(f"Regime: {msg.structured_data['regime_assessment']}")
                if "event_risk_score" in msg.structured_data:
                    score = msg.structured_data['event_risk_score']
                    details.append(f"Event Risk: {score:.0%}")
                # Greeks Analyst
                if "portfolio_fit" in msg.structured_data:
                    details.append(f"Portfolio Fit: {msg.structured_data['portfolio_fit']}")
                if "position_size_recommendation" in msg.structured_data:
                    details.append(f"Size: {msg.structured_data['position_size_recommendation']}")

            detail_str = f" ({', '.join(details)})" if details else ""
            summaries.append(f"**{agent_name}** (confidence: {msg.confidence:.0%}){detail_str}: {content}")

        return "\n\n".join(summaries)


class BearResearcher(DebateAgent):
    """Researcher that argues AGAINST entering the trade.

    Highlights risks and unfavorable signals from analyst reports.
    Builds a compelling case for skipping the trade. Counters bull
    arguments in multi-round debates.
    """

    def __init__(self, claude: ClaudeClient | None = None, *, router: LLMRouter | None = None):
        super().__init__(claude, agent_id="bear_researcher", router=router)

    @property
    def role(self) -> str:
        return "Bear Researcher arguing against trade entry"

    @property
    def system_prompt(self) -> str:
        return BEAR_RESEARCHER_SYSTEM

    @property
    def perspective(self) -> Literal["bull", "bear"]:
        return "bear"

    async def argue(self, context: AgentContext, round_number: int) -> AgentMessage:
        """Produce a bear argument for this debate round."""
        spread = context.spread

        # Summarize analyst reports
        analyst_summaries = self._summarize_analysts(context)

        # Get opponent's last argument (if any)
        opponent_msg = context.get_last_opponent_message(self.agent_id)
        opponent_argument = ""
        if opponent_msg and round_number > 1:
            opponent_argument = f"\n**Bull's Previous Argument:**\n{opponent_msg.content}"
            if opponent_msg.structured_data:
                bull_points = opponent_msg.structured_data.get("key_arguments", [])
                if bull_points:
                    opponent_argument += f"\nKey points: {', '.join(bull_points)}"

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        prompt = BEAR_RESEARCHER_USER.format(
            round_number=round_number,
            underlying=spread.underlying,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            credit=spread.credit,
            max_loss=spread.max_loss / 100,  # Per spread
            expiration=spread.expiration,
            dte=dte,
            analyst_summaries=analyst_summaries,
            opponent_argument=opponent_argument,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        # Build content from key arguments
        key_args = data.get("key_arguments", [])
        content = "; ".join(key_args) if key_args else "Trade entry not recommended"

        return self._create_message(
            message_type=MessageType.ARGUMENT,
            content=content,
            structured_data=data,
            confidence=data.get("conviction", 0.5),
        )

    def _summarize_analysts(self, context: AgentContext) -> str:
        """Build summary of analyst reports for the prompt."""
        # Same implementation as BullResearcher - could be refactored to base class
        analyst_msgs = context.get_analyst_messages()

        if not analyst_msgs:
            return "No analyst reports available."

        summaries = []
        for msg in analyst_msgs:
            agent_name = msg.agent_id.replace("_", " ").title()
            content = msg.content[:200] if len(msg.content) > 200 else msg.content

            # Extract key structured data
            details = []
            if msg.structured_data:
                # IV Analyst
                if "iv_signal" in msg.structured_data:
                    details.append(f"IV Signal: {msg.structured_data['iv_signal']}")
                # Technical Analyst
                if "trend" in msg.structured_data:
                    details.append(f"Trend: {msg.structured_data['trend']}")
                if "short_strike_assessment" in msg.structured_data:
                    details.append(f"Strike: {msg.structured_data['short_strike_assessment']}")
                # Macro Analyst
                if "regime_assessment" in msg.structured_data:
                    details.append(f"Regime: {msg.structured_data['regime_assessment']}")
                if "event_risk_score" in msg.structured_data:
                    score = msg.structured_data['event_risk_score']
                    details.append(f"Event Risk: {score:.0%}")
                # Greeks Analyst
                if "portfolio_fit" in msg.structured_data:
                    details.append(f"Portfolio Fit: {msg.structured_data['portfolio_fit']}")
                if "position_size_recommendation" in msg.structured_data:
                    details.append(f"Size: {msg.structured_data['position_size_recommendation']}")

                # Also extract risks for bear focus
                risks = msg.structured_data.get("risks", [])
                if risks:
                    details.append(f"Risks: {', '.join(risks[:2])}")

            detail_str = f" ({', '.join(details)})" if details else ""
            summaries.append(f"**{agent_name}** (confidence: {msg.confidence:.0%}){detail_str}: {content}")

        return "\n\n".join(summaries)
