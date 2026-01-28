"""Specialist analyst agents for the multi-agent trading system.

Each analyst focuses on a specific domain:
- IVAnalyst: Implied volatility, term structure, mean reversion
- TechnicalAnalyst: Price action, support/resistance, trend
- MacroAnalyst: VIX, market regime, event risk
- GreeksAnalyst: Delta, gamma, theta, vega, portfolio fit
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from core.agents.base import (
    AgentContext,
    AgentMessage,
    AnalystAgent,
    MessageType,
)
from core.ai.prompts import (
    GREEKS_ANALYST_SYSTEM,
    GREEKS_ANALYST_USER,
    IV_ANALYST_SYSTEM,
    IV_ANALYST_USER,
    MACRO_ANALYST_SYSTEM,
    MACRO_ANALYST_USER,
    TECHNICAL_ANALYST_SYSTEM,
    TECHNICAL_ANALYST_USER,
)

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient


class IVAnalyst(AnalystAgent):
    """Analyst specializing in implied volatility analysis.

    Examines:
    - IV Rank and IV Percentile
    - Term structure (contango vs backwardation)
    - Mean reversion signals
    - Historical IV context
    """

    def __init__(self, claude: ClaudeClient):
        super().__init__(claude, agent_id="iv_analyst")

    @property
    def role(self) -> str:
        return "IV Specialist analyzing volatility environment for premium selling"

    @property
    def system_prompt(self) -> str:
        return IV_ANALYST_SYSTEM

    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Analyze IV environment for the spread."""
        market = context.market_data
        iv = market.iv_metrics

        # Build prompt with available data
        term_structure_regime = "unknown"
        term_structure_ratio = 1.0
        term_structure_signal = "unknown"
        if market.term_structure:
            term_structure_regime = market.term_structure.regime.value
            term_structure_ratio = market.term_structure.ratio_30_90
            term_structure_signal = market.term_structure.signal

        mean_reversion_z = 0.0
        mean_reversion_mean = 0.0
        mean_reversion_signal = "hold"
        is_stationary = False
        if market.mean_reversion:
            mean_reversion_z = market.mean_reversion.z_score
            mean_reversion_mean = market.mean_reversion.long_term_mean
            mean_reversion_signal = market.mean_reversion.signal.value
            is_stationary = market.mean_reversion.is_stationary

        prompt = IV_ANALYST_USER.format(
            underlying=market.underlying,
            underlying_price=market.underlying_price,
            current_iv=iv.current_iv if iv else 0.20,
            iv_rank=iv.iv_rank if iv else 50.0,
            iv_percentile=iv.iv_percentile if iv else 50.0,
            iv_high=iv.iv_high if iv else 0.25,
            iv_low=iv.iv_low if iv else 0.15,
            term_structure_regime=term_structure_regime,
            term_structure_ratio=term_structure_ratio,
            term_structure_signal=term_structure_signal,
            mean_reversion_z=mean_reversion_z,
            mean_reversion_mean=mean_reversion_mean,
            mean_reversion_signal=mean_reversion_signal,
            is_stationary=is_stationary,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        return self._create_message(
            message_type=MessageType.ANALYSIS,
            content=data.get("recommendation", ""),
            structured_data=data,
            confidence=data.get("confidence", 0.5),
        )


class TechnicalAnalyst(AnalystAgent):
    """Analyst specializing in technical analysis.

    Examines:
    - Trend direction and strength
    - Support and resistance levels
    - Strike proximity to key levels
    - Recent price action
    """

    def __init__(self, claude: ClaudeClient):
        super().__init__(claude, agent_id="technical_analyst")

    @property
    def role(self) -> str:
        return "Technical Analyst assessing price action and key levels"

    @property
    def system_prompt(self) -> str:
        return TECHNICAL_ANALYST_SYSTEM

    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Analyze technical setup for the spread."""
        spread = context.spread
        market = context.market_data

        # Calculate indicators from price bars
        bars = market.price_bars or []
        sma_20, sma_50, high_20, low_20, atr = self._calculate_indicators(bars)

        # Format price bars for prompt (last 20)
        price_bars_str = self._format_price_bars(bars[-20:] if bars else [])

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        prompt = TECHNICAL_ANALYST_USER.format(
            underlying=market.underlying,
            underlying_price=market.underlying_price,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            short_strike=spread.short_strike,
            long_strike=spread.long_strike,
            expiration=spread.expiration,
            dte=dte,
            price_bars=price_bars_str,
            sma_20=sma_20,
            sma_50=sma_50,
            high_20=high_20,
            low_20=low_20,
            atr=atr,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        # Build content summary
        trend = data.get("trend", "unknown")
        assessment = data.get("short_strike_assessment", "unknown")
        content = f"Trend: {trend}, Short strike: {assessment}"

        return self._create_message(
            message_type=MessageType.ANALYSIS,
            content=content,
            structured_data=data,
            confidence=data.get("confidence", 0.5),
        )

    def _calculate_indicators(self, bars: list[dict]) -> tuple[float, float, float, float, float]:
        """Calculate technical indicators from price bars."""
        if not bars:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        closes = [b.get("close", 0) for b in bars]
        highs = [b.get("high", 0) for b in bars]
        lows = [b.get("low", 0) for b in bars]

        # SMAs
        sma_20 = sum(closes[-20:]) / min(20, len(closes)) if closes else 0
        sma_50 = sum(closes[-50:]) / min(50, len(closes)) if closes else 0

        # 20-day high/low
        high_20 = max(highs[-20:]) if highs else 0
        low_20 = min(lows[-20:]) if lows else 0

        # ATR (14-day)
        atr = self._calculate_atr(bars[-14:]) if len(bars) >= 14 else 0

        return sma_20, sma_50, high_20, low_20, atr

    def _calculate_atr(self, bars: list[dict]) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i].get("high", 0)
            low = bars[i].get("low", 0)
            prev_close = bars[i - 1].get("close", 0)

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def _format_price_bars(self, bars: list[dict]) -> str:
        """Format price bars for the prompt."""
        if not bars:
            return "No price data available"

        lines = []
        for bar in bars:
            date = bar.get("date", bar.get("t", ""))
            o = bar.get("open", bar.get("o", 0))
            h = bar.get("high", bar.get("h", 0))
            l = bar.get("low", bar.get("l", 0))
            c = bar.get("close", bar.get("c", 0))
            lines.append(f"{date}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}")

        return "\n".join(lines)


class MacroAnalyst(AnalystAgent):
    """Analyst specializing in macro and event risk analysis.

    Examines:
    - VIX level and regime
    - Market regime (bull/bear, high/low vol)
    - Upcoming events (Fed, earnings, economic data)
    - Calendar considerations
    """

    def __init__(self, claude: ClaudeClient):
        super().__init__(claude, agent_id="macro_analyst")

    @property
    def role(self) -> str:
        return "Macro Analyst assessing event risk and market regime"

    @property
    def system_prompt(self) -> str:
        return MACRO_ANALYST_SYSTEM

    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Analyze macro environment for the spread."""
        spread = context.spread
        market = context.market_data

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        # Format VIX 3M
        vix_3m_str = f"{market.vix_3m:.2f}" if market.vix_3m else "N/A"

        # Format regime probability
        regime_prob = market.regime_probability if market.regime_probability else 0.5

        prompt = MACRO_ANALYST_USER.format(
            underlying=market.underlying,
            underlying_price=market.underlying_price,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            expiration=spread.expiration,
            dte=dte,
            current_vix=market.current_vix or 20.0,
            vix_3m=vix_3m_str,
            regime=market.regime or "unknown",
            regime_probability=regime_prob,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        # Build content summary
        regime = data.get("regime_assessment", "unknown")
        vix_signal = data.get("vix_signal", "unknown")
        event_risk = data.get("event_risk_score", 0.0)
        content = f"Regime: {regime}, VIX: {vix_signal}, Event risk: {event_risk:.0%}"

        return self._create_message(
            message_type=MessageType.ANALYSIS,
            content=content,
            structured_data=data,
            confidence=data.get("confidence", 0.5),
        )


class GreeksAnalyst(AnalystAgent):
    """Analyst specializing in Greeks and portfolio risk analysis.

    Examines:
    - Spread Greeks (delta, gamma, theta, vega)
    - Second-order Greeks (vanna, volga)
    - Portfolio impact and fit
    - Correlation and concentration risk
    """

    def __init__(self, claude: ClaudeClient):
        super().__init__(claude, agent_id="greeks_analyst")

    @property
    def role(self) -> str:
        return "Greeks Analyst assessing risk metrics and portfolio fit"

    @property
    def system_prompt(self) -> str:
        return GREEKS_ANALYST_SYSTEM

    async def analyze(self, context: AgentContext) -> AgentMessage:
        """Analyze Greeks profile for the spread."""
        spread = context.spread
        market = context.market_data
        portfolio = context.portfolio

        # Calculate DTE
        exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d")
        dte = (exp_date - datetime.now()).days

        # Get spread Greeks from contracts
        short_greeks = spread.short_contract.greeks
        long_greeks = spread.long_contract.greeks

        # Calculate net spread Greeks
        short_delta = short_greeks.delta if short_greeks else 0.0
        long_delta = long_greeks.delta if long_greeks else 0.0
        spread_delta = short_delta - long_delta  # Net (we're short the spread)

        short_gamma = short_greeks.gamma if short_greeks else 0.0
        long_gamma = long_greeks.gamma if long_greeks else 0.0
        spread_gamma = short_gamma - long_gamma

        short_theta = short_greeks.theta if short_greeks else 0.0
        long_theta = long_greeks.theta if long_greeks else 0.0
        spread_theta = -(short_theta - long_theta)  # Positive for credit spreads

        short_vega = short_greeks.vega if short_greeks else 0.0
        long_vega = long_greeks.vega if long_greeks else 0.0
        spread_vega = short_vega - long_vega

        # Portfolio Greeks (if available)
        pg = portfolio.portfolio_greeks
        portfolio_delta = pg.delta if pg else 0.0
        portfolio_gamma = pg.gamma if pg else 0.0
        portfolio_theta = pg.theta if pg else 0.0
        portfolio_vega = pg.vega if pg else 0.0
        equity_risk = pg.equity_risk if pg else 0.0
        treasury_risk = pg.treasury_risk if pg else 0.0
        commodity_risk = pg.commodity_risk if pg else 0.0

        prompt = GREEKS_ANALYST_USER.format(
            underlying=market.underlying,
            underlying_price=market.underlying_price,
            spread_type=spread.spread_type.value.replace("_", " ").title(),
            short_strike=spread.short_strike,
            short_delta=short_delta,
            long_strike=spread.long_strike,
            long_delta=long_delta,
            expiration=spread.expiration,
            dte=dte,
            credit=spread.credit,
            max_loss=spread.max_loss / 100,  # Per spread
            spread_delta=spread_delta,
            spread_gamma=spread_gamma,
            spread_theta=spread_theta,
            spread_vega=spread_vega,
            spread_vanna=0.0,  # TODO: Pass from context if available
            spread_volga=0.0,
            portfolio_delta=portfolio_delta,
            portfolio_gamma=portfolio_gamma,
            portfolio_theta=portfolio_theta,
            portfolio_vega=portfolio_vega,
            equity_risk=equity_risk,
            treasury_risk=treasury_risk,
            commodity_risk=commodity_risk,
        )

        response = await self.claude._request(
            [{"role": "user", "content": prompt}],
            self.system_prompt,
        )

        data = self.claude._parse_json_response(response)

        # Build content summary
        delta_assessment = data.get("delta_assessment", "unknown")
        portfolio_fit = data.get("portfolio_fit", "unknown")
        size_rec = data.get("position_size_recommendation", "full")
        content = f"Delta: {delta_assessment}, Portfolio fit: {portfolio_fit}, Size: {size_rec}"

        return self._create_message(
            message_type=MessageType.ANALYSIS,
            content=content,
            structured_data=data,
            confidence=data.get("confidence", 0.5),
        )
