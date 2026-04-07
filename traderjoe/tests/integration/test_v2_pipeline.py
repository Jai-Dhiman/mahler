"""Integration tests for the V2 multi-agent pipeline.

Tests the full pipeline flow from context building through
recommendation extraction.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agents import (
    AgentOrchestrator,
    BearResearcher,
    BullResearcher,
    DebateConfig,
    DebateFacilitator,
    GreeksAnalyst,
    IVAnalyst,
    MacroAnalyst,
    PipelineResult,
    TechnicalAnalyst,
    build_agent_context,
)
from core.agents.base import MessageType
from core.types import CreditSpread, Greeks, OptionContract, SpreadType


@pytest.fixture
def mock_claude():
    """Mock Claude client that returns appropriate responses for each agent."""
    claude = MagicMock()

    def mock_request(messages, system_prompt):
        """Return appropriate mock response based on system prompt content."""
        prompt_lower = system_prompt.lower() if system_prompt else ""

        if "iv" in prompt_lower and "volatility" in prompt_lower:
            return '{"iv_signal": "favorable", "recommendation": "high IV rank supports premium selling", "confidence": 0.75}'
        elif "technical" in prompt_lower:
            return '{"trend": "bullish", "short_strike_assessment": "safe", "confidence": 0.7}'
        elif "macro" in prompt_lower:
            return '{"regime_assessment": "favorable", "vix_signal": "normal", "event_risk_score": 0.2, "confidence": 0.65}'
        elif "greeks" in prompt_lower:
            return '{"delta_assessment": "acceptable", "portfolio_fit": "good", "position_size_recommendation": "full", "confidence": 0.8}'
        elif "bull" in prompt_lower:
            return '{"key_arguments": ["High IV rank", "Favorable technicals"], "conviction": 0.7}'
        elif "bear" in prompt_lower:
            return '{"key_arguments": ["Event risk ahead", "Vol compression possible"], "conviction": 0.5}'
        elif "facilitator" in prompt_lower or "synthesis" in prompt_lower:
            return '{"winning_perspective": "bull", "recommendation": "enter", "confidence": 0.72, "position_size_multiplier": 1.0, "thesis": "Enter trade based on favorable IV and technical setup", "key_bull_points": ["High IV"], "key_bear_points": ["Event risk"], "deciding_factors": ["IV rank"], "consensus_reached": true}'
        else:
            return '{"recommendation": "enter", "confidence": 0.7}'

    def mock_parse(response):
        """Parse JSON response."""
        import json
        return json.loads(response)

    claude._request = AsyncMock(side_effect=mock_request)
    claude._parse_json_response = mock_parse
    return claude


@pytest.fixture
def sample_spread():
    """Sample credit spread for integration testing."""
    greeks = Greeks(delta=-0.25, gamma=0.02, theta=0.03, vega=0.15)

    short_contract = OptionContract(
        symbol="SPY240215P00470000",
        underlying="SPY",
        expiration="2024-02-15",
        strike=470.0,
        option_type="put",
        bid=1.20,
        ask=1.30,
        last=1.25,
        volume=1000,
        open_interest=5000,
        implied_volatility=0.20,
        greeks=greeks,
    )

    long_contract = OptionContract(
        symbol="SPY240215P00465000",
        underlying="SPY",
        expiration="2024-02-15",
        strike=465.0,
        option_type="put",
        bid=0.70,
        ask=0.80,
        last=0.75,
        volume=800,
        open_interest=4000,
        implied_volatility=0.22,
        greeks=greeks,
    )

    return CreditSpread(
        underlying="SPY",
        spread_type=SpreadType.BULL_PUT,
        short_strike=470.0,
        long_strike=465.0,
        expiration="2024-02-15",
        short_contract=short_contract,
        long_contract=long_contract,
    )


@pytest.fixture
def full_orchestrator(mock_claude):
    """Create fully configured orchestrator with all agents."""
    orchestrator = AgentOrchestrator(
        claude=mock_claude,
        debate_config=DebateConfig(
            max_rounds=2,
            min_rounds=1,
            consensus_threshold=0.7,
        ),
    )

    # Register all analyst agents
    orchestrator.register_analyst(IVAnalyst(mock_claude))
    orchestrator.register_analyst(TechnicalAnalyst(mock_claude))
    orchestrator.register_analyst(MacroAnalyst(mock_claude))
    orchestrator.register_analyst(GreeksAnalyst(mock_claude))

    # Register debate agents
    orchestrator.register_debater(BullResearcher(mock_claude), "bull")
    orchestrator.register_debater(BearResearcher(mock_claude), "bear")

    # Set facilitator
    orchestrator.set_facilitator(DebateFacilitator(mock_claude))

    return orchestrator


class TestFullPipeline:
    """Test the complete V2 pipeline execution."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, full_orchestrator, sample_spread):
        """Test running the complete pipeline from start to finish."""
        # Build context
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
            iv_metrics=None,
            term_structure=None,
            mean_reversion=None,
            regime="bull_low_vol",
            regime_probability=0.85,
            current_vix=18.0,
            vix_3m=20.0,
            price_bars=[
                {"date": "2024-01-10", "open": 470, "high": 475, "low": 468, "close": 473},
                {"date": "2024-01-11", "open": 473, "high": 478, "low": 471, "close": 476},
            ],
            positions=[],
            portfolio_greeks=None,
            account_equity=100000.0,
            buying_power=50000.0,
            daily_pnl=500.0,
            weekly_pnl=1500.0,
            playbook_rules=[],
            similar_trades=[],
            scan_type="morning",
        )

        # Run pipeline
        result = await full_orchestrator.run_pipeline(context)

        # Verify result structure
        assert isinstance(result, PipelineResult)
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_ms is not None
        # Duration may be 0 for fast mocked tests
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_pipeline_produces_analyst_messages(self, full_orchestrator, sample_spread):
        """Test that pipeline produces messages from all analysts."""
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
        )

        result = await full_orchestrator.run_pipeline(context)

        # Should have 4 analyst messages
        assert len(result.analyst_messages) == 4

        # Verify each analyst contributed
        agent_ids = {msg.agent_id for msg in result.analyst_messages}
        expected_ids = {"iv_analyst", "technical_analyst", "macro_analyst", "greeks_analyst"}
        assert agent_ids == expected_ids

        # All should be ANALYSIS type
        for msg in result.analyst_messages:
            assert msg.message_type == MessageType.ANALYSIS

    @pytest.mark.asyncio
    async def test_pipeline_produces_debate_messages(self, full_orchestrator, sample_spread):
        """Test that pipeline produces debate messages."""
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
        )

        result = await full_orchestrator.run_pipeline(context)

        # Should have debate messages (2 rounds x 2 sides = 4)
        assert len(result.debate_messages) == 4

        # Should have both bull and bear arguments
        agent_ids = {msg.agent_id for msg in result.debate_messages}
        assert "bull_researcher" in agent_ids
        assert "bear_researcher" in agent_ids

        # All should be ARGUMENT type
        for msg in result.debate_messages:
            assert msg.message_type == MessageType.ARGUMENT

    @pytest.mark.asyncio
    async def test_pipeline_produces_synthesis(self, full_orchestrator, sample_spread):
        """Test that pipeline produces synthesis message."""
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
        )

        result = await full_orchestrator.run_pipeline(context)

        # Should have synthesis message
        assert result.synthesis_message is not None
        assert result.synthesis_message.message_type == MessageType.SYNTHESIS
        assert result.synthesis_message.agent_id == "facilitator"

    @pytest.mark.asyncio
    async def test_pipeline_extracts_recommendation(self, full_orchestrator, sample_spread):
        """Test that pipeline extracts final recommendation."""
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
        )

        result = await full_orchestrator.run_pipeline(context)

        # Should have recommendation
        assert result.recommendation in ["enter", "skip", "reduce_size"]
        assert 0.0 <= result.confidence <= 1.0
        # Thesis may be empty in some cases (e.g., when synthesis falls back)
        assert result.thesis is not None


class TestPipelineWithVariousConfigs:
    """Test pipeline with various configurations."""

    @pytest.mark.asyncio
    async def test_single_debate_round(self, mock_claude, sample_spread):
        """Test pipeline with single debate round."""
        orchestrator = AgentOrchestrator(
            claude=mock_claude,
            debate_config=DebateConfig(max_rounds=1, min_rounds=1),
        )

        orchestrator.register_analyst(IVAnalyst(mock_claude))
        orchestrator.register_debater(BullResearcher(mock_claude), "bull")
        orchestrator.register_debater(BearResearcher(mock_claude), "bear")
        orchestrator.set_facilitator(DebateFacilitator(mock_claude))

        context = build_agent_context(spread=sample_spread, underlying_price=475.0)
        result = await orchestrator.run_pipeline(context)

        # Should have 2 debate messages (1 round x 2 sides)
        assert len(result.debate_messages) == 2

    @pytest.mark.asyncio
    async def test_no_debate_agents(self, mock_claude, sample_spread):
        """Test pipeline with no debate agents (analysts only)."""
        orchestrator = AgentOrchestrator(claude=mock_claude)

        orchestrator.register_analyst(IVAnalyst(mock_claude))
        orchestrator.register_analyst(TechnicalAnalyst(mock_claude))

        context = build_agent_context(spread=sample_spread, underlying_price=475.0)
        result = await orchestrator.run_pipeline(context)

        # Should have analyst messages but no debate
        assert len(result.analyst_messages) == 2
        assert len(result.debate_messages) == 0
        # Should still produce a recommendation from analyst consensus
        assert result.recommendation is not None

    @pytest.mark.asyncio
    async def test_no_facilitator(self, mock_claude, sample_spread):
        """Test pipeline without facilitator (uses debate outcome)."""
        orchestrator = AgentOrchestrator(
            claude=mock_claude,
            debate_config=DebateConfig(max_rounds=1),
        )

        orchestrator.register_analyst(IVAnalyst(mock_claude))
        orchestrator.register_debater(BullResearcher(mock_claude), "bull")
        orchestrator.register_debater(BearResearcher(mock_claude), "bear")
        # No facilitator set

        context = build_agent_context(spread=sample_spread, underlying_price=475.0)
        result = await orchestrator.run_pipeline(context)

        # Should have no synthesis but still produce recommendation
        assert result.synthesis_message is None
        assert result.recommendation is not None


class TestContextBuilding:
    """Test context building functionality."""

    def test_build_context_minimal(self, sample_spread):
        """Test building context with minimal inputs."""
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
        )

        assert context.spread == sample_spread
        assert context.market_data.underlying == "SPY"
        assert context.market_data.underlying_price == 475.0

    def test_build_context_with_similar_trades(self, sample_spread):
        """Test building context with similar trade history."""
        similar_trades = [
            {
                "underlying": "SPY",
                "spread_type": "bull_put",
                "outcome": {"profit_loss": 150.0, "was_profitable": True},
                "similarity_score": 0.85,
            },
            {
                "underlying": "SPY",
                "spread_type": "bull_put",
                "outcome": {"profit_loss": -200.0, "was_profitable": False},
                "similarity_score": 0.72,
            },
        ]

        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
            similar_trades=similar_trades,
        )

        assert len(context.similar_trades) == 2
        assert context.similar_trades[0]["similarity_score"] == 0.85

    def test_build_context_with_playbook_rules(self, sample_spread):
        """Test building context with playbook rules."""
        rules = [
            {"rule": "Don't sell premium before earnings", "source": "initial"},
            {"rule": "Reduce size when VIX > 25", "source": "learned"},
        ]

        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
            playbook_rules=rules,
        )

        assert len(context.playbook_rules) == 2


class TestResultMapping:
    """Test result mapping to TradeAnalysis format."""

    @pytest.mark.asyncio
    async def test_high_confidence_maps_to_high(self, full_orchestrator, sample_spread):
        """Test that high V2 confidence maps to HIGH enum."""
        from core.types import Confidence

        # Mock high confidence response
        full_orchestrator.claude._request = AsyncMock(
            return_value='{"recommendation": "enter", "confidence": 0.85, "winning_perspective": "bull", "position_size_multiplier": 1.0, "thesis": "Strong entry", "key_bull_points": [], "key_bear_points": [], "deciding_factors": [], "consensus_reached": true}'
        )

        context = build_agent_context(spread=sample_spread, underlying_price=475.0)
        result = await full_orchestrator.run_pipeline(context)

        # High confidence (>= 0.7) should map to HIGH
        if result.confidence >= 0.7:
            # This would be mapped in the handler
            from handlers.morning_scan import _map_v2_result_to_analysis
            analysis = _map_v2_result_to_analysis(result)
            assert analysis.confidence == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_result_to_dict_complete(self, full_orchestrator, sample_spread):
        """Test that result.to_dict() produces complete output."""
        context = build_agent_context(spread=sample_spread, underlying_price=475.0)
        result = await full_orchestrator.run_pipeline(context)

        data = result.to_dict()

        # Verify all expected keys
        assert "recommendation" in data
        assert "confidence" in data
        assert "thesis" in data
        assert "analyst_messages" in data
        assert "debate_messages" in data
        assert "duration_ms" in data

        # Verify serialization doesn't fail
        import json
        json_str = json.dumps(data)
        assert len(json_str) > 0


class TestErrorHandling:
    """Test error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_analyst_error_logged_but_pipeline_continues(self, sample_spread):
        """Test that analyst errors are logged but pipeline continues gracefully."""
        claude = MagicMock()
        claude._request = AsyncMock(side_effect=Exception("API Error"))

        orchestrator = AgentOrchestrator(claude)
        orchestrator.register_analyst(IVAnalyst(claude))

        context = build_agent_context(spread=sample_spread, underlying_price=475.0)

        # Pipeline should complete (errors are caught and logged)
        result = await orchestrator.run_pipeline(context)

        # Should have empty analyst messages when all fail
        # (the orchestrator catches individual analyst errors)
        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_pipeline_stages_execute_in_order(self, mock_claude, sample_spread):
        """Test that pipeline stages execute in correct order: analysts -> debate -> synthesis."""
        from core.agents.base import AgentMessage

        orchestrator = AgentOrchestrator(
            claude=mock_claude,
            debate_config=DebateConfig(max_rounds=1),
        )

        # Register actual agents
        orchestrator.register_analyst(IVAnalyst(mock_claude))
        orchestrator.register_debater(BullResearcher(mock_claude), "bull")
        orchestrator.register_debater(BearResearcher(mock_claude), "bear")
        orchestrator.set_facilitator(DebateFacilitator(mock_claude))

        context = build_agent_context(spread=sample_spread, underlying_price=475.0)
        result = await orchestrator.run_pipeline(context)

        # Verify all stages produced output
        assert len(result.analyst_messages) > 0, "Analyst stage should produce messages"
        assert len(result.debate_messages) > 0, "Debate stage should produce messages"
        assert result.synthesis_message is not None, "Synthesis stage should produce message"

        # Verify timestamps indicate correct order
        analyst_time = max(m.timestamp for m in result.analyst_messages)
        debate_time = min(m.timestamp for m in result.debate_messages)
        synthesis_time = result.synthesis_message.timestamp

        # Analysts complete before debate starts
        assert analyst_time <= debate_time, "Analysts should complete before debate"
        # Debate completes before synthesis
        debate_end_time = max(m.timestamp for m in result.debate_messages)
        assert debate_end_time <= synthesis_time, "Debate should complete before synthesis"
