"""Tests for the AgentOrchestrator.

Tests the orchestration logic including:
- Agent registration
- Pipeline execution stages
- Result extraction
- Debate configuration
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agents.base import (
    AgentContext,
    AgentMessage,
    MarketData,
    MessageType,
    PortfolioContext,
)
from core.agents.orchestrator import (
    AgentOrchestrator,
    DebateConfig,
    PipelineResult,
    build_agent_context,
)
from core.types import CreditSpread, Greeks, OptionContract, SpreadType


@pytest.fixture
def mock_claude():
    """Mock Claude client."""
    claude = MagicMock()
    claude._request = AsyncMock(return_value='{"recommendation": "enter", "confidence": 0.7}')
    claude._parse_json_response = MagicMock(return_value={"recommendation": "enter", "confidence": 0.7})
    return claude


@pytest.fixture
def sample_spread():
    """Sample credit spread for testing."""
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
def sample_context(sample_spread):
    """Sample agent context for testing."""
    return AgentContext(
        spread=sample_spread,
        market_data=MarketData(
            underlying="SPY",
            underlying_price=475.0,
            current_vix=18.0,
        ),
        portfolio=PortfolioContext(
            account_equity=100000.0,
            buying_power=50000.0,
        ),
        playbook_rules=[],
        similar_trades=[],
    )


class TestAgentOrchestrator:
    """Test AgentOrchestrator class."""

    def test_init_with_defaults(self, mock_claude):
        """Test initialization with default debate config."""
        orchestrator = AgentOrchestrator(mock_claude)

        assert orchestrator.claude == mock_claude
        assert orchestrator.debate_config.max_rounds == 3
        assert orchestrator.debate_config.min_rounds == 1
        assert orchestrator._analysts == []
        assert orchestrator._bull_debater is None
        assert orchestrator._bear_debater is None
        assert orchestrator._facilitator is None

    def test_init_with_custom_config(self, mock_claude):
        """Test initialization with custom debate config."""
        config = DebateConfig(max_rounds=2, min_rounds=1, consensus_threshold=0.8)
        orchestrator = AgentOrchestrator(mock_claude, debate_config=config)

        assert orchestrator.debate_config.max_rounds == 2
        assert orchestrator.debate_config.consensus_threshold == 0.8

    def test_register_analyst(self, mock_claude):
        """Test analyst registration."""
        orchestrator = AgentOrchestrator(mock_claude)

        mock_analyst = MagicMock()
        mock_analyst.agent_id = "test_analyst"

        orchestrator.register_analyst(mock_analyst)

        assert len(orchestrator._analysts) == 1
        assert orchestrator._analysts[0] == mock_analyst

    def test_register_multiple_analysts(self, mock_claude):
        """Test registering multiple analysts."""
        orchestrator = AgentOrchestrator(mock_claude)

        for i in range(4):
            mock_analyst = MagicMock()
            mock_analyst.agent_id = f"analyst_{i}"
            orchestrator.register_analyst(mock_analyst)

        assert len(orchestrator._analysts) == 4

    def test_register_bull_debater(self, mock_claude):
        """Test bull debater registration."""
        orchestrator = AgentOrchestrator(mock_claude)

        mock_debater = MagicMock()
        mock_debater.perspective = "bull"

        orchestrator.register_debater(mock_debater, "bull")

        assert orchestrator._bull_debater == mock_debater

    def test_register_bear_debater(self, mock_claude):
        """Test bear debater registration."""
        orchestrator = AgentOrchestrator(mock_claude)

        mock_debater = MagicMock()
        mock_debater.perspective = "bear"

        orchestrator.register_debater(mock_debater, "bear")

        assert orchestrator._bear_debater == mock_debater

    def test_register_debater_invalid_perspective(self, mock_claude):
        """Test that invalid perspective raises error."""
        orchestrator = AgentOrchestrator(mock_claude)
        mock_debater = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            orchestrator.register_debater(mock_debater, "invalid")

        assert "Invalid perspective" in str(exc_info.value)

    def test_set_facilitator(self, mock_claude):
        """Test facilitator registration."""
        orchestrator = AgentOrchestrator(mock_claude)

        mock_facilitator = MagicMock()
        orchestrator.set_facilitator(mock_facilitator)

        assert orchestrator._facilitator == mock_facilitator


class TestPipelineExecution:
    """Test pipeline execution."""

    @pytest.mark.asyncio
    async def test_run_pipeline_no_agents(self, mock_claude, sample_context):
        """Test running pipeline with no registered agents."""
        orchestrator = AgentOrchestrator(mock_claude)

        result = await orchestrator.run_pipeline(sample_context)

        assert isinstance(result, PipelineResult)
        assert result.analyst_messages == []
        assert result.debate_messages == []

    @pytest.mark.asyncio
    async def test_run_analysts_parallel(self, mock_claude, sample_context):
        """Test that analysts run in parallel."""
        orchestrator = AgentOrchestrator(mock_claude)

        # Create mock analysts
        messages = []
        for i in range(3):
            mock_analyst = MagicMock()
            mock_analyst.agent_id = f"analyst_{i}"
            msg = AgentMessage(
                agent_id=f"analyst_{i}",
                timestamp=datetime.now(),
                message_type=MessageType.ANALYSIS,
                content=f"Analysis {i}",
                confidence=0.7,
            )
            mock_analyst.analyze = AsyncMock(return_value=msg)
            orchestrator.register_analyst(mock_analyst)
            messages.append(msg)

        result = await orchestrator.run_pipeline(sample_context)

        assert len(result.analyst_messages) == 3
        # Verify all analysts were called
        for analyst in orchestrator._analysts:
            analyst.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_debate_rounds(self, mock_claude, sample_context):
        """Test debate round execution."""
        config = DebateConfig(max_rounds=2, min_rounds=1)
        orchestrator = AgentOrchestrator(mock_claude, debate_config=config)

        # Mock bull researcher
        bull_msg = AgentMessage(
            agent_id="bull_researcher",
            timestamp=datetime.now(),
            message_type=MessageType.ARGUMENT,
            content="Bull argument",
            confidence=0.6,
        )
        mock_bull = MagicMock()
        mock_bull.agent_id = "bull_researcher"
        mock_bull.argue = AsyncMock(return_value=bull_msg)

        # Mock bear researcher
        bear_msg = AgentMessage(
            agent_id="bear_researcher",
            timestamp=datetime.now(),
            message_type=MessageType.ARGUMENT,
            content="Bear argument",
            confidence=0.5,
        )
        mock_bear = MagicMock()
        mock_bear.agent_id = "bear_researcher"
        mock_bear.argue = AsyncMock(return_value=bear_msg)

        orchestrator.register_debater(mock_bull, "bull")
        orchestrator.register_debater(mock_bear, "bear")

        result = await orchestrator.run_pipeline(sample_context)

        # Should have 4 debate messages (2 rounds x 2 sides)
        assert len(result.debate_messages) == 4
        # Verify argue was called for each round
        assert mock_bull.argue.call_count == 2
        assert mock_bear.argue.call_count == 2

    @pytest.mark.asyncio
    async def test_synthesis_stage(self, mock_claude, sample_context):
        """Test synthesis stage with facilitator."""
        orchestrator = AgentOrchestrator(mock_claude)

        synthesis_msg = AgentMessage(
            agent_id="facilitator",
            timestamp=datetime.now(),
            message_type=MessageType.SYNTHESIS,
            content="Final recommendation",
            confidence=0.75,
            structured_data={"recommendation": "enter", "position_size_multiplier": 1.0},
        )
        mock_facilitator = MagicMock()
        mock_facilitator.synthesize = AsyncMock(return_value=synthesis_msg)

        orchestrator.set_facilitator(mock_facilitator)

        result = await orchestrator.run_pipeline(sample_context)

        assert result.synthesis_message == synthesis_msg
        mock_facilitator.synthesize.assert_called_once()


class TestResultExtraction:
    """Test result extraction from pipeline."""

    def test_extract_from_synthesis(self, mock_claude):
        """Test extracting recommendation from synthesis message."""
        orchestrator = AgentOrchestrator(mock_claude)

        result = PipelineResult()
        result.synthesis_message = AgentMessage(
            agent_id="facilitator",
            timestamp=datetime.now(),
            message_type=MessageType.SYNTHESIS,
            content="Enter trade with conviction",
            confidence=0.8,
            structured_data={"recommendation": "enter"},
        )

        orchestrator._extract_recommendation(result)

        assert result.recommendation == "enter"
        assert result.confidence == 0.8
        assert "Enter trade" in result.thesis

    def test_extract_from_debate_fallback(self, mock_claude):
        """Test fallback to debate message when no synthesis."""
        orchestrator = AgentOrchestrator(mock_claude)

        result = PipelineResult()
        result.debate_messages = [
            AgentMessage(
                agent_id="bear_researcher",
                timestamp=datetime.now(),
                message_type=MessageType.ARGUMENT,
                content="Skip this trade",
                confidence=0.6,
                structured_data={"recommendation": "skip"},
            )
        ]

        orchestrator._extract_recommendation(result)

        assert result.recommendation == "skip"
        assert result.confidence == 0.6

    def test_extract_from_analysts_fallback(self, mock_claude):
        """Test fallback to analyst average when no debate."""
        orchestrator = AgentOrchestrator(mock_claude)

        result = PipelineResult()
        result.analyst_messages = [
            AgentMessage(
                agent_id="analyst_1",
                timestamp=datetime.now(),
                message_type=MessageType.ANALYSIS,
                content="Positive outlook",
                confidence=0.7,
            ),
            AgentMessage(
                agent_id="analyst_2",
                timestamp=datetime.now(),
                message_type=MessageType.ANALYSIS,
                content="Moderate outlook",
                confidence=0.5,
            ),
        ]

        orchestrator._extract_recommendation(result)

        # Average confidence: (0.7 + 0.5) / 2 = 0.6
        assert result.confidence == 0.6
        assert result.recommendation == "enter"  # > 0.5


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_duration_ms_calculation(self):
        """Test duration calculation."""
        result = PipelineResult()
        result.started_at = datetime(2024, 1, 15, 10, 0, 0)
        result.completed_at = datetime(2024, 1, 15, 10, 0, 5)  # 5 seconds

        assert result.duration_ms == 5000

    def test_duration_ms_none_when_incomplete(self):
        """Test duration is None when not completed."""
        result = PipelineResult()
        result.started_at = datetime.now()

        assert result.duration_ms is None

    def test_to_dict_serialization(self):
        """Test to_dict produces valid dict."""
        result = PipelineResult()
        result.started_at = datetime(2024, 1, 15, 10, 0, 0)
        result.completed_at = datetime(2024, 1, 15, 10, 0, 5)
        result.recommendation = "enter"
        result.confidence = 0.75
        result.thesis = "Test thesis"
        result.analyst_messages = [
            AgentMessage(
                agent_id="test",
                timestamp=datetime.now(),
                message_type=MessageType.ANALYSIS,
                content="Test",
                confidence=0.7,
            )
        ]

        data = result.to_dict()

        assert data["recommendation"] == "enter"
        assert data["confidence"] == 0.75
        assert data["thesis"] == "Test thesis"
        assert data["duration_ms"] == 5000
        assert len(data["analyst_messages"]) == 1


class TestBuildAgentContext:
    """Test build_agent_context helper function."""

    def test_builds_complete_context(self, sample_spread):
        """Test building context with all parameters."""
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
            price_bars=[{"close": 475.0}],
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

        assert context.spread == sample_spread
        assert context.market_data.underlying == "SPY"
        assert context.market_data.underlying_price == 475.0
        assert context.market_data.current_vix == 18.0
        assert context.market_data.regime == "bull_low_vol"
        assert context.portfolio.account_equity == 100000.0
        assert context.scan_type == "morning"

    def test_builds_minimal_context(self, sample_spread):
        """Test building context with minimal parameters."""
        context = build_agent_context(
            spread=sample_spread,
            underlying_price=475.0,
        )

        assert context.spread == sample_spread
        assert context.market_data.underlying == "SPY"
        assert context.portfolio.account_equity == 0.0
        assert context.similar_trades == []


class TestDebateConfig:
    """Test DebateConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DebateConfig()

        assert config.max_rounds == 3
        assert config.min_rounds == 1
        assert config.consensus_threshold == 0.7

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DebateConfig(
            max_rounds=5,
            min_rounds=2,
            consensus_threshold=0.8,
        )

        assert config.max_rounds == 5
        assert config.min_rounds == 2
        assert config.consensus_threshold == 0.8
