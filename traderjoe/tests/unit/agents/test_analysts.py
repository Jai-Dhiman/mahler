"""Tests for analyst agents.

Tests the individual analyst agents:
- IVAnalyst
- TechnicalAnalyst
- MacroAnalyst
- GreeksAnalyst
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
from core.agents.analysts import (
    GreeksAnalyst,
    IVAnalyst,
    MacroAnalyst,
    TechnicalAnalyst,
)
from core.types import CreditSpread, Greeks, OptionContract, SpreadType


@pytest.fixture
def mock_claude():
    """Mock Claude client with JSON response."""
    claude = MagicMock()
    claude._request = AsyncMock(return_value='{"recommendation": "enter", "confidence": 0.7}')
    claude._parse_json_response = MagicMock(return_value={
        "recommendation": "Favorable IV environment for premium selling",
        "iv_signal": "high_rank",
        "confidence": 0.75,
    })
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
    """Sample agent context with IV metrics."""
    # Mock IV metrics
    iv_metrics = MagicMock()
    iv_metrics.current_iv = 0.20
    iv_metrics.iv_rank = 65.0
    iv_metrics.iv_percentile = 70.0
    iv_metrics.iv_high = 0.30
    iv_metrics.iv_low = 0.15

    return AgentContext(
        spread=sample_spread,
        market_data=MarketData(
            underlying="SPY",
            underlying_price=475.0,
            current_vix=18.0,
            vix_3m=20.0,
            iv_metrics=iv_metrics,
            regime="bull_low_vol",
            regime_probability=0.85,
        ),
        portfolio=PortfolioContext(
            account_equity=100000.0,
            buying_power=50000.0,
        ),
        playbook_rules=[],
        similar_trades=[],
    )


class TestIVAnalyst:
    """Test IVAnalyst class."""

    def test_init(self, mock_claude):
        """Test initialization."""
        analyst = IVAnalyst(mock_claude)

        assert analyst.agent_id == "iv_analyst"
        assert analyst.claude == mock_claude

    def test_role_description(self, mock_claude):
        """Test role property."""
        analyst = IVAnalyst(mock_claude)

        assert "IV" in analyst.role
        assert "volatility" in analyst.role.lower()

    def test_system_prompt_exists(self, mock_claude):
        """Test system prompt is defined."""
        analyst = IVAnalyst(mock_claude)

        assert analyst.system_prompt is not None
        assert len(analyst.system_prompt) > 100

    @pytest.mark.asyncio
    async def test_analyze_returns_message(self, mock_claude, sample_context):
        """Test analyze returns AgentMessage."""
        analyst = IVAnalyst(mock_claude)

        message = await analyst.analyze(sample_context)

        assert isinstance(message, AgentMessage)
        assert message.agent_id == "iv_analyst"
        assert message.message_type == MessageType.ANALYSIS

    @pytest.mark.asyncio
    async def test_analyze_includes_confidence(self, mock_claude, sample_context):
        """Test analyze includes confidence score."""
        analyst = IVAnalyst(mock_claude)

        message = await analyst.analyze(sample_context)

        assert 0.0 <= message.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_includes_structured_data(self, mock_claude, sample_context):
        """Test analyze includes structured data."""
        analyst = IVAnalyst(mock_claude)

        message = await analyst.analyze(sample_context)

        assert message.structured_data is not None
        assert "iv_signal" in message.structured_data


class TestTechnicalAnalyst:
    """Test TechnicalAnalyst class."""

    def test_init(self, mock_claude):
        """Test initialization."""
        analyst = TechnicalAnalyst(mock_claude)

        assert analyst.agent_id == "technical_analyst"

    def test_role_description(self, mock_claude):
        """Test role property."""
        analyst = TechnicalAnalyst(mock_claude)

        assert "Technical" in analyst.role

    @pytest.mark.asyncio
    async def test_analyze_with_price_bars(self, mock_claude, sample_context):
        """Test analyze with price bar data."""
        # Add price bars to context
        sample_context.market_data.price_bars = [
            {"date": "2024-01-10", "open": 470, "high": 475, "low": 468, "close": 473},
            {"date": "2024-01-11", "open": 473, "high": 478, "low": 471, "close": 476},
        ]

        mock_claude._parse_json_response = MagicMock(return_value={
            "trend": "bullish",
            "short_strike_assessment": "safe",
            "confidence": 0.7,
        })

        analyst = TechnicalAnalyst(mock_claude)
        message = await analyst.analyze(sample_context)

        assert isinstance(message, AgentMessage)
        assert message.message_type == MessageType.ANALYSIS

    @pytest.mark.asyncio
    async def test_analyze_without_price_bars(self, mock_claude, sample_context):
        """Test analyze handles missing price bars gracefully."""
        sample_context.market_data.price_bars = None

        mock_claude._parse_json_response = MagicMock(return_value={
            "trend": "unknown",
            "short_strike_assessment": "unknown",
            "confidence": 0.5,
        })

        analyst = TechnicalAnalyst(mock_claude)
        message = await analyst.analyze(sample_context)

        assert isinstance(message, AgentMessage)

    def test_calculate_indicators(self, mock_claude):
        """Test indicator calculation helper."""
        analyst = TechnicalAnalyst(mock_claude)

        bars = [
            {"open": 450, "high": 455, "low": 445, "close": 452},
            {"open": 452, "high": 458, "low": 450, "close": 456},
            {"open": 456, "high": 460, "low": 454, "close": 458},
        ]

        sma_20, sma_50, high_20, low_20, atr = analyst._calculate_indicators(bars)

        assert sma_20 > 0
        assert high_20 >= low_20

    def test_calculate_indicators_empty_bars(self, mock_claude):
        """Test indicator calculation with empty bars."""
        analyst = TechnicalAnalyst(mock_claude)

        sma_20, sma_50, high_20, low_20, atr = analyst._calculate_indicators([])

        assert sma_20 == 0.0
        assert atr == 0.0


class TestMacroAnalyst:
    """Test MacroAnalyst class."""

    def test_init(self, mock_claude):
        """Test initialization."""
        analyst = MacroAnalyst(mock_claude)

        assert analyst.agent_id == "macro_analyst"

    def test_role_description(self, mock_claude):
        """Test role property."""
        analyst = MacroAnalyst(mock_claude)

        assert "Macro" in analyst.role

    @pytest.mark.asyncio
    async def test_analyze_includes_vix_assessment(self, mock_claude, sample_context):
        """Test analyze assesses VIX environment."""
        mock_claude._parse_json_response = MagicMock(return_value={
            "regime_assessment": "favorable",
            "vix_signal": "normal",
            "event_risk_score": 0.2,
            "confidence": 0.7,
        })

        analyst = MacroAnalyst(mock_claude)
        message = await analyst.analyze(sample_context)

        assert isinstance(message, AgentMessage)
        assert "vix_signal" in message.structured_data or "regime_assessment" in message.structured_data


class TestGreeksAnalyst:
    """Test GreeksAnalyst class."""

    def test_init(self, mock_claude):
        """Test initialization."""
        analyst = GreeksAnalyst(mock_claude)

        assert analyst.agent_id == "greeks_analyst"

    def test_role_description(self, mock_claude):
        """Test role property."""
        analyst = GreeksAnalyst(mock_claude)

        assert "Greeks" in analyst.role

    @pytest.mark.asyncio
    async def test_analyze_calculates_spread_greeks(self, mock_claude, sample_context):
        """Test analyze calculates net spread Greeks."""
        mock_claude._parse_json_response = MagicMock(return_value={
            "delta_assessment": "acceptable",
            "portfolio_fit": "good",
            "position_size_recommendation": "full",
            "confidence": 0.75,
        })

        analyst = GreeksAnalyst(mock_claude)
        message = await analyst.analyze(sample_context)

        assert isinstance(message, AgentMessage)
        assert "portfolio_fit" in message.structured_data

    @pytest.mark.asyncio
    async def test_analyze_handles_missing_greeks(self, mock_claude, sample_context):
        """Test analyze handles contracts without Greeks."""
        # Remove Greeks from contracts
        sample_context.spread.short_contract.greeks = None
        sample_context.spread.long_contract.greeks = None

        mock_claude._parse_json_response = MagicMock(return_value={
            "delta_assessment": "unknown",
            "portfolio_fit": "unknown",
            "position_size_recommendation": "reduced",
            "confidence": 0.5,
        })

        analyst = GreeksAnalyst(mock_claude)
        message = await analyst.analyze(sample_context)

        assert isinstance(message, AgentMessage)


class TestAnalystIntegration:
    """Integration tests for analyst agents working together."""

    @pytest.mark.asyncio
    async def test_all_analysts_produce_valid_messages(self, mock_claude, sample_context):
        """Test all analyst types produce valid messages."""
        analysts = [
            IVAnalyst(mock_claude),
            TechnicalAnalyst(mock_claude),
            MacroAnalyst(mock_claude),
            GreeksAnalyst(mock_claude),
        ]

        for analyst in analysts:
            message = await analyst.analyze(sample_context)

            assert isinstance(message, AgentMessage)
            assert message.message_type == MessageType.ANALYSIS
            assert message.agent_id is not None
            assert message.timestamp is not None
            assert 0.0 <= message.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analysts_have_unique_ids(self, mock_claude):
        """Test all analysts have unique agent IDs."""
        analysts = [
            IVAnalyst(mock_claude),
            TechnicalAnalyst(mock_claude),
            MacroAnalyst(mock_claude),
            GreeksAnalyst(mock_claude),
        ]

        agent_ids = [a.agent_id for a in analysts]
        assert len(agent_ids) == len(set(agent_ids)), "Agent IDs should be unique"
