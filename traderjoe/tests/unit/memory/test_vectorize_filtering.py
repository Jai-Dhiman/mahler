"""Tests for Vectorize namespace partitioning and metadata filtering."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.vectorize import EpisodicMemoryStore


class TestNamespacePartitioning:
    """Tests for namespace-based query partitioning."""

    @pytest.fixture
    def mock_bindings(self):
        """Create mock Cloudflare bindings."""
        vectorize = MagicMock()
        vectorize.upsert = AsyncMock()
        vectorize.query = AsyncMock()
        vectorize.getByIds = AsyncMock()

        ai = MagicMock()
        ai.run = AsyncMock(return_value=MagicMock(data=[[0.1] * 384]))

        d1 = MagicMock()
        d1.prepare = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock(
                run=AsyncMock(),
                first=AsyncMock(return_value=None),
                all=AsyncMock(return_value=MagicMock(results=[])),
            ))
        ))

        return vectorize, ai, d1

    @pytest.fixture
    def store(self, mock_bindings):
        """Create EpisodicMemoryStore with mocks."""
        vectorize, ai, d1 = mock_bindings
        return EpisodicMemoryStore(vectorize, ai, d1)

    async def test_store_memory_uses_namespace(self, store, mock_bindings):
        """Storing memory should use underlying-based namespace."""
        vectorize, ai, d1 = mock_bindings

        # Mock the agent message
        class MockMessage:
            def to_dict(self):
                return {"content": "test"}
            structured_data = None

        await store.store_memory(
            trade_id="trade-1",
            underlying="SPY",
            spread_type="bull_put",
            short_strike=450.0,
            long_strike=445.0,
            expiration="2024-02-15",
            analyst_messages=[MockMessage()],
            debate_messages=[],
            synthesis_message=None,
            market_regime="bull_low_vol",
        )

        # Verify upsert was called with correct namespace
        vectorize.upsert.assert_called_once()
        call_args = vectorize.upsert.call_args
        assert call_args.kwargs.get("namespace") == "spy-trades"

    async def test_store_memory_qqq_namespace(self, store, mock_bindings):
        """QQQ trades should use qqq-trades namespace."""
        vectorize, ai, d1 = mock_bindings

        class MockMessage:
            def to_dict(self):
                return {"content": "test"}
            structured_data = None

        await store.store_memory(
            trade_id="trade-2",
            underlying="QQQ",
            spread_type="bear_call",
            short_strike=400.0,
            long_strike=405.0,
            expiration="2024-02-15",
            analyst_messages=[MockMessage()],
            debate_messages=[],
            synthesis_message=None,
        )

        call_args = vectorize.upsert.call_args
        assert call_args.kwargs.get("namespace") == "qqq-trades"

    async def test_find_similar_uses_namespace(self, store, mock_bindings):
        """find_similar should query the correct namespace."""
        vectorize, ai, d1 = mock_bindings

        # Setup mock response
        vectorize.query.return_value = MagicMock(matches=[])

        await store.find_similar(
            underlying="IWM",
            spread_type="bull_put",
            market_regime="bear_high_vol",
        )

        # Verify query was called with namespace as keyword argument
        vectorize.query.assert_called_once()
        call_args = vectorize.query.call_args
        assert call_args.kwargs.get("namespace") == "iwm-trades"


class TestMetadataFiltering:
    """Tests for metadata filter construction."""

    @pytest.fixture
    def mock_bindings(self):
        """Create mock Cloudflare bindings."""
        vectorize = MagicMock()
        vectorize.upsert = AsyncMock()
        vectorize.query = AsyncMock(return_value=MagicMock(matches=[]))
        vectorize.getByIds = AsyncMock()

        ai = MagicMock()
        ai.run = AsyncMock(return_value=MagicMock(data=[[0.1] * 384]))

        d1 = MagicMock()
        d1.prepare = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock(
                run=AsyncMock(),
                first=AsyncMock(return_value=None),
                all=AsyncMock(return_value=MagicMock(results=[])),
            ))
        ))

        return vectorize, ai, d1

    @pytest.fixture
    def store(self, mock_bindings):
        """Create EpisodicMemoryStore with mocks."""
        vectorize, ai, d1 = mock_bindings
        return EpisodicMemoryStore(vectorize, ai, d1)

    async def test_market_regime_filter(self, store, mock_bindings):
        """Market regime should be included in filter."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
            market_regime="bull_high_vol",
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        assert query_options["filter"]["market_regime"] == {"$eq": "bull_high_vol"}

    async def test_spread_type_filter(self, store, mock_bindings):
        """spread_type should be filterable via filter_spread_type."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bear_call",
            filter_spread_type=True,
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        assert query_options["filter"]["spread_type"] == {"$eq": "bear_call"}

    async def test_layer_filter(self, store, mock_bindings):
        """layer_filter should use $in operator."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
            layer_filter=["intermediate", "deep"],
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        assert query_options["filter"]["memory_layer"] == {"$in": ["intermediate", "deep"]}

    async def test_critical_only_filter(self, store, mock_bindings):
        """include_critical_only should filter to critical events."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
            include_critical_only=True,
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        assert query_options["filter"]["critical_event"] == {"$eq": True}

    async def test_win_filter(self, store, mock_bindings):
        """filter_to_wins should filter to winning trades."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
            filter_to_wins=True,
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        assert query_options["filter"]["win"] == {"$eq": True}

    async def test_combined_filters(self, store, mock_bindings):
        """Multiple filters should be combined."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
            market_regime="bear_high_vol",
            filter_spread_type=True,
            include_critical_only=True,
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        filters = query_options["filter"]

        assert "market_regime" in filters
        assert "spread_type" in filters
        assert "critical_event" in filters

    async def test_return_metadata_indexed(self, store, mock_bindings):
        """returnMetadata should be 'indexed' for faster queries."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        assert query_options["returnMetadata"] == "indexed"

    async def test_no_filter_when_no_conditions(self, store, mock_bindings):
        """No filter should be set when no filter conditions provided."""
        vectorize, _, _ = mock_bindings

        await store.find_similar(
            underlying="SPY",
            spread_type="bull_put",
        )

        call_args = vectorize.query.call_args
        query_options = call_args.args[1]
        # Filter should be empty dict or not present
        assert query_options.get("filter", {}) == {}


class TestOutcomeMetadataUpdate:
    """Tests for updating win/loss metadata after trade closes."""

    @pytest.fixture
    def mock_bindings(self):
        """Create mock Cloudflare bindings."""
        vectorize = MagicMock()
        vectorize.upsert = AsyncMock()
        vectorize.query = AsyncMock()
        vectorize.getByIds = AsyncMock()

        ai = MagicMock()
        ai.run = AsyncMock(return_value=MagicMock(data=[[0.1] * 384]))

        d1 = MagicMock()
        d1.prepare = MagicMock(return_value=MagicMock(
            bind=MagicMock(return_value=MagicMock(
                run=AsyncMock(),
                first=AsyncMock(return_value=None),
                all=AsyncMock(return_value=MagicMock(results=[])),
            ))
        ))

        return vectorize, ai, d1

    @pytest.fixture
    def store(self, mock_bindings):
        """Create EpisodicMemoryStore with mocks."""
        vectorize, ai, d1 = mock_bindings
        return EpisodicMemoryStore(vectorize, ai, d1)

    async def test_update_outcome_sets_win_true(self, store, mock_bindings):
        """Profitable outcome should set win=True in metadata."""
        vectorize, _, d1 = mock_bindings

        # Setup mocks
        d1.prepare.return_value.bind.return_value.first = AsyncMock(
            return_value={"underlying": "SPY", "embedding_id": "episodic_123"}
        )
        vectorize.getByIds = AsyncMock(return_value=MagicMock(
            vectors=[MagicMock(
                values=[0.1] * 384,
                metadata={"memory_id": "123", "underlying": "SPY", "spread_type": "bull_put"},
            )]
        ))

        await store.update_actual_outcome(
            memory_id="123",
            actual_outcome={"profit_loss": 150.0, "pnl_percent": 5.0},
        )

        # Verify upsert was called with win=True
        upsert_call = vectorize.upsert.call_args
        upserted_data = upsert_call.args[0][0]
        assert upserted_data["metadata"]["win"] is True

    async def test_update_outcome_sets_win_false(self, store, mock_bindings):
        """Losing outcome should set win=False in metadata."""
        vectorize, _, d1 = mock_bindings

        # Setup mocks
        d1.prepare.return_value.bind.return_value.first = AsyncMock(
            return_value={"underlying": "QQQ", "embedding_id": "episodic_456"}
        )
        vectorize.getByIds = AsyncMock(return_value=MagicMock(
            vectors=[MagicMock(
                values=[0.1] * 384,
                metadata={"memory_id": "456", "underlying": "QQQ", "spread_type": "bear_call"},
            )]
        ))

        await store.update_actual_outcome(
            memory_id="456",
            actual_outcome={"profit_loss": -200.0, "pnl_percent": -8.0},
        )

        # Verify upsert was called with win=False
        upsert_call = vectorize.upsert.call_args
        upserted_data = upsert_call.args[0][0]
        assert upserted_data["metadata"]["win"] is False

    async def test_update_outcome_uses_correct_namespace(self, store, mock_bindings):
        """Outcome update should use correct namespace based on underlying."""
        vectorize, _, d1 = mock_bindings

        # Setup mocks
        d1.prepare.return_value.bind.return_value.first = AsyncMock(
            return_value={"underlying": "IWM", "embedding_id": "episodic_789"}
        )
        vectorize.getByIds = AsyncMock(return_value=MagicMock(
            vectors=[MagicMock(
                values=[0.1] * 384,
                metadata={"memory_id": "789"},
            )]
        ))

        await store.update_actual_outcome(
            memory_id="789",
            actual_outcome={"profit_loss": 50.0},
        )

        # Verify namespace
        upsert_call = vectorize.upsert.call_args
        assert upsert_call.kwargs.get("namespace") == "iwm-trades"
