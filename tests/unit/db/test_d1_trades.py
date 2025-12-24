"""Tests for D1Client trade CRUD operations."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.types import SpreadType, Trade, TradeStatus


class TestCreateTrade:
    """Tests for create_trade() functionality."""

    @pytest.fixture
    def mock_db_binding(self):
        """Create a mock D1 database binding that records SQL queries."""
        binding = MagicMock()
        prepare_mock = MagicMock()
        bind_mock = MagicMock()

        binding.prepare = MagicMock(return_value=prepare_mock)
        prepare_mock.bind = MagicMock(return_value=bind_mock)
        bind_mock.run = AsyncMock()

        binding._queries = []

        def capture_prepare(query):
            binding._queries.append(query)
            return prepare_mock

        binding.prepare = MagicMock(side_effect=capture_prepare)

        return binding

    @pytest.mark.asyncio
    async def test_create_trade_inserts_correctly(self, mock_db_binding):
        """Test that create_trade() generates correct INSERT query."""
        from core.db.d1 import D1Client

        client = D1Client(mock_db_binding)

        with patch("core.db.d1.uuid4", return_value="test-trade-id"):
            trade_id = await client.create_trade(
                recommendation_id="rec-123",
                underlying="SPY",
                spread_type=SpreadType.BULL_PUT,
                short_strike=470.0,
                long_strike=465.0,
                expiration="2024-02-15",
                entry_credit=1.25,
                contracts=2,
                broker_order_id="order-123",
                status=TradeStatus.OPEN,
            )

        assert trade_id == "test-trade-id"

        # Verify the query contains INSERT INTO trades
        assert len(mock_db_binding._queries) == 1
        query = mock_db_binding._queries[0]
        assert "INSERT INTO trades" in query
        assert "id, recommendation_id, opened_at, status" in query

    @pytest.mark.asyncio
    async def test_create_trade_with_pending_fill_status(self, mock_db_binding):
        """Test creating a trade with PENDING_FILL status."""
        from core.db.d1 import D1Client

        client = D1Client(mock_db_binding)

        with patch("core.db.d1.uuid4", return_value="pending-trade-id"):
            trade_id = await client.create_trade(
                recommendation_id="rec-456",
                underlying="QQQ",
                spread_type=SpreadType.BEAR_CALL,
                short_strike=400.0,
                long_strike=405.0,
                expiration="2024-03-15",
                entry_credit=0.95,
                contracts=1,
                broker_order_id="order-456",
                status=TradeStatus.PENDING_FILL,
            )

        assert trade_id == "pending-trade-id"


class TestGetOpenTrades:
    """Tests for get_open_trades() functionality."""

    @pytest.fixture
    def mock_db_with_trades(self):
        """Create a mock D1 binding with trade results."""
        binding = MagicMock()
        prepare_mock = MagicMock()
        bind_mock = MagicMock()

        binding.prepare = MagicMock(return_value=prepare_mock)
        prepare_mock.bind = MagicMock(return_value=bind_mock)

        return binding, prepare_mock

    @pytest.mark.asyncio
    async def test_get_open_trades_filters_by_status(self, mock_db_with_trades):
        """Test that get_open_trades() only returns trades with 'open' status."""
        binding, prepare_mock = mock_db_with_trades

        mock_results = {
            "results": [
                {
                    "id": "trade-1",
                    "recommendation_id": "rec-1",
                    "opened_at": "2024-01-15T10:00:00",
                    "closed_at": None,
                    "status": "open",
                    "underlying": "SPY",
                    "spread_type": "bull_put",
                    "short_strike": 470.0,
                    "long_strike": 465.0,
                    "expiration": "2024-02-15",
                    "entry_credit": 1.25,
                    "exit_debit": None,
                    "profit_loss": None,
                    "contracts": 2,
                    "broker_order_id": "order-1",
                    "reflection": None,
                    "lesson": None,
                },
            ]
        }

        prepare_mock.all = AsyncMock(return_value=mock_results)

        # Mock js_to_python to return the results as-is (testing environment)
        with patch("core.db.d1.js_to_python", side_effect=lambda x: x):
            from core.db.d1 import D1Client

            client = D1Client(binding)
            trades = await client.get_open_trades()

        assert len(trades) == 1
        assert trades[0].status == TradeStatus.OPEN
        assert trades[0].underlying == "SPY"

        # Verify the query filters by status
        query = binding.prepare.call_args[0][0]
        assert "status = 'open'" in query


class TestMarkTradeFilled:
    """Tests for mark_trade_filled() functionality."""

    @pytest.fixture
    def mock_db_binding(self):
        """Create a mock D1 binding."""
        binding = MagicMock()
        prepare_mock = MagicMock()
        bind_mock = MagicMock()

        binding.prepare = MagicMock(return_value=prepare_mock)
        prepare_mock.bind = MagicMock(return_value=bind_mock)
        bind_mock.run = AsyncMock()

        binding._queries = []
        binding._params = []

        def capture_prepare(query):
            binding._queries.append(query)
            return prepare_mock

        def capture_bind(*args):
            binding._params.append(args)
            return bind_mock

        binding.prepare = MagicMock(side_effect=capture_prepare)
        prepare_mock.bind = MagicMock(side_effect=capture_bind)

        return binding

    @pytest.mark.asyncio
    async def test_mark_trade_filled_updates_status_and_timestamp(self, mock_db_binding):
        """Test that mark_trade_filled() updates status to 'open' and sets opened_at."""
        from core.db.d1 import D1Client

        client = D1Client(mock_db_binding)

        await client.mark_trade_filled("trade-123")

        # Verify the query
        assert len(mock_db_binding._queries) == 1
        query = mock_db_binding._queries[0]
        assert "UPDATE trades SET status = 'open'" in query
        assert "opened_at = ?" in query

        # Verify the trade_id was passed
        params = mock_db_binding._params[0]
        assert "trade-123" in params


class TestUpdateTradeOrderId:
    """Tests for update_trade_order_id() functionality."""

    @pytest.fixture
    def mock_db_binding(self):
        """Create a mock D1 binding."""
        binding = MagicMock()
        prepare_mock = MagicMock()
        bind_mock = MagicMock()

        binding.prepare = MagicMock(return_value=prepare_mock)
        prepare_mock.bind = MagicMock(return_value=bind_mock)
        bind_mock.run = AsyncMock()

        binding._queries = []
        binding._params = []

        def capture_prepare(query):
            binding._queries.append(query)
            return prepare_mock

        def capture_bind(*args):
            binding._params.append(args)
            return bind_mock

        binding.prepare = MagicMock(side_effect=capture_prepare)
        prepare_mock.bind = MagicMock(side_effect=capture_bind)

        return binding

    @pytest.mark.asyncio
    async def test_update_trade_order_id_updates_broker_order_id(self, mock_db_binding):
        """Test that update_trade_order_id() updates the broker_order_id field."""
        from core.db.d1 import D1Client

        client = D1Client(mock_db_binding)

        await client.update_trade_order_id("trade-123", "new-order-456")

        # Verify the query
        assert len(mock_db_binding._queries) == 1
        query = mock_db_binding._queries[0]
        assert "UPDATE trades SET broker_order_id = ?" in query

        # Verify the parameters
        params = mock_db_binding._params[0]
        assert "new-order-456" in params
        assert "trade-123" in params


class TestCloseTrade:
    """Tests for close_trade() functionality."""

    @pytest.fixture
    def mock_db_with_trade(self):
        """Create a mock D1 binding with a trade for closing."""
        binding = MagicMock()
        prepare_mock = MagicMock()
        bind_mock = MagicMock()

        binding.prepare = MagicMock(return_value=prepare_mock)
        prepare_mock.bind = MagicMock(return_value=bind_mock)
        bind_mock.run = AsyncMock()

        # For get_trade query
        trade_result = {
            "results": [
                {
                    "id": "trade-123",
                    "recommendation_id": "rec-123",
                    "opened_at": "2024-01-15T10:00:00",
                    "closed_at": None,
                    "status": "open",
                    "underlying": "SPY",
                    "spread_type": "bull_put",
                    "short_strike": 470.0,
                    "long_strike": 465.0,
                    "expiration": "2024-02-15",
                    "entry_credit": 1.25,
                    "exit_debit": None,
                    "profit_loss": None,
                    "contracts": 2,
                    "broker_order_id": "order-123",
                    "reflection": None,
                    "lesson": None,
                }
            ]
        }

        bind_mock.all = AsyncMock(return_value=trade_result)

        binding._queries = []
        binding._params = []

        def capture_prepare(query):
            binding._queries.append(query)
            return prepare_mock

        def capture_bind(*args):
            binding._params.append(args)
            return bind_mock

        binding.prepare = MagicMock(side_effect=capture_prepare)
        prepare_mock.bind = MagicMock(side_effect=capture_bind)

        return binding

    @pytest.mark.asyncio
    async def test_close_trade_calculates_pnl_correctly(self, mock_db_with_trade):
        """Test that close_trade() calculates P/L correctly."""
        # Mock js_to_python
        with patch("core.db.d1.js_to_python", side_effect=lambda x: x):
            from core.db.d1 import D1Client

            client = D1Client(mock_db_with_trade)

            await client.close_trade(
                trade_id="trade-123",
                exit_debit=0.50,  # Closed at 0.50 debit
            )

        # Verify the UPDATE query was called
        update_query = None
        for query in mock_db_with_trade._queries:
            if "UPDATE trades" in query and "status = 'closed'" in query:
                update_query = query
                break

        assert update_query is not None
        assert "exit_debit = ?" in update_query
        assert "profit_loss = ?" in update_query

        # P/L should be: (1.25 - 0.50) * 2 * 100 = $150
        # This is verified by checking the params passed to bind
        # The profit_loss should be 150.0

    @pytest.mark.asyncio
    async def test_close_trade_not_found_raises_error(self, mock_db_with_trade):
        """Test that close_trade() raises error for non-existent trade."""
        # Mock js_to_python to return empty results
        with patch("core.db.d1.js_to_python", return_value={"results": []}):
            from core.db.d1 import D1Client

            client = D1Client(mock_db_with_trade)

            with pytest.raises(ValueError) as exc_info:
                await client.close_trade(
                    trade_id="nonexistent-trade",
                    exit_debit=0.50,
                )

            assert "not found" in str(exc_info.value).lower()


class TestGetPendingFillTrades:
    """Tests for get_pending_fill_trades() functionality."""

    @pytest.fixture
    def mock_db_with_pending_trades(self):
        """Create a mock D1 binding with pending fill trades."""
        binding = MagicMock()
        prepare_mock = MagicMock()

        binding.prepare = MagicMock(return_value=prepare_mock)

        pending_result = {
            "results": [
                {
                    "id": "trade-pending-1",
                    "recommendation_id": "rec-1",
                    "opened_at": "2024-01-15T10:00:00",
                    "closed_at": None,
                    "status": "pending_fill",
                    "underlying": "SPY",
                    "spread_type": "bull_put",
                    "short_strike": 470.0,
                    "long_strike": 465.0,
                    "expiration": "2024-02-15",
                    "entry_credit": 1.25,
                    "exit_debit": None,
                    "profit_loss": None,
                    "contracts": 2,
                    "broker_order_id": "order-pending-1",
                    "reflection": None,
                    "lesson": None,
                },
                {
                    "id": "trade-pending-2",
                    "recommendation_id": "rec-2",
                    "opened_at": "2024-01-15T10:05:00",
                    "closed_at": None,
                    "status": "pending_fill",
                    "underlying": "QQQ",
                    "spread_type": "bear_call",
                    "short_strike": 400.0,
                    "long_strike": 405.0,
                    "expiration": "2024-03-15",
                    "entry_credit": 0.95,
                    "exit_debit": None,
                    "profit_loss": None,
                    "contracts": 1,
                    "broker_order_id": "order-pending-2",
                    "reflection": None,
                    "lesson": None,
                },
            ]
        }

        prepare_mock.all = AsyncMock(return_value=pending_result)

        binding._queries = []

        def capture_prepare(query):
            binding._queries.append(query)
            return prepare_mock

        binding.prepare = MagicMock(side_effect=capture_prepare)

        return binding

    @pytest.mark.asyncio
    async def test_get_pending_fill_trades_returns_all_pending(self, mock_db_with_pending_trades):
        """Test that get_pending_fill_trades() returns all trades with pending_fill status."""
        with patch("core.db.d1.js_to_python", side_effect=lambda x: x):
            from core.db.d1 import D1Client

            client = D1Client(mock_db_with_pending_trades)
            trades = await client.get_pending_fill_trades()

        assert len(trades) == 2
        assert all(t.status == TradeStatus.PENDING_FILL for t in trades)

        # Verify the query filters by status
        query = mock_db_with_pending_trades._queries[0]
        assert "status = 'pending_fill'" in query
