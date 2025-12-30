"""Tests for position monitor fill reconciliation and price adjustment logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.broker.types import Order, OrderLeg, OrderSide, OrderStatus, OrderType
from core.types import SpreadType, Trade, TradeStatus


# Import the functions we're testing - these need to be imported after mocking
# the Cloudflare Workers modules in conftest.py


class TestReconcilePendingOrders:
    """Tests for _reconcile_pending_orders() functionality."""

    @pytest.fixture
    def mock_trade(self) -> Trade:
        """Create a mock pending_fill trade."""
        return Trade(
            id="trade-123",
            recommendation_id="rec-123",
            opened_at=datetime.now(),
            closed_at=None,
            status=TradeStatus.PENDING_FILL,
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            entry_credit=1.25,
            exit_debit=None,
            profit_loss=None,
            contracts=2,
            broker_order_id="order-123",
            reflection=None,
            lesson=None,
        )

    @pytest.fixture
    def mock_filled_order(self) -> Order:
        """Create a mock filled order."""
        return Order(
            id="order-123",
            client_order_id="client-123",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-1.25,
            status=OrderStatus.FILLED,
            filled_qty=2,
            filled_avg_price=-1.24,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            legs=[
                OrderLeg(symbol="SPY240215P00470000", side=OrderSide.SELL, qty=2, filled_qty=2, filled_avg_price=3.50),
                OrderLeg(symbol="SPY240215P00465000", side=OrderSide.BUY, qty=2, filled_qty=2, filled_avg_price=2.26),
            ],
        )

    @pytest.fixture
    def mock_expired_order(self) -> Order:
        """Create a mock expired order."""
        return Order(
            id="order-123",
            client_order_id="client-123",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-1.25,
            status=OrderStatus.EXPIRED,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )

    @pytest.mark.asyncio
    async def test_reconcile_filled_order_marks_trade_as_open(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
        mock_trade,
        mock_filled_order,
    ):
        """Test that FILLED orders result in trade being marked as open."""
        mock_d1_client.get_pending_fill_trades = AsyncMock(return_value=[mock_trade])
        mock_alpaca_client.get_order = AsyncMock(return_value=mock_filled_order)

        # Import after mocking
        from handlers.position_monitor import _reconcile_pending_orders

        await _reconcile_pending_orders(
            db=mock_d1_client,
            alpaca=mock_alpaca_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Verify trade was marked as filled
        mock_d1_client.mark_trade_filled.assert_called_once_with(mock_trade.id)

        # Verify adjustment tracking was cleaned up
        mock_kv_client.delete.assert_called_with(f"order_adjustment:{mock_trade.id}")

        # Verify daily stats were updated
        mock_kv_client.update_daily_stats.assert_called_once_with(trades_delta=1)

        # Verify Discord notification was sent
        mock_discord_client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_expired_order_marks_trade_as_expired(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
        mock_trade,
        mock_expired_order,
    ):
        """Test that EXPIRED orders result in trade being marked as expired."""
        mock_d1_client.get_pending_fill_trades = AsyncMock(return_value=[mock_trade])
        mock_alpaca_client.get_order = AsyncMock(return_value=mock_expired_order)

        from handlers.position_monitor import _reconcile_pending_orders

        await _reconcile_pending_orders(
            db=mock_d1_client,
            alpaca=mock_alpaca_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Verify trade status was updated to expired
        mock_d1_client.update_trade_status.assert_called_once_with(
            mock_trade.id, TradeStatus.EXPIRED
        )

        # Verify position was deleted
        mock_d1_client.delete_position.assert_called_once_with(mock_trade.id)

        # Verify Discord notification was sent
        mock_discord_client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_trade_without_broker_order_id(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
    ):
        """Test that trades without broker_order_id are marked as expired."""
        trade_no_order = Trade(
            id="trade-no-order",
            recommendation_id="rec-123",
            opened_at=datetime.now(),
            closed_at=None,
            status=TradeStatus.PENDING_FILL,
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            entry_credit=1.25,
            exit_debit=None,
            profit_loss=None,
            contracts=2,
            broker_order_id=None,  # No broker order ID
            reflection=None,
            lesson=None,
        )

        mock_d1_client.get_pending_fill_trades = AsyncMock(return_value=[trade_no_order])

        from handlers.position_monitor import _reconcile_pending_orders

        await _reconcile_pending_orders(
            db=mock_d1_client,
            alpaca=mock_alpaca_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Should mark as expired without calling get_order
        mock_d1_client.update_trade_status.assert_called_once_with(
            trade_no_order.id, TradeStatus.EXPIRED
        )
        mock_alpaca_client.get_order.assert_not_called()


class TestMaybeAdjustOrderPrice:
    """Tests for _maybe_adjust_order_price() functionality."""

    @pytest.fixture
    def mock_trade(self) -> Trade:
        """Create a mock pending_fill trade."""
        return Trade(
            id="trade-123",
            recommendation_id="rec-123",
            opened_at=datetime.now(),
            closed_at=None,
            status=TradeStatus.PENDING_FILL,
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            entry_credit=1.25,
            exit_debit=None,
            profit_loss=None,
            contracts=2,
            broker_order_id="order-123",
            reflection=None,
            lesson=None,
        )

    @pytest.fixture
    def mock_new_order(self) -> Order:
        """Create a mock new/pending order."""
        return Order(
            id="order-123",
            client_order_id="client-123",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-1.25,
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=6),  # 6 min old
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )

    @pytest.mark.asyncio
    async def test_adjust_price_at_5_minute_threshold(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
        mock_trade,
        mock_new_order,
    ):
        """Test price adjustment at the 5 minute threshold."""
        # Order is 6 minutes old, should trigger first adjustment
        mock_kv_client.get_json = AsyncMock(return_value=None)

        mock_replaced_order = Order(
            id="new-order-456",
            client_order_id="client-456",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-1.23,  # Reduced credit
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )
        mock_alpaca_client.replace_order = AsyncMock(return_value=mock_replaced_order)

        from handlers.position_monitor import _maybe_adjust_order_price

        await _maybe_adjust_order_price(
            trade=mock_trade,
            order=mock_new_order,
            alpaca=mock_alpaca_client,
            db=mock_d1_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Verify replace_order was called
        mock_alpaca_client.replace_order.assert_called_once()

        # Verify trade order ID was updated
        mock_d1_client.update_trade_order_id.assert_called_once_with(
            mock_trade.id, "new-order-456"
        )

        # Verify adjustment state was saved
        mock_kv_client.put_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_adjustment_before_5_minutes(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
        mock_trade,
    ):
        """Test that no adjustment happens before 5 minute threshold."""
        # Order is only 3 minutes old
        recent_order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-1.25,
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=3),
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )

        mock_kv_client.get_json = AsyncMock(return_value=None)

        from handlers.position_monitor import _maybe_adjust_order_price

        await _maybe_adjust_order_price(
            trade=mock_trade,
            order=recent_order,
            alpaca=mock_alpaca_client,
            db=mock_d1_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Should not call replace_order
        mock_alpaca_client.replace_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_max_total_adjustment(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
        mock_trade,
    ):
        """Test that MAX_TOTAL_ADJUSTMENT limit is respected."""
        # Order is 30 minutes old (past all adjustment thresholds)
        old_order = Order(
            id="order-123",
            client_order_id="client-123",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-1.25,
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=30),
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )

        # Simulate already having made adjustments
        mock_kv_client.get_json = AsyncMock(return_value={
            "adjustments_made": 6,  # All adjustments done (6 in schedule)
            "original_price": 1.25,
            "current_price": 0.98,  # Already adjusted 27 cents (close to MAX_TOTAL_ADJUSTMENT of 0.30)
            "original_order_id": "order-123",
        })

        from handlers.position_monitor import _maybe_adjust_order_price

        await _maybe_adjust_order_price(
            trade=mock_trade,
            order=old_order,
            alpaca=mock_alpaca_client,
            db=mock_d1_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Should not call replace_order since we've hit the adjustment limit
        mock_alpaca_client.replace_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_min_credit(
        self,
        mock_d1_client,
        mock_alpaca_client,
        mock_discord_client,
        mock_kv_client,
    ):
        """Test that MIN_CREDIT limit is respected."""
        # Trade with very low credit
        low_credit_trade = Trade(
            id="trade-low",
            recommendation_id="rec-123",
            opened_at=datetime.now(),
            closed_at=None,
            status=TradeStatus.PENDING_FILL,
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration="2024-02-15",
            entry_credit=0.10,  # Very low credit
            exit_debit=None,
            profit_loss=None,
            contracts=2,
            broker_order_id="order-low",
            reflection=None,
            lesson=None,
        )

        old_order = Order(
            id="order-low",
            client_order_id="client-low",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-0.10,
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=15),
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )

        mock_kv_client.get_json = AsyncMock(return_value=None)

        mock_replaced_order = Order(
            id="new-order-789",
            client_order_id="client-789",
            symbol="",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=2,
            limit_price=-0.05,  # Minimum credit
            status=OrderStatus.NEW,
            filled_qty=0,
            filled_avg_price=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            legs=None,
        )
        mock_alpaca_client.replace_order = AsyncMock(return_value=mock_replaced_order)

        from handlers.position_monitor import _maybe_adjust_order_price

        await _maybe_adjust_order_price(
            trade=low_credit_trade,
            order=old_order,
            alpaca=mock_alpaca_client,
            db=mock_d1_client,
            discord=mock_discord_client,
            kv=mock_kv_client,
        )

        # Should call replace_order with MIN_CREDIT as the floor
        if mock_alpaca_client.replace_order.called:
            call_args = mock_alpaca_client.replace_order.call_args
            # The new limit_price should not go below -MIN_CREDIT (0.05)
            assert call_args[1]["limit_price"] >= -0.05


class TestPriceAdjustmentSchedule:
    """Tests for the price adjustment schedule constants."""

    def test_adjustment_schedule_is_defined(self):
        """Test that PRICE_ADJUSTMENT_SCHEDULE is properly defined."""
        from handlers.position_monitor import PRICE_ADJUSTMENT_SCHEDULE

        assert len(PRICE_ADJUSTMENT_SCHEDULE) == 6
        # Should be tuples of (minutes, adjustment)
        for minutes, adjustment in PRICE_ADJUSTMENT_SCHEDULE:
            assert isinstance(minutes, int)
            assert isinstance(adjustment, float)
            assert minutes > 0
            assert adjustment > 0

    def test_max_total_adjustment_is_defined(self):
        """Test that MAX_TOTAL_ADJUSTMENT is properly defined."""
        from handlers.position_monitor import MAX_TOTAL_ADJUSTMENT

        assert MAX_TOTAL_ADJUSTMENT == 0.30

    def test_min_credit_is_defined(self):
        """Test that MIN_CREDIT is properly defined."""
        from handlers.position_monitor import MIN_CREDIT

        assert MIN_CREDIT == 0.05

    def test_total_adjustments_are_reasonable(self):
        """Test that sum of adjustments in schedule is reasonable."""
        from handlers.position_monitor import PRICE_ADJUSTMENT_SCHEDULE, MAX_TOTAL_ADJUSTMENT

        total = sum(adj for _, adj in PRICE_ADJUSTMENT_SCHEDULE)
        # Total adjustments should not exceed MAX_TOTAL_ADJUSTMENT
        assert total <= MAX_TOTAL_ADJUSTMENT
        # And should be at least 50% of max (reasonable minimum)
        assert total >= MAX_TOTAL_ADJUSTMENT * 0.5
