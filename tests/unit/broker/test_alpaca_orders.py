"""Tests for Alpaca order placement, replacement, and cancellation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.broker.alpaca import AlpacaClient, AlpacaError
from core.broker.types import (
    Order,
    OrderLeg,
    OrderSide,
    OrderStatus,
    OrderType,
    SpreadOrder,
)


class TestPlaceSpreadOrder:
    """Tests for place_spread_order() functionality."""

    @pytest.fixture
    def alpaca_client(self):
        """Create an AlpacaClient with mocked HTTP."""
        return AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_place_spread_order_constructs_correct_payload(self, alpaca_client):
        """Test that place_spread_order() constructs the correct API payload."""
        spread = SpreadOrder(
            underlying="SPY",
            short_symbol="SPY240215P00470000",
            long_symbol="SPY240215P00465000",
            contracts=2,
            limit_price=1.25,
        )

        mock_response = {
            "id": "order-123",
            "client_order_id": "client-order-123",
            "symbol": "",
            "side": "",
            "type": "limit",
            "qty": "2",
            "limit_price": "-1.25",
            "status": "pending_new",
            "filled_qty": "0",
            "filled_avg_price": None,
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
            "legs": [
                {"symbol": "SPY240215P00470000", "side": "sell", "qty": "2", "filled_qty": "0", "filled_avg_price": None},
                {"symbol": "SPY240215P00465000", "side": "buy", "qty": "2", "filled_qty": "0", "filled_avg_price": None},
            ],
        }

        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            order = await alpaca_client.place_spread_order(spread)

            # Verify the request was made with correct payload
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/v2/orders"

            json_data = call_args[1]["json_data"]
            assert json_data["order_class"] == "mleg"
            assert json_data["qty"] == "2"
            assert json_data["type"] == "limit"
            assert json_data["time_in_force"] == "day"
            # Credit spread uses negative limit price
            assert json_data["limit_price"] == "-1.25"

            # Verify legs
            assert len(json_data["legs"]) == 2
            assert json_data["legs"][0]["symbol"] == "SPY240215P00470000"
            assert json_data["legs"][0]["side"] == "sell"
            assert json_data["legs"][0]["position_intent"] == "sell_to_open"
            assert json_data["legs"][1]["symbol"] == "SPY240215P00465000"
            assert json_data["legs"][1]["side"] == "buy"
            assert json_data["legs"][1]["position_intent"] == "buy_to_open"

    @pytest.mark.asyncio
    async def test_place_spread_order_rounds_limit_price(self, alpaca_client):
        """Test that limit price is rounded to 2 decimal places."""
        spread = SpreadOrder(
            underlying="SPY",
            short_symbol="SPY240215P00470000",
            long_symbol="SPY240215P00465000",
            contracts=1,
            limit_price=1.256789,  # Should be rounded to 1.26
        )

        mock_response = {
            "id": "order-123",
            "client_order_id": "client-order-123",
            "symbol": "",
            "side": "",
            "type": "limit",
            "qty": "1",
            "limit_price": "-1.26",
            "status": "pending_new",
            "filled_qty": "0",
            "filled_avg_price": None,
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
            "legs": [
                {"symbol": "SPY240215P00470000", "side": "sell", "qty": "1", "filled_qty": "0", "filled_avg_price": None},
                {"symbol": "SPY240215P00465000", "side": "buy", "qty": "1", "filled_qty": "0", "filled_avg_price": None},
            ],
        }

        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await alpaca_client.place_spread_order(spread)

            json_data = mock_request.call_args[1]["json_data"]
            assert json_data["limit_price"] == "-1.26"


class TestReplaceOrder:
    """Tests for replace_order() functionality."""

    @pytest.fixture
    def alpaca_client(self):
        """Create an AlpacaClient with mocked HTTP."""
        return AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_replace_order_with_valid_parameters(self, alpaca_client):
        """Test replace_order() with valid qty and limit_price."""
        mock_response = {
            "id": "new-order-456",
            "client_order_id": "client-order-456",
            "symbol": "",
            "side": "",
            "type": "limit",
            "qty": "3",
            "limit_price": "-1.20",
            "status": "pending_new",
            "filled_qty": "0",
            "filled_avg_price": None,
            "created_at": "2024-01-15T10:05:00Z",
            "updated_at": "2024-01-15T10:05:00Z",
            "legs": [
                {"symbol": "SPY240215P00470000", "side": "sell", "qty": "3", "filled_qty": "0", "filled_avg_price": None},
                {"symbol": "SPY240215P00465000", "side": "buy", "qty": "3", "filled_qty": "0", "filled_avg_price": None},
            ],
        }

        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            new_order = await alpaca_client.replace_order(
                order_id="order-123",
                qty=3,
                limit_price=-1.20,
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "PATCH"
            assert call_args[0][1] == "/v2/orders/order-123"

            json_data = call_args[1]["json_data"]
            assert json_data["qty"] == "3"
            assert json_data["limit_price"] == "-1.2"

            assert new_order.id == "new-order-456"

    @pytest.mark.asyncio
    async def test_replace_order_handles_errors(self, alpaca_client):
        """Test that replace_order() properly propagates errors."""
        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = AlpacaError("Order not found", status_code=404)

            with pytest.raises(AlpacaError) as exc_info:
                await alpaca_client.replace_order(
                    order_id="nonexistent-order",
                    limit_price=-1.15,
                )

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_replace_order_rounds_limit_price(self, alpaca_client):
        """Test that replace_order() rounds limit price to 2 decimal places."""
        mock_response = {
            "id": "new-order-456",
            "client_order_id": "client-order-456",
            "symbol": "",
            "side": "",
            "type": "limit",
            "qty": "2",
            "limit_price": "-1.23",
            "status": "pending_new",
            "filled_qty": "0",
            "filled_avg_price": None,
            "created_at": "2024-01-15T10:05:00Z",
            "updated_at": "2024-01-15T10:05:00Z",
            "legs": [
                {"symbol": "SPY240215P00470000", "side": "sell", "qty": "2", "filled_qty": "0", "filled_avg_price": None},
                {"symbol": "SPY240215P00465000", "side": "buy", "qty": "2", "filled_qty": "0", "filled_avg_price": None},
            ],
        }

        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await alpaca_client.replace_order(
                order_id="order-123",
                limit_price=-1.234567,
            )

            json_data = mock_request.call_args[1]["json_data"]
            assert json_data["limit_price"] == "-1.23"


class TestParseOrder:
    """Tests for _parse_order() functionality."""

    @pytest.fixture
    def alpaca_client(self):
        """Create an AlpacaClient."""
        return AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            paper=True,
        )

    def test_parse_mleg_order_with_empty_side(self, alpaca_client, sample_alpaca_order_response):
        """Test parsing multi-leg order where top-level side is empty."""
        order = alpaca_client._parse_order(sample_alpaca_order_response)

        assert order.id == "order-123"
        assert order.qty == 2
        assert order.status == OrderStatus.PENDING
        # Side should be derived from first leg
        assert order.side == OrderSide.SELL
        assert len(order.legs) == 2
        assert order.legs[0].side == OrderSide.SELL
        assert order.legs[1].side == OrderSide.BUY

    def test_parse_single_leg_order(self, alpaca_client, sample_alpaca_single_order_response):
        """Test parsing single-leg order with valid side."""
        order = alpaca_client._parse_order(sample_alpaca_single_order_response)

        assert order.id == "order-456"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.FILLED
        assert order.legs is None

    def test_parse_order_handles_none_side(self, alpaca_client):
        """Test that _parse_order handles None side values gracefully."""
        data = {
            "id": "order-789",
            "client_order_id": "client-789",
            "symbol": "",
            "side": None,  # Explicitly None
            "type": "limit",
            "qty": "1",
            "limit_price": "-1.00",
            "status": "new",
            "filled_qty": "0",
            "filled_avg_price": None,
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
            "legs": [
                {"symbol": "SPY240215P00470000", "side": None, "qty": "1", "filled_qty": "0", "filled_avg_price": None},
                {"symbol": "SPY240215P00465000", "side": None, "qty": "1", "filled_qty": "0", "filled_avg_price": None},
            ],
        }

        order = alpaca_client._parse_order(data)

        # Should default legs to sell/buy based on position
        assert order.legs[0].side == OrderSide.SELL
        assert order.legs[1].side == OrderSide.BUY
        # Top-level side should be derived from first leg
        assert order.side == OrderSide.SELL


class TestParseOccSymbol:
    """Tests for parse_occ_symbol() functionality."""

    @pytest.fixture
    def alpaca_client(self):
        """Create an AlpacaClient."""
        return AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            paper=True,
        )

    def test_parse_valid_put_symbol(self, alpaca_client):
        """Test parsing a valid put option symbol."""
        result = alpaca_client.parse_occ_symbol("SPY240215P00470000")

        assert result is not None
        assert result["underlying"] == "SPY"
        assert result["expiration"] == "2024-02-15"
        assert result["option_type"] == "put"
        assert result["strike"] == 470.0

    def test_parse_valid_call_symbol(self, alpaca_client):
        """Test parsing a valid call option symbol."""
        result = alpaca_client.parse_occ_symbol("QQQ250117C00400000")

        assert result is not None
        assert result["underlying"] == "QQQ"
        assert result["expiration"] == "2025-01-17"
        assert result["option_type"] == "call"
        assert result["strike"] == 400.0

    def test_parse_symbol_with_decimal_strike(self, alpaca_client):
        """Test parsing a symbol with decimal strike price."""
        result = alpaca_client.parse_occ_symbol("IWM240315P00195500")

        assert result is not None
        assert result["underlying"] == "IWM"
        assert result["strike"] == 195.5

    def test_parse_invalid_symbol_returns_none(self, alpaca_client):
        """Test that invalid symbols return None."""
        # Too short
        assert alpaca_client.parse_occ_symbol("SPY") is None
        # Empty string
        assert alpaca_client.parse_occ_symbol("") is None
        # Missing strike digits
        assert alpaca_client.parse_occ_symbol("SPY240215P") is None

    def test_parse_stock_symbol_returns_none(self, alpaca_client):
        """Test that stock symbols (not options) return None."""
        assert alpaca_client.parse_occ_symbol("AAPL") is None
        assert alpaca_client.parse_occ_symbol("TSLA") is None


class TestCancelOrder:
    """Tests for cancel_order() functionality."""

    @pytest.fixture
    def alpaca_client(self):
        """Create an AlpacaClient with mocked HTTP."""
        return AlpacaClient(
            api_key="test-key",
            secret_key="test-secret",
            paper=True,
        )

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, alpaca_client):
        """Test successful order cancellation."""
        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None
            await alpaca_client.cancel_order("order-123")

            mock_request.assert_called_once_with("DELETE", "/v2/orders/order-123")

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, alpaca_client):
        """Test cancellation of non-existent order."""
        with patch.object(alpaca_client, "_trading_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = AlpacaError("Order not found", status_code=404)

            with pytest.raises(AlpacaError) as exc_info:
                await alpaca_client.cancel_order("nonexistent-order")

            assert exc_info.value.status_code == 404
