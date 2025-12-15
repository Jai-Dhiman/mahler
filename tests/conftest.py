"""Pytest configuration and fixtures for Mahler tests."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# Mock Cloudflare Workers-specific modules before any imports
# These modules are only available in the Pyodide/Workers environment
js_mock = MagicMock()
pyodide_mock = MagicMock()
workers_mock = MagicMock()

sys.modules["js"] = js_mock
sys.modules["pyodide"] = pyodide_mock
sys.modules["pyodide.ffi"] = MagicMock()
sys.modules["workers"] = workers_mock
workers_mock.Response = MagicMock()


# Mock the Cloudflare Workers environment
@pytest.fixture
def mock_env():
    """Mock Cloudflare Workers environment bindings."""
    env = MagicMock()
    env.ALPACA_API_KEY = "test-api-key"
    env.ALPACA_SECRET_KEY = "test-secret-key"
    env.DISCORD_BOT_TOKEN = "test-bot-token"
    env.DISCORD_PUBLIC_KEY = "test-public-key"
    env.DISCORD_CHANNEL_ID = "test-channel-id"
    env.ENVIRONMENT = "paper"
    return env


@pytest.fixture
def mock_d1_db():
    """Mock D1 database binding."""
    db = MagicMock()
    db.prepare = MagicMock(return_value=MagicMock())
    db.prepare.return_value.bind = MagicMock(return_value=MagicMock())
    db.prepare.return_value.bind.return_value.all = AsyncMock(return_value={"results": []})
    db.prepare.return_value.bind.return_value.run = AsyncMock()
    db.prepare.return_value.all = AsyncMock(return_value={"results": []})
    db.prepare.return_value.run = AsyncMock()
    return db


@pytest.fixture
def mock_kv():
    """Mock KV namespace binding."""
    kv = MagicMock()
    kv.get = AsyncMock(return_value=None)
    kv.put = AsyncMock()
    kv.delete = AsyncMock()
    return kv


# Sample data fixtures
@pytest.fixture
def sample_recommendation_row() -> dict:
    """Sample recommendation row from database."""
    return {
        "id": "rec-123",
        "created_at": "2024-01-15T10:00:00",
        "expires_at": "2024-01-15T10:15:00",
        "status": "pending",
        "underlying": "SPY",
        "spread_type": "bull_put",
        "short_strike": 470.0,
        "long_strike": 465.0,
        "expiration": "2024-02-15",
        "credit": 1.25,
        "max_loss": 375.0,
        "iv_rank": 55.0,
        "delta": -0.25,
        "theta": 0.03,
        "thesis": "Bullish on SPY",
        "confidence": "medium",
        "suggested_contracts": 2,
        "analysis_price": 598.50,
        "discord_message_id": "msg-123",
    }


@pytest.fixture
def sample_trade_row() -> dict:
    """Sample trade row from database."""
    return {
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


@pytest.fixture
def sample_position_row() -> dict:
    """Sample position row from database."""
    return {
        "id": "pos-123",
        "trade_id": "trade-123",
        "underlying": "SPY",
        "short_strike": 470.0,
        "long_strike": 465.0,
        "expiration": "2024-02-15",
        "contracts": 2,
        "current_value": 0.75,
        "unrealized_pnl": 100.0,
        "updated_at": "2024-01-15T14:00:00",
    }


@pytest.fixture
def sample_alpaca_order_response() -> dict:
    """Sample Alpaca order API response."""
    return {
        "id": "order-123",
        "client_order_id": "client-order-123",
        "symbol": "",  # Empty for mleg orders
        "side": "",  # Empty for mleg orders - THIS IS THE BUG SCENARIO
        "type": "limit",
        "qty": "2",
        "limit_price": "-1.25",
        "status": "pending_new",
        "filled_qty": "0",
        "filled_avg_price": None,
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z",
        "legs": [
            {
                "symbol": "SPY240215P00470000",
                "side": "sell",
                "qty": "2",
                "filled_qty": "0",
                "filled_avg_price": None,
            },
            {
                "symbol": "SPY240215P00465000",
                "side": "buy",
                "qty": "2",
                "filled_qty": "0",
                "filled_avg_price": None,
            },
        ],
    }


@pytest.fixture
def sample_alpaca_single_order_response() -> dict:
    """Sample Alpaca single-leg order API response."""
    return {
        "id": "order-456",
        "client_order_id": "client-order-456",
        "symbol": "SPY",
        "side": "buy",
        "type": "limit",
        "qty": "100",
        "limit_price": "598.50",
        "status": "filled",
        "filled_qty": "100",
        "filled_avg_price": "598.45",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:05Z",
        "legs": None,
    }


@pytest.fixture
def sample_alpaca_position_response() -> dict:
    """Sample Alpaca position API response."""
    return {
        "symbol": "SPY240215P00470000",
        "qty": "-2",
        "side": "short",
        "avg_entry_price": "3.50",
        "market_value": "-700.00",
        "cost_basis": "-700.00",
        "unrealized_pl": "100.00",
        "unrealized_plpc": "0.1428",
        "current_price": "3.00",
    }


@pytest.fixture
def sample_option_snapshot() -> dict:
    """Sample Alpaca option snapshot response."""
    return {
        "latestQuote": {
            "bp": 1.20,  # bid price
            "ap": 1.30,  # ask price
        },
        "latestTrade": {
            "p": 1.25,  # last price
        },
        "greeks": {
            "delta": -0.25,
            "gamma": 0.02,
            "theta": -0.03,
            "vega": 0.15,
            "impliedVolatility": 0.22,
        },
        "openInterest": 5000,
        "dailyBar": {
            "v": 1500,  # volume
        },
    }


@pytest.fixture
def sample_discord_interaction_payload() -> dict:
    """Sample Discord interaction payload for button click."""
    return {
        "type": 3,  # MESSAGE_COMPONENT
        "id": "interaction-123",
        "token": "interaction-token",
        "data": {
            "custom_id": "approve_trade:rec-123",
        },
        "message": {
            "id": "msg-123",
        },
    }


# Utility fixtures
@pytest.fixture
def future_date() -> str:
    """Return a date 35 days in the future (within 30-45 DTE range)."""
    return (datetime.now() + timedelta(days=35)).strftime("%Y-%m-%d")


@pytest.fixture
def past_date() -> str:
    """Return a date in the past."""
    return (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")


@pytest.fixture
def near_expiry_date() -> str:
    """Return a date 15 days in the future (below 21 DTE threshold)."""
    return (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
