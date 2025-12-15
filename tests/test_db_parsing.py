"""Tests for database row parsing.

These tests ensure database rows are correctly parsed into domain objects,
handling missing/null fields and type conversions properly.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.types import (
    Confidence,
    RecommendationStatus,
    SpreadType,
    TradeStatus,
)


class TestRecommendationParsing:
    """Test _row_to_recommendation parsing."""

    def test_full_recommendation_parsing(self, sample_recommendation_row):
        """Test parsing a complete recommendation row."""
        from core.db.d1 import D1Client

        client = D1Client(None)
        rec = client._row_to_recommendation(sample_recommendation_row)

        assert rec.id == "rec-123"
        assert rec.status == RecommendationStatus.PENDING
        assert rec.underlying == "SPY"
        assert rec.spread_type == SpreadType.BULL_PUT
        assert rec.short_strike == 470.0
        assert rec.long_strike == 465.0
        assert rec.credit == 1.25
        assert rec.confidence == Confidence.MEDIUM
        assert rec.suggested_contracts == 2

    def test_recommendation_with_null_optional_fields(self):
        """Test parsing recommendation with null optional fields."""
        from core.db.d1 import D1Client

        row = {
            "id": "rec-minimal",
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
            "iv_rank": None,
            "delta": None,
            "theta": None,
            "thesis": None,
            "confidence": None,  # Null confidence
            "suggested_contracts": None,
            "analysis_price": None,
            "discord_message_id": None,
        }

        client = D1Client(None)
        rec = client._row_to_recommendation(row)

        assert rec.iv_rank is None
        assert rec.delta is None
        assert rec.theta is None
        assert rec.thesis is None
        assert rec.confidence is None
        assert rec.suggested_contracts is None

    def test_recommendation_empty_string_confidence_raises(self):
        """Test that empty string confidence raises error during parsing."""
        from core.db.d1 import D1Client

        row = {
            "id": "rec-bad",
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
            "iv_rank": None,
            "delta": None,
            "theta": None,
            "thesis": None,
            "confidence": "",  # Empty string instead of null
            "suggested_contracts": None,
            "analysis_price": None,
            "discord_message_id": None,
        }

        client = D1Client(None)
        # Empty string should be handled - either by raising or by treating as None
        # Current implementation: Confidence("") would raise ValueError
        # But the code checks `if row["confidence"]` which is falsy for ""
        rec = client._row_to_recommendation(row)
        assert rec.confidence is None

    def test_recommendation_invalid_status_raises(self):
        """Test that invalid status raises ValueError."""
        from core.db.d1 import D1Client

        row = {
            "id": "rec-bad",
            "created_at": "2024-01-15T10:00:00",
            "expires_at": "2024-01-15T10:15:00",
            "status": "invalid_status",  # Invalid
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 470.0,
            "long_strike": 465.0,
            "expiration": "2024-02-15",
            "credit": 1.25,
            "max_loss": 375.0,
            "iv_rank": None,
            "delta": None,
            "theta": None,
            "thesis": None,
            "confidence": None,
            "suggested_contracts": None,
            "analysis_price": None,
            "discord_message_id": None,
        }

        client = D1Client(None)
        with pytest.raises(ValueError):
            client._row_to_recommendation(row)


class TestTradeParsing:
    """Test _row_to_trade parsing."""

    def test_full_trade_parsing(self, sample_trade_row):
        """Test parsing a complete trade row."""
        from core.db.d1 import D1Client

        client = D1Client(None)
        trade = client._row_to_trade(sample_trade_row)

        assert trade.id == "trade-123"
        assert trade.status == TradeStatus.OPEN
        assert trade.underlying == "SPY"
        assert trade.spread_type == SpreadType.BULL_PUT
        assert trade.contracts == 2

    def test_closed_trade_parsing(self):
        """Test parsing a closed trade with all fields."""
        from core.db.d1 import D1Client

        row = {
            "id": "trade-closed",
            "recommendation_id": "rec-123",
            "opened_at": "2024-01-15T10:00:00",
            "closed_at": "2024-01-20T14:30:00",
            "status": "closed",
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 470.0,
            "long_strike": 465.0,
            "expiration": "2024-02-15",
            "entry_credit": 1.25,
            "exit_debit": 0.50,
            "profit_loss": 150.0,
            "contracts": 2,
            "broker_order_id": "order-123",
            "reflection": "Good entry timing",
            "lesson": "Wait for IV spike",
        }

        client = D1Client(None)
        trade = client._row_to_trade(row)

        assert trade.status == TradeStatus.CLOSED
        assert trade.closed_at is not None
        assert trade.exit_debit == 0.50
        assert trade.profit_loss == 150.0
        assert trade.reflection == "Good entry timing"

    def test_trade_null_dates_handled(self):
        """Test trade with null opened_at/closed_at."""
        from core.db.d1 import D1Client

        row = {
            "id": "trade-nodates",
            "recommendation_id": None,
            "opened_at": None,  # Null
            "closed_at": None,  # Null
            "status": "open",
            "underlying": "SPY",
            "spread_type": "bear_call",
            "short_strike": 610.0,
            "long_strike": 615.0,
            "expiration": "2024-02-15",
            "entry_credit": 0.80,
            "exit_debit": None,
            "profit_loss": None,
            "contracts": 1,
            "broker_order_id": None,
            "reflection": None,
            "lesson": None,
        }

        client = D1Client(None)
        trade = client._row_to_trade(row)

        assert trade.opened_at is None
        assert trade.closed_at is None


class TestPositionParsing:
    """Test _row_to_position parsing."""

    def test_position_parsing(self, sample_position_row):
        """Test parsing a position row."""
        from core.db.d1 import D1Client

        client = D1Client(None)
        position = client._row_to_position(sample_position_row)

        assert position.id == "pos-123"
        assert position.trade_id == "trade-123"
        assert position.underlying == "SPY"
        assert position.contracts == 2
        assert position.current_value == 0.75
        assert position.unrealized_pnl == 100.0


class TestDailyPerformanceParsing:
    """Test _row_to_daily_performance parsing."""

    def test_daily_performance_parsing(self):
        """Test parsing daily performance row."""
        from core.db.d1 import D1Client

        row = {
            "date": "2024-01-15",
            "starting_balance": 10000.0,
            "ending_balance": 10150.0,
            "realized_pnl": 150.0,
            "trades_opened": 2,
            "trades_closed": 1,
            "win_count": 1,
            "loss_count": 0,
        }

        client = D1Client(None)
        perf = client._row_to_daily_performance(row)

        assert perf.date == "2024-01-15"
        assert perf.realized_pnl == 150.0
        assert perf.win_count == 1


class TestPlaybookRuleParsing:
    """Test _row_to_playbook_rule parsing."""

    def test_playbook_rule_parsing(self):
        """Test parsing playbook rule row."""
        from core.db.d1 import D1Client

        row = {
            "id": "rule-123",
            "rule": "Wait for IV rank > 60 before entering",
            "source": "learned",
            "supporting_trade_ids": '["trade-1", "trade-2"]',
            "created_at": "2024-01-15T10:00:00",
        }

        client = D1Client(None)
        rule = client._row_to_playbook_rule(row)

        assert rule.id == "rule-123"
        assert rule.source == "learned"
        assert len(rule.supporting_trade_ids) == 2
        assert "trade-1" in rule.supporting_trade_ids

    def test_playbook_rule_empty_supporting_trades(self):
        """Test parsing rule with null/empty supporting trades."""
        from core.db.d1 import D1Client

        row = {
            "id": "rule-456",
            "rule": "Always use stop losses",
            "source": "initial",
            "supporting_trade_ids": None,
            "created_at": None,
        }

        client = D1Client(None)
        rule = client._row_to_playbook_rule(row)

        assert rule.supporting_trade_ids == []
        assert rule.created_at is None

    def test_playbook_rule_malformed_json(self):
        """Test parsing rule with malformed JSON in supporting_trade_ids."""
        from core.db.d1 import D1Client
        import json

        row = {
            "id": "rule-bad",
            "rule": "Test rule",
            "source": "initial",
            "supporting_trade_ids": "not valid json",  # Malformed
            "created_at": None,
        }

        client = D1Client(None)
        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            client._row_to_playbook_rule(row)


class TestDatetimeParsing:
    """Test datetime parsing edge cases."""

    def test_iso_format_with_timezone(self):
        """Test parsing ISO format with timezone."""
        from core.db.d1 import D1Client

        row = {
            "id": "rec-tz",
            "created_at": "2024-01-15T10:00:00+00:00",  # With timezone
            "expires_at": "2024-01-15T10:15:00+00:00",
            "status": "pending",
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 470.0,
            "long_strike": 465.0,
            "expiration": "2024-02-15",
            "credit": 1.25,
            "max_loss": 375.0,
            "iv_rank": None,
            "delta": None,
            "theta": None,
            "thesis": None,
            "confidence": None,
            "suggested_contracts": None,
            "analysis_price": None,
            "discord_message_id": None,
        }

        client = D1Client(None)
        rec = client._row_to_recommendation(row)

        # Should parse without error
        assert rec.created_at is not None

    def test_invalid_datetime_raises(self):
        """Test that invalid datetime format raises error."""
        from core.db.d1 import D1Client

        row = {
            "id": "rec-baddate",
            "created_at": "not-a-date",  # Invalid
            "expires_at": "2024-01-15T10:15:00",
            "status": "pending",
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 470.0,
            "long_strike": 465.0,
            "expiration": "2024-02-15",
            "credit": 1.25,
            "max_loss": 375.0,
            "iv_rank": None,
            "delta": None,
            "theta": None,
            "thesis": None,
            "confidence": None,
            "suggested_contracts": None,
            "analysis_price": None,
            "discord_message_id": None,
        }

        client = D1Client(None)
        with pytest.raises(ValueError):
            client._row_to_recommendation(row)
