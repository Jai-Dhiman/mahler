"""Tests for trade validators and risk management.

These tests ensure validation logic handles edge cases correctly,
particularly around expiration dates, price drift, and exit conditions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.types import Recommendation, RecommendationStatus, SpreadType, Confidence


class TestTradeValidatorRecommendation:
    """Test TradeValidator.validate_recommendation()."""

    @pytest.fixture
    def valid_recommendation(self, future_date):
        """Create a valid recommendation for testing."""
        return Recommendation(
            id="rec-123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=15),
            status=RecommendationStatus.PENDING,
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration=future_date,
            credit=1.25,
            max_loss=375.0,
            iv_rank=55.0,
            delta=-0.25,
            theta=0.03,
            thesis="Test thesis",
            confidence=Confidence.MEDIUM,
            suggested_contracts=2,
            analysis_price=598.50,
        )

    def test_valid_recommendation_passes(self, valid_recommendation):
        """Test that valid recommendation passes validation."""
        from core.risk.validators import TradeValidator

        validator = TradeValidator()
        result = validator.validate_recommendation(valid_recommendation)

        assert result.valid is True
        assert result.reason is None

    def test_non_pending_status_fails(self, valid_recommendation):
        """Test that non-pending status fails validation."""
        from core.risk.validators import TradeValidator

        valid_recommendation.status = RecommendationStatus.APPROVED

        validator = TradeValidator()
        result = validator.validate_recommendation(valid_recommendation)

        assert result.valid is False
        assert "approved" in result.reason.lower()

    def test_expired_recommendation_fails(self, valid_recommendation):
        """Test that expired recommendation fails validation."""
        from core.risk.validators import TradeValidator

        valid_recommendation.expires_at = datetime.now() - timedelta(minutes=5)

        validator = TradeValidator()
        result = validator.validate_recommendation(valid_recommendation)

        assert result.valid is False
        assert "expired" in result.reason.lower()

    def test_low_dte_fails(self, valid_recommendation, near_expiry_date):
        """Test that low DTE (below 21) fails validation."""
        from core.risk.validators import TradeValidator

        valid_recommendation.expiration = near_expiry_date

        validator = TradeValidator()
        result = validator.validate_recommendation(valid_recommendation)

        assert result.valid is False
        assert "DTE" in result.reason

    def test_past_expiration_fails(self, valid_recommendation, past_date):
        """Test that past expiration fails validation."""
        from core.risk.validators import TradeValidator

        valid_recommendation.expiration = past_date

        validator = TradeValidator()
        result = validator.validate_recommendation(valid_recommendation)

        assert result.valid is False


class TestTradeValidatorPriceDrift:
    """Test TradeValidator.validate_price_drift()."""

    @pytest.fixture
    def recommendation_with_price(self, future_date):
        """Create recommendation with analysis price."""
        return Recommendation(
            id="rec-123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=15),
            status=RecommendationStatus.PENDING,
            underlying="SPY",
            spread_type=SpreadType.BULL_PUT,
            short_strike=470.0,
            long_strike=465.0,
            expiration=future_date,
            credit=1.25,
            max_loss=375.0,
            analysis_price=1.25,  # Original price at analysis
        )

    def test_no_drift_passes(self, recommendation_with_price):
        """Test that no drift passes validation."""
        from core.risk.validators import TradeValidator

        validator = TradeValidator()
        result = validator.validate_price_drift(recommendation_with_price, 1.25)

        assert result.valid is True

    def test_small_drift_passes(self, recommendation_with_price):
        """Test that small drift (within 1%) passes."""
        from core.risk.validators import TradeValidator

        validator = TradeValidator()
        # 1.25 * 0.005 = 0.00625 drift
        result = validator.validate_price_drift(recommendation_with_price, 1.256)

        assert result.valid is True

    def test_large_drift_fails(self, recommendation_with_price):
        """Test that large drift (>1%) fails."""
        from core.risk.validators import TradeValidator

        validator = TradeValidator()
        # 1.25 * 0.02 = 0.025 drift = 2% - exceeds 1%
        result = validator.validate_price_drift(recommendation_with_price, 1.28)

        assert result.valid is False
        assert "drift" in result.reason.lower()

    def test_no_analysis_price_passes(self, recommendation_with_price):
        """Test that missing analysis price passes (can't validate)."""
        from core.risk.validators import TradeValidator

        recommendation_with_price.analysis_price = None

        validator = TradeValidator()
        result = validator.validate_price_drift(recommendation_with_price, 1.50)

        assert result.valid is True

    def test_zero_analysis_price_fails(self, recommendation_with_price):
        """Test that zero analysis price fails validation."""
        from core.risk.validators import TradeValidator

        recommendation_with_price.analysis_price = 0.0

        validator = TradeValidator()
        result = validator.validate_price_drift(recommendation_with_price, 1.25)

        assert result.valid is False
        assert "invalid" in result.reason.lower()


class TestExitValidator:
    """Test ExitValidator exit condition checks."""

    def test_profit_target_reached(self):
        """Test profit target detection at 50% of max."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()
        # Entry credit: 1.00, current value: 0.40 = 60% profit
        result = validator.check_profit_target(entry_credit=1.00, current_value=0.40)

        assert result.valid is True
        assert "profit" in result.reason.lower()

    def test_profit_target_not_reached(self):
        """Test profit target not reached."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()
        # Entry credit: 1.00, current value: 0.80 = 20% profit
        result = validator.check_profit_target(entry_credit=1.00, current_value=0.80)

        assert result.valid is False

    def test_stop_loss_triggered(self):
        """Test stop loss detection at 200% of credit."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()
        # Entry credit: 1.00, current value: 3.50 = 250% loss
        result = validator.check_stop_loss(entry_credit=1.00, current_value=3.50)

        assert result.valid is True
        assert "stop" in result.reason.lower()

    def test_stop_loss_not_triggered(self):
        """Test stop loss not triggered."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()
        # Entry credit: 1.00, current value: 2.00 = 100% loss (below 200%)
        result = validator.check_stop_loss(entry_credit=1.00, current_value=2.00)

        assert result.valid is False

    def test_time_exit_triggered(self, near_expiry_date):
        """Test time exit at 21 DTE."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()
        result = validator.check_time_exit(near_expiry_date)

        assert result.valid is True
        assert "time" in result.reason.lower()

    def test_time_exit_not_triggered(self, future_date):
        """Test time exit not triggered (>21 DTE)."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()
        result = validator.check_time_exit(future_date)

        assert result.valid is False

    def test_zero_entry_credit_handled(self):
        """Test handling of zero entry credit edge case."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()

        profit_result = validator.check_profit_target(entry_credit=0.0, current_value=0.50)
        assert profit_result.valid is False
        assert "invalid" in profit_result.reason.lower()

        stop_result = validator.check_stop_loss(entry_credit=0.0, current_value=0.50)
        assert stop_result.valid is False

    def test_negative_entry_credit_handled(self):
        """Test handling of negative entry credit edge case."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()

        result = validator.check_profit_target(entry_credit=-1.0, current_value=0.50)
        assert result.valid is False

    def test_check_all_exit_conditions(self, future_date):
        """Test combined exit condition check."""
        from core.risk.validators import ExitValidator

        validator = ExitValidator()

        # No exit condition met
        should_exit, reason = validator.check_all_exit_conditions(
            entry_credit=1.00,
            current_value=0.80,  # 20% profit
            expiration=future_date,  # >21 DTE
        )
        assert should_exit is False
        assert reason is None

        # Profit target met
        should_exit, reason = validator.check_all_exit_conditions(
            entry_credit=1.00,
            current_value=0.40,  # 60% profit
            expiration=future_date,
        )
        assert should_exit is True
        assert "profit" in reason.lower()

    def test_win_rate_adjustment(self):
        """Test stop loss adjustment based on win rate."""
        from core.risk.validators import ExitValidator, ExitConfig

        config = ExitConfig(
            stop_loss_pct=2.00,  # Default 200%
            tighter_stop_loss_pct=1.50,  # Tighter 150%
            win_rate_threshold=0.80,
        )
        validator = ExitValidator(config)

        # Win rate above threshold - no adjustment
        validator.adjust_for_win_rate(0.85)
        assert validator.config.stop_loss_pct == 2.00

        # Win rate below threshold - use tighter stop
        validator.adjust_for_win_rate(0.70)
        assert validator.config.stop_loss_pct == 1.50
