"""Tests for dynamic exit manager."""

import pytest

from core.risk.validators import (
    DynamicExitCalculator,
    DynamicExitResult,
    ExitConfig,
    ExitValidator,
    TradingStyle,
)


@pytest.fixture
def default_config():
    """Create default exit configuration."""
    return ExitConfig()


@pytest.fixture
def calculator(default_config):
    """Create a dynamic exit calculator with default config."""
    return DynamicExitCalculator(default_config)


class TestTradingStyleMultipliers:
    """Test that trading styles have correct multipliers."""

    def test_aggressive_multipliers(self, calculator):
        """Aggressive style should have wider targets."""
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.AGGRESSIVE,
        )

        # Aggressive: tp=0.6, sl=1.5
        assert result.base_profit_target == pytest.approx(0.60)
        assert result.base_stop_loss == pytest.approx(1.50)

    def test_neutral_multipliers(self, calculator):
        """Neutral style should have balanced targets."""
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
        )

        # Neutral: tp=0.5, sl=1.25
        assert result.base_profit_target == pytest.approx(0.50)
        assert result.base_stop_loss == pytest.approx(1.25)

    def test_conservative_multipliers(self, calculator):
        """Conservative style should have tighter targets."""
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.CONSERVATIVE,
        )

        # Conservative: tp=0.4, sl=1.0
        assert result.base_profit_target == pytest.approx(0.40)
        assert result.base_stop_loss == pytest.approx(1.00)


class TestVolatilityAdjustment:
    """Test volatility-based exit adjustments."""

    def test_high_volatility_tightens_profit_target(self, calculator):
        """Higher volatility should result in tighter profit targets."""
        # Base vol is 0.15 (15%), use 0.30 (30%)
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.30,  # 2x baseline
        )

        # Base tp = 0.5, with 2x vol ratio
        # Adjusted tp = 0.5 / 2.0 = 0.25
        assert result.profit_target == pytest.approx(0.25)
        assert result.vol_adjustment == pytest.approx(2.0)

    def test_high_volatility_widens_stop_loss(self, calculator):
        """Higher volatility should result in wider stop losses."""
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.30,  # 2x baseline
        )

        # Base sl = 1.25, with 2x vol ratio
        # Adjusted sl = 1.25 * 2.0 = 2.50
        assert result.stop_loss == pytest.approx(2.50)

    def test_low_volatility_widens_profit_target(self, calculator):
        """Lower volatility should allow profit targets to run longer."""
        # Use 0.075 (7.5% vol = 0.5x baseline)
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.075,  # 0.5x baseline
        )

        # Base tp = 0.5, with 0.5x vol ratio
        # Adjusted tp = 0.5 / 0.5 = 1.0
        assert result.profit_target == pytest.approx(1.0)
        assert result.vol_adjustment == pytest.approx(0.5)

    def test_vol_ratio_clamped_at_bounds(self, calculator):
        """Vol ratio should be clamped between 0.5 and 2.0."""
        # Very high vol (0.45 = 3x baseline, clamped to 2.0)
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.45,
        )
        assert result.vol_adjustment == pytest.approx(2.0)

        # Very low vol (0.05 = 0.33x baseline, clamped to 0.5)
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.05,
        )
        assert result.vol_adjustment == pytest.approx(0.5)

    def test_no_vol_adjustment_when_disabled(self):
        """Vol adjustment should not apply when disabled."""
        config = ExitConfig(vol_10d_adjustment_enabled=False)
        calculator = DynamicExitCalculator(config)

        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.30,  # Would normally adjust
        )

        # Should use base values
        assert result.profit_target == result.base_profit_target
        assert result.stop_loss == result.base_stop_loss
        assert result.vol_adjustment == 1.0

    def test_no_vol_adjustment_when_vol_none(self, calculator):
        """Vol adjustment should not apply when vol_10d is None."""
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=None,
        )

        assert result.profit_target == result.base_profit_target
        assert result.stop_loss == result.base_stop_loss


class TestExitPriceCalculation:
    """Test calculation of actual exit prices."""

    def test_exit_prices_neutral_no_vol(self, calculator):
        """Exit prices should be calculated correctly."""
        entry_credit = 1.00

        tp_value, sl_value = calculator.calculate_exit_prices(
            entry_credit=entry_credit,
            trading_style=TradingStyle.NEUTRAL,
        )

        # Profit target: close when value drops to 0.50 (captured 0.50 profit)
        assert tp_value == pytest.approx(0.50)

        # Stop loss: close when value rises to 2.25 (lost 1.25)
        assert sl_value == pytest.approx(2.25)

    def test_exit_prices_with_vol_adjustment(self, calculator):
        """Exit prices should incorporate volatility adjustment."""
        entry_credit = 1.00

        tp_value, sl_value = calculator.calculate_exit_prices(
            entry_credit=entry_credit,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.30,  # 2x baseline
        )

        # With 2x vol: tp=0.25, sl=2.50
        # TP value = 1.00 - 0.25 = 0.75 (exit faster)
        assert tp_value == pytest.approx(0.75)

        # SL value = 1.00 + 2.50 = 3.50 (wider stop)
        assert sl_value == pytest.approx(3.50)


class TestExitValidatorIntegration:
    """Test integration with ExitValidator."""

    def test_dynamic_exit_in_check_all_conditions(self):
        """ExitValidator should use dynamic exits when style provided."""
        # Disable gamma protection to isolate dynamic exit testing
        config = ExitConfig(gamma_protection_enabled=False)
        validator = ExitValidator(config)

        entry_credit = 1.00
        # Position has achieved 55% profit (current_value = 0.45)
        current_value = 0.45

        # Use far future expiration to avoid time exits
        from datetime import datetime, timedelta
        future_exp = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

        # Check with aggressive style (tp multiplier = 0.6)
        should_exit, reason, _ = validator.check_all_exit_conditions(
            entry_credit=entry_credit,
            current_value=current_value,
            expiration=future_exp,
            trading_style=TradingStyle.AGGRESSIVE,
        )

        # 55% profit should not trigger aggressive exit (target = 60%)
        assert not should_exit

    def test_dynamic_exit_triggers_at_target(self):
        """Exit should trigger when profit target is reached."""
        config = ExitConfig(gamma_protection_enabled=False)
        validator = ExitValidator(config)

        entry_credit = 1.00
        # Position has achieved 65% profit (current_value = 0.35)
        current_value = 0.35

        from datetime import datetime, timedelta
        future_exp = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

        should_exit, reason, _ = validator.check_all_exit_conditions(
            entry_credit=entry_credit,
            current_value=current_value,
            expiration=future_exp,
            trading_style=TradingStyle.AGGRESSIVE,
        )

        # 65% profit should trigger aggressive exit (target = 60%)
        assert should_exit
        assert "dynamic_profit" in reason

    def test_dynamic_stop_loss_triggers(self):
        """Stop loss should trigger based on dynamic calculation."""
        config = ExitConfig(gamma_protection_enabled=False)
        validator = ExitValidator(config)

        entry_credit = 1.00
        # Position has 160% loss (current_value = 2.60)
        current_value = 2.60

        from datetime import datetime, timedelta
        future_exp = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

        should_exit, reason, _ = validator.check_all_exit_conditions(
            entry_credit=entry_credit,
            current_value=current_value,
            expiration=future_exp,
            trading_style=TradingStyle.AGGRESSIVE,  # sl = 1.5
        )

        # 160% loss should trigger aggressive stop (target = 150%)
        assert should_exit
        assert "dynamic_stop_loss" in reason

    def test_vol_adjustment_in_exit_check(self):
        """Vol adjustment should affect exit checks."""
        config = ExitConfig(gamma_protection_enabled=False)
        validator = ExitValidator(config)

        entry_credit = 1.00
        # 40% profit normally wouldn't trigger conservative (40% target)
        # But with high vol, target tightens
        current_value = 0.60  # 40% profit

        from datetime import datetime, timedelta
        future_exp = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

        should_exit, reason, _ = validator.check_all_exit_conditions(
            entry_credit=entry_credit,
            current_value=current_value,
            expiration=future_exp,
            trading_style=TradingStyle.CONSERVATIVE,
            vol_10d=0.30,  # 2x baseline, tightens target to 20%
        )

        # With high vol, 40% profit exceeds the 20% target
        assert should_exit
        assert "dynamic_profit" in reason


class TestSerialization:
    """Test result serialization."""

    def test_dynamic_exit_result_to_dict(self, calculator):
        """DynamicExitResult should serialize to dictionary."""
        result = calculator.calculate_exits(
            entry_credit=1.00,
            trading_style=TradingStyle.NEUTRAL,
            vol_10d=0.20,
        )

        d = result.to_dict()

        assert "profit_target" in d
        assert "stop_loss" in d
        assert "trading_style" in d
        assert d["trading_style"] == "neutral"
        assert "vol_adjustment" in d
