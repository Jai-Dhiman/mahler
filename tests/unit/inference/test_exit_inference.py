"""Tests for exit inference module."""

import pytest

from core.inference.exit_inference import ExitParams, PrecomputedExitProvider
from core.inference.model_loader import ExitModelParams


@pytest.fixture
def sample_params():
    """Create sample exit model parameters."""
    return ExitModelParams(
        version="1.0.0",
        trained_at="2025-01-06T10:00:00",
        n_samples=200,
        profit_target=0.45,
        stop_loss=0.18,
        time_exit_dte=18,
        sharpe_ratio=1.8,
    )


class TestExitParams:
    """Test ExitParams dataclass."""

    def test_creation(self):
        """Verify ExitParams can be created."""
        params = ExitParams(
            profit_target=0.50,
            stop_loss=0.20,
            time_exit_dte=21,
        )

        assert params.profit_target == 0.50
        assert params.stop_loss == 0.20
        assert params.time_exit_dte == 21


class TestPrecomputedExitProvider:
    """Test PrecomputedExitProvider class."""

    def test_get_exit_params_with_params(self, sample_params):
        """Verify get_exit_params returns optimized parameters."""
        provider = PrecomputedExitProvider(sample_params)

        params = provider.get_exit_params()

        assert isinstance(params, ExitParams)
        assert params.profit_target == 0.45
        assert params.stop_loss == 0.18
        assert params.time_exit_dte == 18

    def test_get_exit_params_with_none_params_returns_defaults(self):
        """Verify None params returns default parameters."""
        provider = PrecomputedExitProvider(None)

        params = provider.get_exit_params()

        assert params == provider.DEFAULT_PARAMS
        assert params.profit_target == 0.65
        assert params.stop_loss == 1.25
        assert params.time_exit_dte == 21

    def test_get_sharpe_ratio(self, sample_params):
        """Verify get_sharpe_ratio returns correct value."""
        provider = PrecomputedExitProvider(sample_params)

        sharpe = provider.get_sharpe_ratio()

        assert sharpe == 1.8

    def test_get_sharpe_ratio_with_none_params(self):
        """Verify None params returns None for Sharpe."""
        provider = PrecomputedExitProvider(None)

        sharpe = provider.get_sharpe_ratio()

        assert sharpe is None

    def test_has_optimized_params(self, sample_params):
        """Verify has_optimized_params works correctly."""
        provider_with = PrecomputedExitProvider(sample_params)
        provider_without = PrecomputedExitProvider(None)

        assert provider_with.has_optimized_params() is True
        assert provider_without.has_optimized_params() is False

    def test_default_params_are_correct(self):
        """Verify DEFAULT_PARAMS has correct backtest-validated values."""
        expected = ExitParams(
            profit_target=0.65,  # 65% of max profit (backtest validated)
            stop_loss=1.25,  # 125% of credit (backtest validated)
            time_exit_dte=21,  # 21 DTE
        )

        assert PrecomputedExitProvider.DEFAULT_PARAMS == expected
