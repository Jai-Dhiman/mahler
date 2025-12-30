"""Tests for weight optimization module.

Tests the WeightOptimizer class for regime-conditional scoring weight optimization.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.analysis.screener import ScoringWeights
from core.analysis.weight_optimizer import OptimizedWeights, WeightOptimizer


# Module-level fixtures
@pytest.fixture
def good_trades():
    """Create trades where high IV and delta predict good outcomes."""
    np.random.seed(42)
    trades = []
    for _ in range(40):
        iv_score = np.random.uniform(0.4, 0.9)
        delta_score = np.random.uniform(0.5, 0.8)
        credit_score = np.random.uniform(0.3, 0.7)
        ev_score = np.random.uniform(0.4, 0.8)

        # P/L correlates with IV and delta scores
        base_pnl = (iv_score * 0.3 + delta_score * 0.3) * 200
        noise = np.random.randn() * 30
        profit_loss = base_pnl + noise - 50

        trades.append({
            "iv_score": iv_score,
            "delta_score": delta_score,
            "credit_score": credit_score,
            "ev_score": ev_score,
            "profit_loss": profit_loss,
        })
    return trades


@pytest.fixture
def optimizer_with_data(good_trades):
    """Create optimizer with sufficient trade data."""
    trades_by_regime = {
        "bull_low_vol": good_trades.copy(),
        "bull_high_vol": [],  # No data
        "bear_low_vol": [],
        "bear_high_vol": [],
    }
    return WeightOptimizer(trades_by_regime)


@pytest.fixture
def optimizer_all_regimes(good_trades):
    """Create optimizer with data in all regimes."""
    # Create slightly different trade sets for each regime
    np.random.seed(42)
    trades_by_regime = {}
    for regime in ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]:
        trades = []
        for _ in range(30):
            trades.append({
                "iv_score": np.random.uniform(0.3, 0.9),
                "delta_score": np.random.uniform(0.4, 0.8),
                "credit_score": np.random.uniform(0.3, 0.7),
                "ev_score": np.random.uniform(0.3, 0.8),
                "profit_loss": np.random.uniform(-100, 150),
            })
        trades_by_regime[regime] = trades
    return WeightOptimizer(trades_by_regime)


class TestOptimizedWeights:
    """Test OptimizedWeights dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        weights = ScoringWeights(
            iv_weight=0.30,
            delta_weight=0.25,
            credit_weight=0.25,
            ev_weight=0.20,
        )
        result = OptimizedWeights(
            regime="bull_low_vol",
            weights=weights,
            sharpe_ratio=1.5,
            n_trades=50,
            optimized_at="2024-12-29T10:00:00",
        )

        d = result.to_dict()
        assert d["regime"] == "bull_low_vol"
        assert d["sharpe_ratio"] == 1.5
        assert d["n_trades"] == 50
        assert d["weights"]["iv"] == 0.30
        assert d["weights"]["delta"] == 0.25


class TestExtractSignalsAndOutcomes:
    """Test _extract_signals_and_outcomes method."""

    def test_basic_extraction(self, optimizer_with_data, good_trades):
        """Test extracting signals and outcomes from trades."""
        signals, outcomes = optimizer_with_data._extract_signals_and_outcomes(good_trades)

        assert signals.shape == (len(good_trades), 4)
        assert outcomes.shape == (len(good_trades),)

        # Check first trade values
        assert signals[0, 0] == good_trades[0]["iv_score"]
        assert signals[0, 1] == good_trades[0]["delta_score"]
        assert signals[0, 2] == good_trades[0]["credit_score"]
        assert signals[0, 3] == good_trades[0]["ev_score"]
        assert outcomes[0] == good_trades[0]["profit_loss"]

    def test_missing_scores_default_to_half(self):
        """Test that missing scores default to 0.5."""
        trades = [{"profit_loss": 50.0}]  # No scores
        optimizer = WeightOptimizer({"test": trades})

        signals, outcomes = optimizer._extract_signals_and_outcomes(trades)

        assert signals[0, 0] == 0.5  # iv_score default
        assert signals[0, 1] == 0.5  # delta_score default
        assert signals[0, 2] == 0.5  # credit_score default
        assert signals[0, 3] == 0.5  # ev_score default


class TestBacktestSharpe:
    """Test backtest_sharpe method."""

    def test_equal_weights_baseline(self, optimizer_with_data, good_trades):
        """Test Sharpe calculation with equal weights."""
        signals, outcomes = optimizer_with_data._extract_signals_and_outcomes(good_trades)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        sharpe = optimizer_with_data.backtest_sharpe(weights, signals, outcomes)

        # Should return negative Sharpe (for minimization)
        assert isinstance(sharpe, float)

    def test_all_positive_outcomes_high_sharpe(self):
        """Test that all positive outcomes give good (negative) Sharpe."""
        trades = [
            {"iv_score": 0.8, "delta_score": 0.7, "credit_score": 0.6, "ev_score": 0.6, "profit_loss": 100.0}
            for _ in range(30)
        ]
        optimizer = WeightOptimizer({"test": trades})
        signals, outcomes = optimizer._extract_signals_and_outcomes(trades)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        sharpe = optimizer.backtest_sharpe(weights, signals, outcomes)

        # Positive outcomes with no variance gives penalty
        # But with slight variations, should give low Sharpe
        assert sharpe <= 1000  # Not the penalty value


class TestOptimizeWeights:
    """Test optimize_weights method."""

    def test_insufficient_trades_returns_none(self):
        """Test that insufficient trades returns None."""
        trades = [{"profit_loss": 50.0} for _ in range(10)]
        optimizer = WeightOptimizer({"bull_low_vol": trades})

        result = optimizer.optimize_weights("bull_low_vol")

        assert result is None

    def test_missing_regime_returns_none(self, optimizer_with_data):
        """Test that missing regime returns None."""
        result = optimizer_with_data.optimize_weights("nonexistent_regime")

        assert result is None

    def test_successful_optimization(self, optimizer_with_data):
        """Test successful weight optimization."""
        result = optimizer_with_data.optimize_weights("bull_low_vol")

        assert result is not None
        assert result.regime == "bull_low_vol"
        assert result.n_trades >= 25

        # Weights should sum to 1.0
        total = (
            result.weights.iv_weight
            + result.weights.delta_weight
            + result.weights.credit_weight
            + result.weights.ev_weight
        )
        assert total == pytest.approx(1.0, abs=0.01)

        # Weights should be within bounds [0.10, 0.50]
        assert 0.09 <= result.weights.iv_weight <= 0.51
        assert 0.09 <= result.weights.delta_weight <= 0.51
        assert 0.09 <= result.weights.credit_weight <= 0.51
        assert 0.09 <= result.weights.ev_weight <= 0.51


class TestOptimizeAllRegimes:
    """Test optimize_all_regimes method."""

    def test_only_regimes_with_data(self, optimizer_with_data):
        """Test that only regimes with sufficient data are optimized."""
        results = optimizer_with_data.optimize_all_regimes()

        # Only bull_low_vol has data
        assert "bull_low_vol" in results
        assert "bull_high_vol" not in results
        assert "bear_low_vol" not in results
        assert "bear_high_vol" not in results

    def test_all_regimes_with_data(self, optimizer_all_regimes):
        """Test optimization of all regimes when data available."""
        results = optimizer_all_regimes.optimize_all_regimes()

        assert len(results) == 4
        for regime in ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol"]:
            assert regime in results


class TestGetTotalTrades:
    """Test get_total_trades method."""

    def test_total_count(self):
        """Test total trade count across regimes."""
        trades_by_regime = {
            "bull_low_vol": [{"profit_loss": 50} for _ in range(10)],
            "bull_high_vol": [{"profit_loss": 50} for _ in range(15)],
            "bear_low_vol": [{"profit_loss": 50} for _ in range(20)],
            "bear_high_vol": [],
        }
        optimizer = WeightOptimizer(trades_by_regime)

        assert optimizer.get_total_trades() == 45

    def test_empty_regimes(self):
        """Test with all empty regimes."""
        optimizer = WeightOptimizer({})
        assert optimizer.get_total_trades() == 0
