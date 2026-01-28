"""Tests for three-perspective risk manager."""

import pytest
from unittest.mock import MagicMock, patch

from core.risk.three_perspective import (
    ThreePerspectiveRiskManager,
    ThreePerspectiveConfig,
    ThreePerspectiveResult,
    PerspectiveAssessment,
    RiskPerspective,
)
from core.risk.position_sizer import PositionSizer, PositionSizeResult, RiskLimits
from core.types import CreditSpread, SpreadType


@pytest.fixture
def mock_sizer():
    """Create a mock position sizer that returns predictable results."""
    sizer = MagicMock(spec=PositionSizer)
    sizer.limits = RiskLimits()

    # Default behavior: return 10 contracts
    sizer.calculate_size.return_value = PositionSizeResult(
        contracts=10,
        risk_amount=500.0,
        risk_percent=0.01,
        reason=None,
    )
    return sizer


@pytest.fixture
def sample_spread():
    """Create a sample credit spread."""
    spread = MagicMock(spec=CreditSpread)
    spread.underlying = "SPY"
    spread.spread_type = SpreadType.BULL_PUT
    spread.short_strike = 450.0
    spread.long_strike = 445.0
    spread.expiration = "2024-02-15"
    spread.credit = 0.50
    spread.max_loss = 450.0
    return spread


class TestThreePerspectiveWeights:
    """Test VIX-based weight calculations."""

    def test_high_vix_weights_conservative_heavy(self, mock_sizer):
        """VIX > 30 should weight conservative perspective heavily."""
        manager = ThreePerspectiveRiskManager(mock_sizer)
        weights = manager._get_weights_for_vix(35.0)

        assert weights[RiskPerspective.AGGRESSIVE] == 0.1
        assert weights[RiskPerspective.NEUTRAL] == 0.3
        assert weights[RiskPerspective.CONSERVATIVE] == 0.6

    def test_moderate_vix_weights_neutral_heavy(self, mock_sizer):
        """VIX 20-30 should weight neutral perspective heavily."""
        manager = ThreePerspectiveRiskManager(mock_sizer)
        weights = manager._get_weights_for_vix(25.0)

        assert weights[RiskPerspective.AGGRESSIVE] == 0.2
        assert weights[RiskPerspective.NEUTRAL] == 0.5
        assert weights[RiskPerspective.CONSERVATIVE] == 0.3

    def test_low_vix_weights_balanced(self, mock_sizer):
        """VIX <= 20 should be more balanced."""
        manager = ThreePerspectiveRiskManager(mock_sizer)
        weights = manager._get_weights_for_vix(15.0)

        assert weights[RiskPerspective.AGGRESSIVE] == 0.3
        assert weights[RiskPerspective.NEUTRAL] == 0.5
        assert weights[RiskPerspective.CONSERVATIVE] == 0.2

    def test_custom_config_weights(self, mock_sizer):
        """Custom configuration should override default weights."""
        config = ThreePerspectiveConfig(
            high_vix_weights=[0.0, 0.2, 0.8],
            moderate_vix_weights=[0.1, 0.6, 0.3],
            low_vix_weights=[0.4, 0.4, 0.2],
        )
        manager = ThreePerspectiveRiskManager(mock_sizer, config)

        # High VIX
        weights = manager._get_weights_for_vix(40.0)
        assert weights[RiskPerspective.AGGRESSIVE] == 0.0
        assert weights[RiskPerspective.CONSERVATIVE] == 0.8


class TestPerspectiveAssessment:
    """Test individual perspective assessments."""

    def test_aggressive_uses_full_limits(self, mock_sizer, sample_spread):
        """Aggressive perspective should use full position limits."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=20.0,
        )

        # Aggressive uses 1.0 multiplier
        assert result.aggressive.position_size_multiplier == 1.0
        assert result.aggressive.recommended_contracts == 10

    def test_neutral_uses_reduced_limits(self, mock_sizer, sample_spread):
        """Neutral perspective should use 75% of limits."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=20.0,
        )

        # Neutral uses 0.75 multiplier
        assert result.neutral.position_size_multiplier == 0.75
        assert result.neutral.recommended_contracts == 7  # 10 * 0.75 = 7.5 -> 7

    def test_conservative_uses_half_limits(self, mock_sizer, sample_spread):
        """Conservative perspective should use 50% of limits."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=20.0,
        )

        # Conservative uses 0.50 multiplier
        assert result.conservative.position_size_multiplier == 0.50
        assert result.conservative.recommended_contracts == 5  # 10 * 0.5 = 5


class TestWeightedContracts:
    """Test weighted contract calculations."""

    def test_low_vix_weighted_contracts(self, mock_sizer, sample_spread):
        """Low VIX should weight towards larger positions."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=15.0,
        )

        # Weights: aggressive=0.3, neutral=0.5, conservative=0.2
        # Contracts: 10, 7, 5
        # Weighted: 10*0.3 + 7*0.5 + 5*0.2 = 3 + 3.5 + 1 = 7.5 -> 8
        assert result.weighted_contracts == 8
        assert result.vix_at_assessment == 15.0

    def test_high_vix_weighted_contracts(self, mock_sizer, sample_spread):
        """High VIX should weight towards smaller positions."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=35.0,
        )

        # Weights: aggressive=0.1, neutral=0.3, conservative=0.6
        # Contracts: 10, 7, 5
        # Weighted: 10*0.1 + 7*0.3 + 5*0.6 = 1 + 2.1 + 3 = 6.1 -> 6
        assert result.weighted_contracts == 6


class TestConservativeSkipRespect:
    """Test respecting conservative skip in high VIX."""

    def test_should_respect_skip_in_high_vix(self, mock_sizer, sample_spread):
        """Conservative skip should be respected when VIX > 30."""
        # Make sizer return 0 contracts (blocked by limits)
        mock_sizer.calculate_size.return_value = PositionSizeResult(
            contracts=0,
            risk_amount=0,
            risk_percent=0,
            reason="Portfolio heat limit reached",
        )

        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=35.0,
        )

        # Conservative recommends skip when base contracts = 0
        assert result.conservative.recommendation == "skip"
        assert manager.should_respect_conservative_skip(result, vix_threshold=30.0)

    def test_should_not_respect_skip_in_low_vix(self, mock_sizer, sample_spread):
        """Conservative skip should not be respected when VIX <= 30."""
        mock_sizer.calculate_size.return_value = PositionSizeResult(
            contracts=0,
            risk_amount=0,
            risk_percent=0,
            reason="Portfolio heat limit reached",
        )

        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=20.0,
        )

        # Should not respect conservative skip when VIX is low
        assert not manager.should_respect_conservative_skip(result, vix_threshold=30.0)


class TestDeliberationSummary:
    """Test deliberation summary generation."""

    def test_deliberation_summary_includes_vix(self, mock_sizer, sample_spread):
        """Deliberation summary should include VIX regime."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=35.0,
        )

        assert "VIX 35.0" in result.deliberation_summary
        assert "conservative-heavy" in result.deliberation_summary

    def test_deliberation_summary_includes_contracts(self, mock_sizer, sample_spread):
        """Deliberation summary should include contract counts."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=20.0,
        )

        assert "contracts" in result.deliberation_summary.lower()
        assert "Weighted result" in result.deliberation_summary


class TestSerialization:
    """Test result serialization."""

    def test_to_dict(self, mock_sizer, sample_spread):
        """Result should serialize to dictionary."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=25.0,
        )

        d = result.to_dict()

        assert "aggressive" in d
        assert "neutral" in d
        assert "conservative" in d
        assert "weighted_contracts" in d
        assert "weights_used" in d
        assert d["vix_at_assessment"] == 25.0

    def test_consensus_recommendation(self, mock_sizer, sample_spread):
        """Consensus recommendation should reflect weighted outcome."""
        manager = ThreePerspectiveRiskManager(mock_sizer)

        result = manager.assess(
            spread=sample_spread,
            account_equity=50000.0,
            current_positions=[],
            current_vix=20.0,
        )

        # With contracts > 0 but < aggressive contracts, should be "reduce_size"
        # weighted=8, aggressive=10, so it's a reduction
        assert result.consensus_recommendation in ("enter", "reduce_size")
        assert result.weighted_contracts > 0
