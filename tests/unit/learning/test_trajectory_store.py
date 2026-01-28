"""Tests for trajectory store."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from core.learning.trajectory_store import TradeTrajectory, TrajectoryStore


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    return TradeTrajectory(
        id="test-traj-123",
        underlying="SPY",
        spread_type="bull_put",
        short_strike=450.0,
        long_strike=445.0,
        expiration="2024-02-15",
        entry_credit=0.50,
        contracts=5,
        market_regime="trending_up",
        iv_rank=65.0,
        vix_at_entry=18.5,
        analyst_outputs=[
            {"agent_id": "iv_analyst", "content": "IV elevated", "confidence": 0.8}
        ],
        debate_transcript=[
            {"agent_id": "bull", "content": "Good setup", "confidence": 0.7}
        ],
        synthesis_output={"recommendation": "enter", "confidence": 0.75},
        decision_output={"decision": "enter", "position_size": 5},
    )


@pytest.fixture
def mock_db():
    """Create a mock D1 database binding."""
    db = MagicMock()

    # Mock prepare method
    stmt = MagicMock()
    stmt.bind = MagicMock(return_value=stmt)
    stmt.run = AsyncMock(return_value={"success": True})
    stmt.all = AsyncMock(return_value={"results": []})
    stmt.first = AsyncMock(return_value=None)

    db.prepare = MagicMock(return_value=stmt)

    return db


class TestTradeTrajectory:
    """Test TradeTrajectory dataclass."""

    def test_default_values(self):
        """Trajectory should have sensible defaults."""
        traj = TradeTrajectory()

        assert traj.id is not None
        assert traj.created_at is not None
        assert traj.underlying == ""
        assert traj.contracts == 0
        assert traj.analyst_outputs == []
        assert traj.has_outcome is False
        assert traj.has_label is False

    def test_has_outcome_property(self, sample_trajectory):
        """has_outcome should reflect actual_pnl presence."""
        assert sample_trajectory.has_outcome is False

        sample_trajectory.actual_pnl = 150.0
        assert sample_trajectory.has_outcome is True

    def test_has_label_property(self, sample_trajectory):
        """has_label should reflect reward_label presence."""
        assert sample_trajectory.has_label is False

        sample_trajectory.reward_label = "positive"
        assert sample_trajectory.has_label is True

    def test_to_dict_serialization(self, sample_trajectory):
        """Trajectory should serialize to dictionary."""
        d = sample_trajectory.to_dict()

        assert d["id"] == "test-traj-123"
        assert d["underlying"] == "SPY"
        assert d["spread_type"] == "bull_put"
        assert d["short_strike"] == 450.0
        assert d["contracts"] == 5
        assert d["market_regime"] == "trending_up"
        assert d["vix_at_entry"] == 18.5

        # JSON-encoded fields
        analyst_outputs = json.loads(d["analyst_outputs"])
        assert len(analyst_outputs) == 1
        assert analyst_outputs[0]["agent_id"] == "iv_analyst"

    def test_from_dict_deserialization(self, sample_trajectory):
        """Trajectory should deserialize from dictionary."""
        d = sample_trajectory.to_dict()
        restored = TradeTrajectory.from_dict(d)

        assert restored.id == sample_trajectory.id
        assert restored.underlying == sample_trajectory.underlying
        assert restored.short_strike == sample_trajectory.short_strike
        assert restored.contracts == sample_trajectory.contracts
        assert restored.market_regime == sample_trajectory.market_regime
        assert len(restored.analyst_outputs) == 1

    def test_roundtrip_serialization(self, sample_trajectory):
        """Serialize then deserialize should preserve data."""
        d = sample_trajectory.to_dict()
        restored = TradeTrajectory.from_dict(d)

        # Compare key fields
        assert restored.id == sample_trajectory.id
        assert restored.underlying == sample_trajectory.underlying
        assert restored.spread_type == sample_trajectory.spread_type
        assert restored.short_strike == sample_trajectory.short_strike
        assert restored.long_strike == sample_trajectory.long_strike
        assert restored.entry_credit == sample_trajectory.entry_credit
        assert restored.iv_rank == sample_trajectory.iv_rank


class TestTrajectoryStore:
    """Test TrajectoryStore database operations."""

    @pytest.mark.asyncio
    async def test_store_trajectory(self, mock_db, sample_trajectory):
        """Store trajectory should insert into database."""
        store = TrajectoryStore(mock_db)

        result_id = await store.store_trajectory(sample_trajectory)

        assert result_id == sample_trajectory.id
        mock_db.prepare.assert_called_once()

        # Verify INSERT statement
        call_args = mock_db.prepare.call_args[0][0]
        assert "INSERT INTO trade_trajectories" in call_args

    @pytest.mark.asyncio
    async def test_update_outcome(self, mock_db):
        """Update outcome should update database record."""
        store = TrajectoryStore(mock_db)

        await store.update_outcome(
            trajectory_id="test-123",
            actual_pnl=150.0,
            actual_pnl_percent=0.30,
            exit_reason="profit_target",
            days_held=14,
        )

        mock_db.prepare.assert_called_once()

        # Verify UPDATE statement
        call_args = mock_db.prepare.call_args[0][0]
        assert "UPDATE trade_trajectories" in call_args
        assert "actual_pnl" in call_args

    @pytest.mark.asyncio
    async def test_update_labels(self, mock_db):
        """Update labels should update database record."""
        store = TrajectoryStore(mock_db)

        await store.update_labels(
            trajectory_id="test-123",
            reward_label="strong_positive",
            reward_score=0.05,
        )

        mock_db.prepare.assert_called_once()

        # Verify UPDATE statement
        call_args = mock_db.prepare.call_args[0][0]
        assert "UPDATE trade_trajectories" in call_args
        assert "reward_label" in call_args

    @pytest.mark.asyncio
    async def test_get_unlabeled_trajectories(self, mock_db, sample_trajectory):
        """Get unlabeled trajectories should query correctly."""
        # Set up mock to return sample trajectory
        stmt = mock_db.prepare.return_value
        sample_dict = sample_trajectory.to_dict()
        stmt.all = AsyncMock(return_value={"results": [sample_dict]})

        store = TrajectoryStore(mock_db)

        trajectories = await store.get_unlabeled_trajectories(with_outcomes=True, limit=50)

        assert len(trajectories) == 1
        assert trajectories[0].id == sample_trajectory.id

        # Verify query conditions
        call_args = mock_db.prepare.call_args[0][0]
        assert "reward_label IS NULL" in call_args
        assert "actual_pnl IS NOT NULL" in call_args

    @pytest.mark.asyncio
    async def test_get_trajectory(self, mock_db, sample_trajectory):
        """Get trajectory should return single record."""
        stmt = mock_db.prepare.return_value
        stmt.first = AsyncMock(return_value=sample_trajectory.to_dict())

        store = TrajectoryStore(mock_db)

        result = await store.get_trajectory("test-123")

        assert result is not None
        assert result.id == sample_trajectory.id

    @pytest.mark.asyncio
    async def test_get_trajectory_not_found(self, mock_db):
        """Get trajectory should return None when not found."""
        stmt = mock_db.prepare.return_value
        stmt.first = AsyncMock(return_value=None)

        store = TrajectoryStore(mock_db)

        result = await store.get_trajectory("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_trajectory_by_trade_id(self, mock_db, sample_trajectory):
        """Get trajectory by trade ID should work."""
        sample_trajectory.trade_id = "trade-456"
        stmt = mock_db.prepare.return_value
        stmt.first = AsyncMock(return_value=sample_trajectory.to_dict())

        store = TrajectoryStore(mock_db)

        result = await store.get_trajectory_by_trade_id("trade-456")

        assert result is not None
        assert result.trade_id == "trade-456"


class TestTrajectoryFromPipelineResult:
    """Test creating trajectories from pipeline results."""

    def test_from_pipeline_result(self):
        """Should create trajectory from pipeline outputs."""
        # Mock agent messages
        analyst_msg = MagicMock()
        analyst_msg.agent_id = "iv_analyst"
        analyst_msg.content = "IV is elevated"
        analyst_msg.structured_data = {"iv_rank": 65}
        analyst_msg.confidence = 0.8

        debate_msg = MagicMock()
        debate_msg.agent_id = "bull"
        debate_msg.message_type = MagicMock(value="argument")
        debate_msg.content = "Good setup"
        debate_msg.structured_data = {}
        debate_msg.confidence = 0.7

        synthesis_msg = MagicMock()
        synthesis_msg.agent_id = "facilitator"
        synthesis_msg.content = "Consensus reached"
        synthesis_msg.structured_data = {"recommendation": "enter"}
        synthesis_msg.confidence = 0.75

        # Mock three-perspective result
        three_persp = MagicMock()
        three_persp.to_dict.return_value = {
            "weighted_contracts": 5,
            "vix_at_assessment": 20.0,
        }

        trajectory = TradeTrajectory.from_pipeline_result(
            underlying="SPY",
            spread_type="bull_put",
            short_strike=450.0,
            long_strike=445.0,
            expiration="2024-02-15",
            entry_credit=0.50,
            contracts=5,
            analyst_messages=[analyst_msg],
            debate_messages=[debate_msg],
            synthesis_message=synthesis_msg,
            decision_output={"decision": "enter"},
            three_perspective=three_persp,
            market_regime="trending_up",
            iv_rank=65.0,
            vix_at_entry=20.0,
            trade_id="trade-123",
        )

        assert trajectory.underlying == "SPY"
        assert trajectory.spread_type == "bull_put"
        assert trajectory.contracts == 5
        assert trajectory.market_regime == "trending_up"
        assert trajectory.trade_id == "trade-123"
        assert len(trajectory.analyst_outputs) == 1
        assert trajectory.analyst_outputs[0]["agent_id"] == "iv_analyst"
        assert trajectory.three_perspective_result is not None
