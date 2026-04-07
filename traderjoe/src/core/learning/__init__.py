"""Learning module for continuous improvement through trajectory tracking.

The learning module provides infrastructure for:
- Storing complete trade trajectories (context + decision + outcome)
- Auto-labeling trajectories with reward scores
- Building datasets for future model improvement

Components:
- TrajectoryStore: Persists trade trajectories to D1
- DataSynthesizer: Auto-labels trajectories with reward calculations

Usage:
    from core.learning import (
        TrajectoryStore,
        TradeTrajectory,
        DataSynthesizer,
        RewardLabel,
    )

    # Store a trajectory after trade execution
    trajectory_store = TrajectoryStore(db)
    trajectory = TradeTrajectory(
        underlying="SPY",
        spread_type="bull_put",
        ...
    )
    trajectory_id = await trajectory_store.store_trajectory(trajectory)

    # Update outcome after trade closes
    await trajectory_store.update_outcome(
        trajectory_id,
        actual_pnl=150.0,
        actual_pnl_percent=0.30,
        exit_reason="profit_target",
        days_held=14,
    )

    # Auto-label trajectories with rewards
    synthesizer = DataSynthesizer(trajectory_store)
    labeled_count = await synthesizer.label_unlabeled_trajectories()
"""

from core.learning.trajectory_store import (
    TradeTrajectory,
    TrajectoryStore,
)
from core.learning.data_synthesis import (
    DataSynthesizer,
    LabelingConfig,
    RewardLabel,
)

__all__ = [
    # Trajectory Store
    "TradeTrajectory",
    "TrajectoryStore",
    # Data Synthesis
    "DataSynthesizer",
    "LabelingConfig",
    "RewardLabel",
]
