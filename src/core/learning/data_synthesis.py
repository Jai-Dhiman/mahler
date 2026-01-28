"""Data synthesis for auto-labeling trade trajectories.

Computes reward scores and labels for trajectories based on:
- Actual P/L performance
- Benchmark comparison (risk-free rate)
- Transaction costs
- Time held

This enables reinforcement learning style improvement by labeling
past decisions as good/bad based on outcomes.

Reward Formula:
    reward = pnl_percent - benchmark_return - transaction_costs

Labels:
- STRONG_POSITIVE: > 2% adjusted return
- POSITIVE: > 0% adjusted return
- NEGATIVE: > -2% adjusted return
- STRONG_NEGATIVE: <= -2% adjusted return
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.learning.trajectory_store import TrajectoryStore, TradeTrajectory


class RewardLabel(str, Enum):
    """Reward label categories for trajectory classification."""

    STRONG_POSITIVE = "strong_positive"  # > 2% adjusted return
    POSITIVE = "positive"  # > 0% adjusted return
    NEGATIVE = "negative"  # > -2% adjusted return
    STRONG_NEGATIVE = "strong_negative"  # <= -2% adjusted return


@dataclass
class LabelingConfig:
    """Configuration for reward calculation and labeling."""

    # Benchmark: Weekly risk-free rate (5% annual / 52 weeks)
    benchmark_weekly_rate: float = 0.001  # ~0.1% per week

    # Transaction costs per contract (round-trip: open + close)
    transaction_cost_per_contract: float = 1.50

    # Label thresholds (adjusted return percentages)
    strong_positive_threshold: float = 0.02  # > 2%
    positive_threshold: float = 0.0  # > 0%
    negative_threshold: float = -0.02  # > -2%
    # Below negative_threshold = strong_negative

    # Annualized benchmark rate (for reference)
    annualized_benchmark: float = 0.05  # 5%

    def get_benchmark_for_days(self, days: int) -> float:
        """Get benchmark return for a specific holding period.

        Args:
            days: Number of days held

        Returns:
            Benchmark return as decimal (e.g., 0.001 for 0.1%)
        """
        if days <= 0:
            return 0.0
        # Convert weekly rate to daily, then scale by days
        daily_rate = self.benchmark_weekly_rate / 7
        return daily_rate * days


@dataclass
class RewardResult:
    """Result of reward calculation for a trajectory."""

    reward_score: float
    reward_label: RewardLabel
    raw_pnl_percent: float
    benchmark_return: float
    transaction_cost_percent: float
    adjusted_return: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "reward_score": self.reward_score,
            "reward_label": self.reward_label.value,
            "raw_pnl_percent": self.raw_pnl_percent,
            "benchmark_return": self.benchmark_return,
            "transaction_cost_percent": self.transaction_cost_percent,
            "adjusted_return": self.adjusted_return,
        }


class DataSynthesizer:
    """Synthesizer for auto-labeling trajectories.

    Computes rewards based on:
    1. Actual P/L percentage
    2. Opportunity cost (benchmark return)
    3. Transaction costs

    The reward formula:
        reward = pnl_pct - benchmark - costs

    This penalizes trades that don't beat the risk-free rate and
    accounts for the friction of transaction costs.
    """

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        config: LabelingConfig | None = None,
    ):
        """Initialize the data synthesizer.

        Args:
            trajectory_store: TrajectoryStore instance for data access
            config: Optional labeling configuration
        """
        self.store = trajectory_store
        self.config = config or LabelingConfig()

    def compute_reward(
        self,
        pnl_percent: float,
        entry_credit: float,
        contracts: int,
        days_held: int,
    ) -> RewardResult:
        """Compute reward score and label for a trade outcome.

        Args:
            pnl_percent: Realized P/L as percentage of entry credit
            entry_credit: Entry credit per spread
            contracts: Number of contracts traded
            days_held: Number of days position was held

        Returns:
            RewardResult with score and label
        """
        # Calculate benchmark return for holding period
        benchmark_return = self.config.get_benchmark_for_days(days_held)

        # Calculate transaction cost as percentage of position value
        total_credit = entry_credit * contracts * 100  # Convert to dollars
        total_cost = self.config.transaction_cost_per_contract * contracts * 2  # Round-trip
        transaction_cost_pct = total_cost / total_credit if total_credit > 0 else 0

        # Calculate adjusted return
        # reward = actual_return - opportunity_cost - friction
        adjusted_return = pnl_percent - benchmark_return - transaction_cost_pct

        # Assign label based on thresholds
        if adjusted_return > self.config.strong_positive_threshold:
            label = RewardLabel.STRONG_POSITIVE
        elif adjusted_return > self.config.positive_threshold:
            label = RewardLabel.POSITIVE
        elif adjusted_return > self.config.negative_threshold:
            label = RewardLabel.NEGATIVE
        else:
            label = RewardLabel.STRONG_NEGATIVE

        # Reward score is the adjusted return (can be used for ranking)
        reward_score = adjusted_return

        return RewardResult(
            reward_score=reward_score,
            reward_label=label,
            raw_pnl_percent=pnl_percent,
            benchmark_return=benchmark_return,
            transaction_cost_percent=transaction_cost_pct,
            adjusted_return=adjusted_return,
        )

    def label_trajectory(self, trajectory: TradeTrajectory) -> RewardResult | None:
        """Compute reward for a single trajectory.

        Args:
            trajectory: TradeTrajectory with outcome data

        Returns:
            RewardResult if trajectory has outcome data, None otherwise
        """
        if not trajectory.has_outcome:
            return None

        if trajectory.actual_pnl_percent is None:
            return None

        return self.compute_reward(
            pnl_percent=trajectory.actual_pnl_percent,
            entry_credit=trajectory.entry_credit,
            contracts=trajectory.contracts,
            days_held=trajectory.days_held or 0,
        )

    async def label_unlabeled_trajectories(self, limit: int = 100) -> int:
        """Label all unlabeled trajectories that have outcomes.

        Args:
            limit: Maximum number of trajectories to process

        Returns:
            Number of trajectories labeled
        """
        # Get unlabeled trajectories with outcomes
        trajectories = await self.store.get_unlabeled_trajectories(
            with_outcomes=True,
            limit=limit,
        )

        labeled_count = 0
        for trajectory in trajectories:
            result = self.label_trajectory(trajectory)
            if result:
                await self.store.update_labels(
                    trajectory_id=trajectory.id,
                    reward_label=result.reward_label.value,
                    reward_score=result.reward_score,
                )
                labeled_count += 1

        return labeled_count

    async def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of labels across all labeled trajectories.

        Returns:
            Dictionary mapping label names to counts
        """
        distribution = {}
        for label in RewardLabel:
            trajectories = await self.store.get_labeled_trajectories(
                label=label.value,
                limit=1000,
            )
            distribution[label.value] = len(trajectories)
        return distribution

    async def get_average_reward_by_regime(self) -> dict[str, float]:
        """Calculate average reward score by market regime.

        Returns:
            Dictionary mapping regime names to average reward scores
        """
        # Get all labeled trajectories
        trajectories = await self.store.get_labeled_trajectories(limit=1000)

        regime_rewards: dict[str, list[float]] = {}
        for traj in trajectories:
            if traj.market_regime and traj.reward_score is not None:
                if traj.market_regime not in regime_rewards:
                    regime_rewards[traj.market_regime] = []
                regime_rewards[traj.market_regime].append(traj.reward_score)

        return {
            regime: sum(scores) / len(scores) if scores else 0.0
            for regime, scores in regime_rewards.items()
        }
