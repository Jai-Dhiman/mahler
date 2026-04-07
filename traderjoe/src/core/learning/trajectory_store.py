"""Trajectory store for trade context and outcome tracking.

Stores complete trade trajectories including:
- Trade details (underlying, spread, strikes, expiration)
- Market context (regime, IV, VIX)
- Agent outputs (analyst, debate, synthesis, decision)
- Outcomes (P/L, exit reason, days held)
- Labels (reward score for learning)

This data enables:
1. Retrospective analysis of decisions
2. Auto-labeling for reinforcement learning
3. Pattern recognition across similar trades
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.agents.base import AgentMessage
    from core.risk.three_perspective import ThreePerspectiveResult


@dataclass
class TradeTrajectory:
    """Complete trajectory for a trade decision.

    V2: Added raw_cot_traces and hit_score for TradingGroup paper requirements.
    """

    # Trade identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Trade details
    underlying: str = ""
    spread_type: str = ""
    short_strike: float = 0.0
    long_strike: float = 0.0
    expiration: str = ""
    entry_credit: float = 0.0
    contracts: int = 0

    # Market context at entry
    market_regime: str | None = None
    iv_rank: float | None = None
    vix_at_entry: float | None = None

    # Agent outputs (serialized JSON)
    analyst_outputs: list[dict] = field(default_factory=list)
    debate_transcript: list[dict] = field(default_factory=list)
    synthesis_output: dict | None = None
    decision_output: dict | None = None

    # Three-perspective assessment (V2)
    three_perspective_result: dict | None = None

    # V2: Chain-of-Thought traces from extended thinking
    # Maps agent_id -> raw thinking content
    raw_cot_traces: dict | None = None

    # Outcome (filled after trade closes)
    actual_pnl: float | None = None
    actual_pnl_percent: float | None = None
    exit_reason: str | None = None
    days_held: int | None = None

    # Labels (filled by data synthesis)
    reward_label: str | None = None
    reward_score: float | None = None

    # V2: Hit-score for forecasting accuracy
    hit_score: float | None = None

    # Linked trade ID (if trade was executed)
    trade_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "underlying": self.underlying,
            "spread_type": self.spread_type,
            "short_strike": self.short_strike,
            "long_strike": self.long_strike,
            "expiration": self.expiration,
            "entry_credit": self.entry_credit,
            "contracts": self.contracts,
            "market_regime": self.market_regime,
            "iv_rank": self.iv_rank,
            "vix_at_entry": self.vix_at_entry,
            "analyst_outputs": json.dumps(self.analyst_outputs),
            "debate_transcript": json.dumps(self.debate_transcript),
            "synthesis_output": json.dumps(self.synthesis_output) if self.synthesis_output else None,
            "decision_output": json.dumps(self.decision_output) if self.decision_output else None,
            "three_perspective_result": json.dumps(self.three_perspective_result) if self.three_perspective_result else None,
            "raw_cot_traces": json.dumps(self.raw_cot_traces) if self.raw_cot_traces else None,
            "actual_pnl": self.actual_pnl,
            "actual_pnl_percent": self.actual_pnl_percent,
            "exit_reason": self.exit_reason,
            "days_held": self.days_held,
            "reward_label": self.reward_label,
            "reward_score": self.reward_score,
            "hit_score": self.hit_score,
            "trade_id": self.trade_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeTrajectory":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.now().isoformat()),
            underlying=data.get("underlying", ""),
            spread_type=data.get("spread_type", ""),
            short_strike=data.get("short_strike", 0.0),
            long_strike=data.get("long_strike", 0.0),
            expiration=data.get("expiration", ""),
            entry_credit=data.get("entry_credit", 0.0),
            contracts=data.get("contracts", 0),
            market_regime=data.get("market_regime"),
            iv_rank=data.get("iv_rank"),
            vix_at_entry=data.get("vix_at_entry"),
            analyst_outputs=json.loads(data.get("analyst_outputs") or "[]"),
            debate_transcript=json.loads(data.get("debate_transcript") or "[]"),
            synthesis_output=json.loads(data.get("synthesis_output")) if data.get("synthesis_output") else None,
            decision_output=json.loads(data.get("decision_output")) if data.get("decision_output") else None,
            three_perspective_result=json.loads(data.get("three_perspective_result")) if data.get("three_perspective_result") else None,
            raw_cot_traces=json.loads(data.get("raw_cot_traces")) if data.get("raw_cot_traces") else None,
            actual_pnl=data.get("actual_pnl"),
            actual_pnl_percent=data.get("actual_pnl_percent"),
            exit_reason=data.get("exit_reason"),
            days_held=data.get("days_held"),
            reward_label=data.get("reward_label"),
            reward_score=data.get("reward_score"),
            hit_score=data.get("hit_score"),
            trade_id=data.get("trade_id"),
        )

    @classmethod
    def from_pipeline_result(
        cls,
        underlying: str,
        spread_type: str,
        short_strike: float,
        long_strike: float,
        expiration: str,
        entry_credit: float,
        contracts: int,
        analyst_messages: list[AgentMessage],
        debate_messages: list[AgentMessage],
        synthesis_message: AgentMessage | None,
        decision_output: dict | None,
        three_perspective: ThreePerspectiveResult | None,
        market_regime: str | None = None,
        iv_rank: float | None = None,
        vix_at_entry: float | None = None,
        trade_id: str | None = None,
    ) -> "TradeTrajectory":
        """Create trajectory from pipeline execution results."""
        # Serialize agent messages
        analyst_outputs = [
            {
                "agent_id": msg.agent_id,
                "content": msg.content,
                "structured_data": msg.structured_data,
                "confidence": msg.confidence,
            }
            for msg in analyst_messages
        ]

        debate_transcript = [
            {
                "agent_id": msg.agent_id,
                "message_type": msg.message_type.value,
                "content": msg.content,
                "structured_data": msg.structured_data,
                "confidence": msg.confidence,
            }
            for msg in debate_messages
        ]

        synthesis_output = None
        if synthesis_message:
            synthesis_output = {
                "agent_id": synthesis_message.agent_id,
                "content": synthesis_message.content,
                "structured_data": synthesis_message.structured_data,
                "confidence": synthesis_message.confidence,
            }

        three_persp_dict = three_perspective.to_dict() if three_perspective else None

        return cls(
            underlying=underlying,
            spread_type=spread_type,
            short_strike=short_strike,
            long_strike=long_strike,
            expiration=expiration,
            entry_credit=entry_credit,
            contracts=contracts,
            market_regime=market_regime,
            iv_rank=iv_rank,
            vix_at_entry=vix_at_entry,
            analyst_outputs=analyst_outputs,
            debate_transcript=debate_transcript,
            synthesis_output=synthesis_output,
            decision_output=decision_output,
            three_perspective_result=three_persp_dict,
            trade_id=trade_id,
        )

    @property
    def has_outcome(self) -> bool:
        """Check if outcome data has been recorded."""
        return self.actual_pnl is not None

    @property
    def has_label(self) -> bool:
        """Check if reward label has been assigned."""
        return self.reward_label is not None


class TrajectoryStore:
    """Store for trade trajectories with D1 backend.

    Provides methods for:
    - Storing new trajectories
    - Updating outcomes after trade closes
    - Retrieving unlabeled trajectories for synthesis
    - Updating reward labels
    """

    def __init__(self, d1_binding):
        """Initialize trajectory store.

        Args:
            d1_binding: Cloudflare D1 database binding
        """
        self.db = d1_binding

    async def store_trajectory(self, trajectory: TradeTrajectory) -> str:
        """Store a new trajectory.

        Args:
            trajectory: TradeTrajectory to store

        Returns:
            Trajectory ID
        """
        data = trajectory.to_dict()

        await self.db.prepare(
            """
            INSERT INTO trade_trajectories (
                id, created_at, underlying, spread_type, short_strike, long_strike,
                expiration, entry_credit, contracts, market_regime, iv_rank, vix_at_entry,
                analyst_outputs, debate_transcript, synthesis_output, decision_output,
                three_perspective_result, trade_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        ).bind(
            data["id"],
            data["created_at"],
            data["underlying"],
            data["spread_type"],
            data["short_strike"],
            data["long_strike"],
            data["expiration"],
            data["entry_credit"],
            data["contracts"],
            data["market_regime"],
            data["iv_rank"],
            data["vix_at_entry"],
            data["analyst_outputs"],
            data["debate_transcript"],
            data["synthesis_output"],
            data["decision_output"],
            data["three_perspective_result"],
            data["trade_id"],
        ).run()

        return trajectory.id

    async def update_outcome(
        self,
        trajectory_id: str,
        actual_pnl: float,
        actual_pnl_percent: float,
        exit_reason: str,
        days_held: int,
    ) -> None:
        """Update trajectory with outcome data.

        Args:
            trajectory_id: ID of trajectory to update
            actual_pnl: Realized P/L in dollars
            actual_pnl_percent: Realized P/L as percentage
            exit_reason: Reason for exit (profit_target, stop_loss, time_exit, etc.)
            days_held: Number of days position was held
        """
        await self.db.prepare(
            """
            UPDATE trade_trajectories
            SET actual_pnl = ?, actual_pnl_percent = ?, exit_reason = ?, days_held = ?
            WHERE id = ?
            """
        ).bind(
            actual_pnl,
            actual_pnl_percent,
            exit_reason,
            days_held,
            trajectory_id,
        ).run()

    async def get_unlabeled_trajectories(
        self,
        with_outcomes: bool = True,
        limit: int = 100,
    ) -> list[TradeTrajectory]:
        """Get trajectories that haven't been labeled yet.

        Args:
            with_outcomes: If True, only return trajectories that have outcome data
            limit: Maximum number of trajectories to return

        Returns:
            List of unlabeled TradeTrajectory objects
        """
        if with_outcomes:
            result = await self.db.prepare(
                """
                SELECT * FROM trade_trajectories
                WHERE reward_label IS NULL AND actual_pnl IS NOT NULL
                ORDER BY created_at ASC
                LIMIT ?
                """
            ).bind(limit).all()
        else:
            result = await self.db.prepare(
                """
                SELECT * FROM trade_trajectories
                WHERE reward_label IS NULL
                ORDER BY created_at ASC
                LIMIT ?
                """
            ).bind(limit).all()

        return [TradeTrajectory.from_dict(row) for row in result.get("results", [])]

    async def update_labels(
        self,
        trajectory_id: str,
        reward_label: str,
        reward_score: float,
    ) -> None:
        """Update trajectory with reward labels.

        Args:
            trajectory_id: ID of trajectory to update
            reward_label: Label category (strong_positive, positive, negative, strong_negative)
            reward_score: Numerical reward score
        """
        await self.db.prepare(
            """
            UPDATE trade_trajectories
            SET reward_label = ?, reward_score = ?
            WHERE id = ?
            """
        ).bind(
            reward_label,
            reward_score,
            trajectory_id,
        ).run()

    async def get_trajectory(self, trajectory_id: str) -> TradeTrajectory | None:
        """Get a trajectory by ID.

        Args:
            trajectory_id: ID of trajectory to retrieve

        Returns:
            TradeTrajectory if found, None otherwise
        """
        result = await self.db.prepare(
            "SELECT * FROM trade_trajectories WHERE id = ?"
        ).bind(trajectory_id).first()

        if result:
            return TradeTrajectory.from_dict(result)
        return None

    async def get_trajectory_by_trade_id(self, trade_id: str) -> TradeTrajectory | None:
        """Get trajectory by linked trade ID.

        Args:
            trade_id: Trade ID to look up

        Returns:
            TradeTrajectory if found, None otherwise
        """
        result = await self.db.prepare(
            "SELECT * FROM trade_trajectories WHERE trade_id = ?"
        ).bind(trade_id).first()

        if result:
            return TradeTrajectory.from_dict(result)
        return None

    async def get_trajectories_by_underlying(
        self,
        underlying: str,
        limit: int = 50,
    ) -> list[TradeTrajectory]:
        """Get trajectories for a specific underlying.

        Args:
            underlying: Underlying symbol
            limit: Maximum number to return

        Returns:
            List of TradeTrajectory objects
        """
        result = await self.db.prepare(
            """
            SELECT * FROM trade_trajectories
            WHERE underlying = ?
            ORDER BY created_at DESC
            LIMIT ?
            """
        ).bind(underlying, limit).all()

        return [TradeTrajectory.from_dict(row) for row in result.get("results", [])]

    async def get_labeled_trajectories(
        self,
        label: str | None = None,
        limit: int = 100,
    ) -> list[TradeTrajectory]:
        """Get labeled trajectories for analysis.

        Args:
            label: Optional filter by reward label
            limit: Maximum number to return

        Returns:
            List of labeled TradeTrajectory objects
        """
        if label:
            result = await self.db.prepare(
                """
                SELECT * FROM trade_trajectories
                WHERE reward_label = ?
                ORDER BY created_at DESC
                LIMIT ?
                """
            ).bind(label, limit).all()
        else:
            result = await self.db.prepare(
                """
                SELECT * FROM trade_trajectories
                WHERE reward_label IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
                """
            ).bind(limit).all()

        return [TradeTrajectory.from_dict(row) for row in result.get("results", [])]
