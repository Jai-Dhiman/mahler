"""Rolling metrics calculation for strategy monitoring.

Tracks key performance metrics over configurable lookback windows
and compares against backtest expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.db.d1 import D1Client


@dataclass
class RollingMetrics:
    """Rolling performance metrics over a lookback window."""

    # Window info
    window_trades: int
    window_start_date: datetime | None
    window_end_date: datetime | None

    # Core metrics
    win_rate: float  # 0-100 %
    profit_factor: float
    total_pnl: float
    avg_pnl_per_trade: float

    # Drawdown
    current_drawdown: float  # 0-100 %
    max_drawdown_in_window: float  # 0-100 %

    # Recent trade results (for context)
    last_n_results: list[str]  # "W" or "L" for last N trades

    # Slippage metrics (if available)
    avg_slippage_pct: float | None
    worst_slippage_pct: float | None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "window_trades": self.window_trades,
            "window_start_date": self.window_start_date.isoformat() if self.window_start_date else None,
            "window_end_date": self.window_end_date.isoformat() if self.window_end_date else None,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "current_drawdown": self.current_drawdown,
            "max_drawdown_in_window": self.max_drawdown_in_window,
            "last_n_results": self.last_n_results,
            "avg_slippage_pct": self.avg_slippage_pct,
            "worst_slippage_pct": self.worst_slippage_pct,
        }


@dataclass
class SlippageMetrics:
    """Slippage tracking for a single trade."""

    trade_id: str
    underlying: str
    expected_credit: float  # Mid price at entry signal
    actual_credit: float  # Filled price
    slippage_pct: float  # (expected - actual) / expected * 100
    trade_date: datetime


class MetricsTracker:
    """Calculates and tracks rolling performance metrics.

    Uses D1 database to retrieve historical trade data and calculate
    rolling metrics over configurable windows.
    """

    def __init__(self, d1_client: D1Client):
        self.d1 = d1_client

    async def get_rolling_metrics(
        self,
        lookback_trades: int = 20,
        underlying: str | None = None,
    ) -> RollingMetrics:
        """Calculate rolling metrics over the last N trades.

        Args:
            lookback_trades: Number of recent trades to include
            underlying: Optional filter by underlying (SPY, QQQ, IWM)

        Returns:
            RollingMetrics with calculated values
        """
        # Build query for closed trades
        query = """
            SELECT
                id, underlying, entry_date, exit_date, entry_credit, exit_debit,
                realized_pnl, contracts, spread_type, status
            FROM trades
            WHERE status = 'closed'
        """
        params: list = []

        if underlying:
            query += " AND underlying = ?"
            params.append(underlying)

        query += " ORDER BY exit_date DESC LIMIT ?"
        params.append(lookback_trades)

        trades = await self.d1.execute(query, params)

        if not trades:
            return RollingMetrics(
                window_trades=0,
                window_start_date=None,
                window_end_date=None,
                win_rate=0.0,
                profit_factor=0.0,
                total_pnl=0.0,
                avg_pnl_per_trade=0.0,
                current_drawdown=0.0,
                max_drawdown_in_window=0.0,
                last_n_results=[],
                avg_slippage_pct=None,
                worst_slippage_pct=None,
            )

        # Calculate metrics
        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0
        total_pnl = 0.0
        results: list[str] = []

        for trade in trades:
            pnl = trade.get("realized_pnl", 0) or 0
            total_pnl += pnl

            if pnl >= 0:
                wins += 1
                total_profit += pnl
                results.append("W")
            else:
                losses += 1
                total_loss += abs(pnl)
                results.append("L")

        # Win rate
        win_rate = (wins / len(trades) * 100) if trades else 0.0

        # Profit factor
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf")
        if profit_factor == float("inf"):
            profit_factor = 999.99  # Cap for display

        # Average P&L
        avg_pnl = total_pnl / len(trades) if trades else 0.0

        # Parse dates
        window_start = None
        window_end = None
        if trades:
            # Trades are ordered DESC, so first is most recent
            if trades[0].get("exit_date"):
                window_end = datetime.fromisoformat(trades[0]["exit_date"])
            if trades[-1].get("exit_date"):
                window_start = datetime.fromisoformat(trades[-1]["exit_date"])

        # Get drawdown from equity curve
        drawdown_metrics = await self._calculate_drawdown()

        # Get slippage metrics if available
        slippage_metrics = await self._calculate_slippage_metrics(lookback_trades, underlying)

        return RollingMetrics(
            window_trades=len(trades),
            window_start_date=window_start,
            window_end_date=window_end,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            current_drawdown=drawdown_metrics["current"],
            max_drawdown_in_window=drawdown_metrics["max"],
            last_n_results=results[:10],  # Last 10 for display
            avg_slippage_pct=slippage_metrics.get("avg"),
            worst_slippage_pct=slippage_metrics.get("worst"),
        )

    async def _calculate_drawdown(self) -> dict[str, float]:
        """Calculate current and max drawdown from equity curve."""
        # Get daily performance records
        query = """
            SELECT date, ending_balance
            FROM daily_performance
            ORDER BY date ASC
        """
        records = await self.d1.execute(query, [])

        if not records:
            return {"current": 0.0, "max": 0.0}

        peak = 0.0
        max_drawdown = 0.0
        current_drawdown = 0.0

        for record in records:
            balance = record.get("ending_balance", 0) or 0
            if balance > peak:
                peak = balance

            if peak > 0:
                drawdown = (peak - balance) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
                current_drawdown = drawdown  # Last value is current

        return {"current": current_drawdown, "max": max_drawdown}

    async def _calculate_slippage_metrics(
        self,
        lookback: int,
        underlying: str | None,
    ) -> dict[str, float | None]:
        """Calculate slippage metrics from recent trades.

        Slippage is calculated as:
        slippage_pct = (expected_credit - actual_credit) / expected_credit * 100

        Where:
        - expected_credit is the mid-price at signal time
        - actual_credit is the filled price
        """
        # Query trades with expected vs actual credit
        query = """
            SELECT
                id, underlying, entry_credit as actual_credit,
                expected_credit, entry_date
            FROM trades
            WHERE status = 'closed'
            AND expected_credit IS NOT NULL
            AND expected_credit > 0
        """
        params: list = []

        if underlying:
            query += " AND underlying = ?"
            params.append(underlying)

        query += " ORDER BY exit_date DESC LIMIT ?"
        params.append(lookback)

        trades = await self.d1.execute(query, params)

        if not trades:
            return {"avg": None, "worst": None}

        slippages = []
        for trade in trades:
            expected = trade.get("expected_credit", 0) or 0
            actual = trade.get("actual_credit", 0) or 0
            if expected > 0:
                # Calculate slippage as percentage of the bid-ask spread captured
                # 0% = filled at bid (worst), 100% = filled at ask (best for seller)
                # For credit spreads, we want to maximize credit received
                slippage = (1 - (actual / expected)) * 100
                slippages.append(slippage)

        if not slippages:
            return {"avg": None, "worst": None}

        return {
            "avg": sum(slippages) / len(slippages),
            "worst": max(slippages),  # Higher % = worse fill
        }

    async def get_iv_history(
        self,
        underlying: str,
        days: int = 5,
    ) -> list[dict]:
        """Get IV percentile history for consecutive day tracking.

        Args:
            underlying: Ticker symbol
            days: Number of days to look back

        Returns:
            List of {date, iv_percentile} dicts
        """
        query = """
            SELECT date, iv_percentile
            FROM market_snapshots
            WHERE underlying = ?
            ORDER BY date DESC
            LIMIT ?
        """
        records = await self.d1.execute(query, [underlying, days])

        return [
            {"date": r.get("date"), "iv_percentile": r.get("iv_percentile")}
            for r in records
        ]

    async def record_slippage(
        self,
        trade_id: str,
        underlying: str,
        expected_credit: float,
        actual_credit: float,
    ) -> SlippageMetrics:
        """Record slippage for a trade.

        Call this after a trade is filled to track fill quality.

        Args:
            trade_id: The trade ID
            underlying: Ticker symbol
            expected_credit: Mid-price credit at signal time
            actual_credit: Actual filled credit

        Returns:
            SlippageMetrics with calculated values
        """
        slippage_pct = 0.0
        if expected_credit > 0:
            slippage_pct = (1 - (actual_credit / expected_credit)) * 100

        # Update the trade record with expected_credit for future analysis
        await self.d1.execute(
            """
            UPDATE trades
            SET expected_credit = ?
            WHERE id = ?
            """,
            [expected_credit, trade_id],
        )

        return SlippageMetrics(
            trade_id=trade_id,
            underlying=underlying,
            expected_credit=expected_credit,
            actual_credit=actual_credit,
            slippage_pct=slippage_pct,
            trade_date=datetime.now(),
        )
