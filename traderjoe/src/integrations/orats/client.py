"""ORATS API client interface and implementation.

API Endpoints (from ORATS documentation):
- POST https://api.orats.io/backtest/submit   # Submit backtest
- POST https://api.orats.io/backtest/status   # Check status
- GET  https://api.orats.io/backtest/results  # Get results
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date, datetime
from typing import Protocol

import httpx

from integrations.orats.types import (
    BacktestJobStatus,
    BacktestResults,
    BacktestStatus,
    BacktestSubmission,
    OptionsChain,
)


class ORATSClient(Protocol):
    """Protocol for ORATS API client.

    This interface allows for different implementations:
    - ORATSClientImpl: Real API client
    - MockORATSClient: Mock for testing without API key
    """

    @abstractmethod
    async def submit_backtest(self, config: BacktestSubmission) -> str:
        """Submit a backtest job to ORATS.

        Args:
            config: BacktestSubmission with strategy parameters

        Returns:
            Job ID for tracking the backtest
        """
        ...

    @abstractmethod
    async def get_backtest_status(self, job_id: str) -> BacktestStatus:
        """Check status of a submitted backtest job.

        Args:
            job_id: ID returned from submit_backtest

        Returns:
            BacktestStatus with current status and progress
        """
        ...

    @abstractmethod
    async def get_backtest_results(self, job_id: str) -> BacktestResults:
        """Get results of a completed backtest.

        Args:
            job_id: ID of a completed backtest

        Returns:
            BacktestResults with all trade data and metrics

        Raises:
            ValueError: If backtest is not yet completed
        """
        ...

    @abstractmethod
    async def get_historical_chain(
        self, symbol: str, quote_date: date
    ) -> OptionsChain:
        """Get historical options chain for a symbol on a specific date.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            quote_date: Date to get chain for

        Returns:
            OptionsChain with all options data
        """
        ...


class ORATSClientImpl:
    """Real ORATS API client implementation.

    Note: Requires ORATS API key. Use MockORATSClient for testing.
    """

    BASE_URL = "https://api.orats.io"

    def __init__(self, api_key: str, timeout: float = 30.0):
        """Initialize ORATS client.

        Args:
            api_key: ORATS API key
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def submit_backtest(self, config: BacktestSubmission) -> str:
        """Submit a backtest job to ORATS API."""
        client = await self._get_client()

        payload = {
            "strategy_name": config.strategy_name,
            "underlying": config.underlying,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "entry_dte_min": config.entry_dte_min,
            "entry_dte_max": config.entry_dte_max,
            "entry_delta_min": config.entry_delta_min,
            "entry_delta_max": config.entry_delta_max,
            "profit_target_pct": config.profit_target_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "dte_exit": config.dte_exit,
            "contracts_per_trade": config.contracts_per_trade,
            "strategy_type": config.strategy_type,
        }

        if config.iv_percentile_min is not None:
            payload["iv_percentile_min"] = config.iv_percentile_min
        if config.vix_max is not None:
            payload["vix_max"] = config.vix_max

        response = await client.post("/backtest/submit", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["job_id"]

    async def get_backtest_status(self, job_id: str) -> BacktestStatus:
        """Check status of a submitted backtest job."""
        client = await self._get_client()

        response = await client.post("/backtest/status", json={"job_id": job_id})
        response.raise_for_status()

        data = response.json()

        return BacktestStatus(
            job_id=job_id,
            status=BacktestJobStatus(data["status"]),
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            error_message=data.get("error_message"),
            progress_pct=data.get("progress_pct", 0.0),
        )

    async def get_backtest_results(self, job_id: str) -> BacktestResults:
        """Get results of a completed backtest."""
        client = await self._get_client()

        response = await client.get(f"/backtest/results?job_id={job_id}")
        response.raise_for_status()

        data = response.json()

        # Parse trades
        trades = []
        for t in data.get("trades", []):
            from integrations.orats.types import BacktestTradeResult

            trades.append(
                BacktestTradeResult(
                    trade_id=t["trade_id"],
                    entry_date=date.fromisoformat(t["entry_date"]),
                    exit_date=date.fromisoformat(t["exit_date"]),
                    underlying=t["underlying"],
                    short_strike=t["short_strike"],
                    long_strike=t["long_strike"],
                    expiration=date.fromisoformat(t["expiration"]),
                    contracts=t["contracts"],
                    entry_credit=t["entry_credit"],
                    exit_debit=t["exit_debit"],
                    gross_pnl=t["gross_pnl"],
                    net_pnl=t["net_pnl"],
                    exit_reason=t["exit_reason"],
                    entry_iv_percentile=t["entry_iv_percentile"],
                    entry_vix=t["entry_vix"],
                )
            )

        return BacktestResults(
            job_id=job_id,
            strategy_name=data["strategy_name"],
            underlying=data["underlying"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            total_trades=data["total_trades"],
            win_count=data["win_count"],
            loss_count=data["loss_count"],
            win_rate=data["win_rate"],
            total_pnl=data["total_pnl"],
            avg_pnl=data["avg_pnl"],
            profit_factor=data["profit_factor"],
            sharpe_ratio=data["sharpe_ratio"],
            max_drawdown=data["max_drawdown"],
            max_drawdown_pct=data["max_drawdown_pct"],
            avg_days_in_trade=data["avg_days_in_trade"],
            avg_dte_at_exit=data["avg_dte_at_exit"],
            total_slippage=data["total_slippage"],
            total_commission=data["total_commission"],
            trades=trades,
        )

    async def get_historical_chain(
        self, symbol: str, quote_date: date
    ) -> OptionsChain:
        """Get historical options chain for a symbol on a specific date."""
        client = await self._get_client()

        response = await client.get(
            f"/data/options/chain?symbol={symbol}&date={quote_date.isoformat()}"
        )
        response.raise_for_status()

        data = response.json()

        from integrations.orats.types import OptionData

        options = []
        for o in data.get("options", []):
            options.append(
                OptionData(
                    underlying_symbol=symbol,
                    underlying_price=data["underlying_price"],
                    quote_date=quote_date,
                    expiration_date=date.fromisoformat(o["expiration"]),
                    strike=o["strike"],
                    option_type=o["option_type"],
                    bid=o["bid"],
                    ask=o["ask"],
                    mid=o["mid"],
                    volume=o["volume"],
                    open_interest=o["open_interest"],
                    implied_volatility=o["iv"],
                    delta=o["delta"],
                    gamma=o["gamma"],
                    theta=o["theta"],
                    vega=o["vega"],
                    rho=o.get("rho", 0.0),
                )
            )

        return OptionsChain(
            underlying_symbol=symbol,
            quote_date=quote_date,
            underlying_price=data["underlying_price"],
            options=options,
            iv_30d_atm=data.get("iv_30d_atm"),
            iv_percentile=data.get("iv_percentile"),
            iv_rank=data.get("iv_rank"),
        )
