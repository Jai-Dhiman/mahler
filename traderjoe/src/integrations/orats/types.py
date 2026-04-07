"""ORATS data types for options data and backtesting.

Based on ORATS API Documentation: https://docs.orats.io/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Literal


class BacktestJobStatus(str, Enum):
    """Status of an ORATS backtest job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OptionData:
    """Single option contract data from ORATS.

    Near-EOD snapshot (14 minutes before close per ORATS methodology).
    """

    # Identification
    underlying_symbol: str
    underlying_price: float
    quote_date: date
    expiration_date: date
    strike: float
    option_type: Literal["call", "put"]

    # Quotes
    bid: float
    ask: float
    mid: float

    # Volume/interest
    volume: int
    open_interest: int

    # Greeks and IV
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        if self.mid == 0:
            return 0.0
        return self.spread / self.mid * 100

    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiration_date - self.quote_date).days


@dataclass
class OptionsChain:
    """Complete options chain for a symbol on a specific date."""

    underlying_symbol: str
    quote_date: date
    underlying_price: float
    options: list[OptionData] = field(default_factory=list)

    # IV metrics
    iv_30d_atm: float | None = None
    iv_percentile: float | None = None
    iv_rank: float | None = None

    def get_expirations(self) -> list[date]:
        """Get unique expiration dates."""
        return sorted(set(o.expiration_date for o in self.options))

    def get_strikes(self, expiration: date) -> list[float]:
        """Get strikes for a specific expiration."""
        return sorted(
            set(o.strike for o in self.options if o.expiration_date == expiration)
        )

    def get_calls(self, expiration: date | None = None) -> list[OptionData]:
        """Get all call options, optionally filtered by expiration."""
        calls = [o for o in self.options if o.option_type == "call"]
        if expiration:
            calls = [o for o in calls if o.expiration_date == expiration]
        return sorted(calls, key=lambda o: o.strike)

    def get_puts(self, expiration: date | None = None) -> list[OptionData]:
        """Get all put options, optionally filtered by expiration."""
        puts = [o for o in self.options if o.option_type == "put"]
        if expiration:
            puts = [o for o in puts if o.expiration_date == expiration]
        return sorted(puts, key=lambda o: o.strike)

    def get_option(
        self,
        expiration: date,
        strike: float,
        option_type: Literal["call", "put"],
    ) -> OptionData | None:
        """Get a specific option by expiration, strike, and type."""
        for o in self.options:
            if (
                o.expiration_date == expiration
                and o.strike == strike
                and o.option_type == option_type
            ):
                return o
        return None

    def filter_by_dte(self, min_dte: int, max_dte: int) -> list[OptionData]:
        """Filter options by DTE range."""
        return [o for o in self.options if min_dte <= o.dte <= max_dte]

    def filter_by_delta(
        self,
        min_delta: float,
        max_delta: float,
        option_type: Literal["call", "put"] | None = None,
    ) -> list[OptionData]:
        """Filter options by delta range."""
        filtered = []
        for o in self.options:
            if option_type and o.option_type != option_type:
                continue
            abs_delta = abs(o.delta)
            if min_delta <= abs_delta <= max_delta:
                filtered.append(o)
        return filtered


@dataclass
class BacktestSubmission:
    """Parameters for submitting a backtest to ORATS API."""

    # Strategy definition
    strategy_name: str
    underlying: str
    start_date: date
    end_date: date

    # Entry criteria
    entry_dte_min: int
    entry_dte_max: int
    entry_delta_min: float
    entry_delta_max: float

    # Exit criteria
    profit_target_pct: float
    stop_loss_pct: float
    dte_exit: int

    # Position sizing
    contracts_per_trade: int = 1

    # Strategy type
    strategy_type: Literal["bull_put", "bear_call", "iron_condor"] = "bull_put"

    # Optional filters
    iv_percentile_min: float | None = None
    vix_max: float | None = None


@dataclass
class BacktestStatus:
    """Status of a submitted backtest job."""

    job_id: str
    status: BacktestJobStatus
    submitted_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    progress_pct: float = 0.0


@dataclass
class BacktestTradeResult:
    """Result of a single trade in backtest."""

    trade_id: str
    entry_date: date
    exit_date: date

    # Position details
    underlying: str
    short_strike: float
    long_strike: float
    expiration: date
    contracts: int

    # P/L
    entry_credit: float
    exit_debit: float
    gross_pnl: float
    net_pnl: float  # After slippage/commissions

    # Exit info
    exit_reason: Literal["profit_target", "stop_loss", "time_exit", "expiration"]

    # Market context
    entry_iv_percentile: float
    entry_vix: float


@dataclass
class BacktestResults:
    """Complete results from ORATS backtest API."""

    job_id: str
    strategy_name: str
    underlying: str
    start_date: date
    end_date: date

    # Summary metrics
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float

    # Returns
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Holding period
    avg_days_in_trade: float
    avg_dte_at_exit: float

    # Costs
    total_slippage: float
    total_commission: float

    # Individual trades
    trades: list[BacktestTradeResult] = field(default_factory=list)

    @property
    def win_loss_ratio(self) -> float:
        """Win to loss count ratio."""
        if self.loss_count == 0:
            return float("inf")
        return self.win_count / self.loss_count

    @property
    def total_costs(self) -> float:
        """Total trading costs."""
        return self.total_slippage + self.total_commission
