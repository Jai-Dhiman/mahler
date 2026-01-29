"""Simulated trade execution for backtesting.

Implements realistic fill simulation using ORATS slippage methodology:
- Fill prices based on bid-ask spread and leg count
- Commission modeling for entry and exit
- Support for profit target, stop loss, and time-based exits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from core.backtesting.config import BacktestConfig, MarketRegime


@dataclass
class OptionQuote:
    """Option quote data for simulation."""

    symbol: str
    underlying: str
    strike: float
    option_type: Literal["call", "put"]
    expiration: date
    bid: float
    ask: float
    underlying_price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class SimulatedFill:
    """Result of a simulated order fill."""

    fill_price: float
    slippage_cost: float  # Cost due to slippage (always positive)
    commission: float
    total_cost: float  # slippage_cost + commission
    timestamp: datetime


@dataclass
class SpreadQuote:
    """Quote for a credit spread (2 legs)."""

    short_leg: OptionQuote
    long_leg: OptionQuote
    legs: int = 2

    @property
    def natural_credit(self) -> float:
        """Natural credit (short bid - long ask)."""
        return self.short_leg.bid - self.long_leg.ask

    @property
    def mid_credit(self) -> float:
        """Mid credit ((short_mid - long_mid))."""
        short_mid = (self.short_leg.bid + self.short_leg.ask) / 2
        long_mid = (self.long_leg.bid + self.long_leg.ask) / 2
        return short_mid - long_mid

    @property
    def spread_width(self) -> float:
        """Width of the spread in dollars."""
        return abs(self.short_leg.strike - self.long_leg.strike)


@dataclass
class BacktestTrade:
    """A single trade in backtest simulation."""

    trade_id: str
    underlying: str
    spread_type: Literal["bull_put", "bear_call"]

    # Entry details
    entry_date: date
    entry_credit: float  # Net credit received after slippage
    entry_short_strike: float
    entry_long_strike: float
    expiration: date
    contracts: int

    # Entry costs
    entry_slippage: float
    entry_commission: float

    # Exit details (populated on close)
    exit_date: date | None = None
    exit_debit: float | None = None  # Net debit paid after slippage
    exit_slippage: float = 0.0
    exit_commission: float = 0.0
    exit_reason: Literal[
        "profit_target", "stop_loss", "time_exit", "expiration"
    ] | None = None

    # Market context at entry
    entry_iv_percentile: float = 0.0
    entry_vix: float = 0.0
    entry_regime: str = "normal"
    entry_underlying_price: float = 0.0

    # Calculated fields
    @property
    def gross_pnl(self) -> float | None:
        """Gross P/L before costs."""
        if self.exit_debit is None:
            return None
        return (self.entry_credit - self.exit_debit) * 100 * self.contracts

    @property
    def total_costs(self) -> float:
        """Total costs (slippage + commissions)."""
        return (
            self.entry_slippage
            + self.entry_commission
            + self.exit_slippage
            + self.exit_commission
        )

    @property
    def net_pnl(self) -> float | None:
        """Net P/L after all costs."""
        if self.gross_pnl is None:
            return None
        return self.gross_pnl - self.total_costs

    @property
    def net_pnl_pct(self) -> float | None:
        """Net P/L as percentage of max risk."""
        if self.net_pnl is None:
            return None
        max_risk = (
            self.spread_width - self.entry_credit
        ) * 100 * self.contracts
        if max_risk == 0:
            return 0.0
        return self.net_pnl / max_risk

    @property
    def spread_width(self) -> float:
        """Width of the spread."""
        return abs(self.entry_short_strike - self.entry_long_strike)

    @property
    def days_in_trade(self) -> int | None:
        """Number of days position was held."""
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days

    @property
    def is_winner(self) -> bool | None:
        """Whether trade was profitable."""
        if self.net_pnl is None:
            return None
        return self.net_pnl > 0


class BacktestExecutor:
    """Executes simulated trades with realistic fills.

    Uses ORATS slippage methodology:
    - Buy: Bid + (Ask - Bid) * slippage_pct
    - Sell: Ask - (Ask - Bid) * slippage_pct

    For credit spreads:
    - Entry: Sell short leg, Buy long leg (receive credit)
    - Exit: Buy short leg, Sell long leg (pay debit)
    """

    def __init__(self, config: BacktestConfig):
        """Initialize executor with backtest configuration.

        Args:
            config: BacktestConfig with slippage and commission settings
        """
        self.config = config

    def calculate_fill_price(
        self,
        bid: float,
        ask: float,
        side: Literal["buy", "sell"],
        legs: int = 2,
    ) -> tuple[float, float]:
        """Calculate fill price with ORATS slippage.

        Args:
            bid: Bid price
            ask: Ask price
            side: "buy" or "sell"
            legs: Number of legs in trade (affects slippage %)

        Returns:
            Tuple of (fill_price, slippage_cost)
        """
        slippage_pct = self.config.get_slippage(legs)
        spread = ask - bid

        if side == "buy":
            # Buy: start at bid, move toward ask
            fill_price = bid + (spread * slippage_pct)
            # Slippage cost is how much worse than mid we got
            mid = (bid + ask) / 2
            slippage_cost = fill_price - mid
        else:
            # Sell: start at ask, move toward bid
            fill_price = ask - (spread * slippage_pct)
            mid = (bid + ask) / 2
            slippage_cost = mid - fill_price

        return fill_price, max(0, slippage_cost)

    def simulate_spread_entry(
        self,
        spread: SpreadQuote,
        contracts: int,
    ) -> SimulatedFill:
        """Simulate entry fill for a credit spread.

        For credit spreads:
        - Sell short leg (receive premium)
        - Buy long leg (pay premium)
        - Net = credit received

        Args:
            spread: SpreadQuote with leg quotes
            contracts: Number of contracts

        Returns:
            SimulatedFill with net credit after slippage
        """
        # Sell short leg
        short_fill, short_slippage = self.calculate_fill_price(
            bid=spread.short_leg.bid,
            ask=spread.short_leg.ask,
            side="sell",
            legs=spread.legs,
        )

        # Buy long leg
        long_fill, long_slippage = self.calculate_fill_price(
            bid=spread.long_leg.bid,
            ask=spread.long_leg.ask,
            side="buy",
            legs=spread.legs,
        )

        # Net credit = what we receive - what we pay
        net_credit = short_fill - long_fill

        # Total slippage cost (per contract, in dollars)
        total_slippage = (short_slippage + long_slippage) * 100 * contracts

        # Commission: $1 per contract per leg on entry
        commission = self.config.commission_per_contract * contracts * spread.legs

        return SimulatedFill(
            fill_price=net_credit,
            slippage_cost=total_slippage,
            commission=commission,
            total_cost=total_slippage + commission,
            timestamp=datetime.now(),
        )

    def simulate_spread_exit(
        self,
        spread: SpreadQuote,
        contracts: int,
        expired_otm: bool = False,
    ) -> SimulatedFill:
        """Simulate exit fill for a credit spread.

        For credit spreads:
        - Buy back short leg (pay to close)
        - Sell long leg (receive to close)
        - Net = debit paid

        Args:
            spread: SpreadQuote with current leg quotes
            contracts: Number of contracts
            expired_otm: If True, no commission charged (expired worthless)

        Returns:
            SimulatedFill with net debit after slippage
        """
        # Buy back short leg
        short_fill, short_slippage = self.calculate_fill_price(
            bid=spread.short_leg.bid,
            ask=spread.short_leg.ask,
            side="buy",
            legs=spread.legs,
        )

        # Sell long leg
        long_fill, long_slippage = self.calculate_fill_price(
            bid=spread.long_leg.bid,
            ask=spread.long_leg.ask,
            side="sell",
            legs=spread.legs,
        )

        # Net debit = what we pay - what we receive
        net_debit = short_fill - long_fill

        # Total slippage cost
        total_slippage = (short_slippage + long_slippage) * 100 * contracts

        # Commission: $0 if expired OTM, else $1 per contract per leg
        if expired_otm:
            commission = 0.0
        else:
            commission = self.config.commission_per_contract * contracts * spread.legs

        return SimulatedFill(
            fill_price=net_debit,
            slippage_cost=total_slippage,
            commission=commission,
            total_cost=total_slippage + commission,
            timestamp=datetime.now(),
        )

    def check_exit_conditions(
        self,
        trade: BacktestTrade,
        current_debit: float,
        current_date: date,
    ) -> Literal["profit_target", "stop_loss", "time_exit", "expiration"] | None:
        """Check if any exit condition is met.

        Args:
            trade: Current BacktestTrade
            current_debit: Current cost to close the spread
            current_date: Current simulation date

        Returns:
            Exit reason if condition met, None otherwise
        """
        # Calculate current P/L percentage relative to credit received
        pnl_pct = (trade.entry_credit - current_debit) / trade.entry_credit

        # Check profit target (50% of credit by default)
        if pnl_pct >= self.config.profit_target_pct:
            return "profit_target"

        # Check stop loss (125% of credit = losing 25% more than received)
        if pnl_pct <= -self.config.stop_loss_pct:
            return "stop_loss"

        # Calculate DTE
        dte = (trade.expiration - current_date).days

        # Check time-based exit
        if dte <= self.config.dte_exit:
            return "time_exit"

        # Check expiration
        if current_date >= trade.expiration:
            return "expiration"

        return None
