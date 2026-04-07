"""Trade cost modeling per ORATS methodology.

Commission Structure (Alpaca-based):
- Entry: $1.00 per contract per leg
- Exit: $0 if expired OTM, else $1.00 per contract per leg

Slippage by Number of Legs:
- 1 leg: 75% of bid-ask spread
- 2 legs: 66% (credit spreads)
- 3 legs: 56%
- 4 legs: 53% (iron condors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Default slippage percentages by leg count (ORATS methodology)
DEFAULT_SLIPPAGE_BY_LEGS: dict[int, float] = {
    1: 0.75,
    2: 0.66,
    3: 0.56,
    4: 0.53,
}


@dataclass
class CostBreakdown:
    """Breakdown of costs for a single trade action (entry or exit)."""

    slippage_cost: float  # Cost due to bid-ask spread slippage
    commission: float  # Broker commission
    contracts: int
    legs: int

    @property
    def total(self) -> float:
        """Total cost for this action."""
        return self.slippage_cost + self.commission

    @property
    def per_contract(self) -> float:
        """Total cost per contract."""
        if self.contracts == 0:
            return 0.0
        return self.total / self.contracts


@dataclass
class RoundTripCosts:
    """Complete round-trip costs for a trade."""

    entry: CostBreakdown
    exit: CostBreakdown

    @property
    def total(self) -> float:
        """Total round-trip cost."""
        return self.entry.total + self.exit.total

    @property
    def total_slippage(self) -> float:
        """Total slippage cost."""
        return self.entry.slippage_cost + self.exit.slippage_cost

    @property
    def total_commission(self) -> float:
        """Total commission cost."""
        return self.entry.commission + self.exit.commission


class TradeCosts:
    """Calculate round-trip costs per ORATS methodology.

    Example for 1 contract credit spread:
    - Entry: $1 * 1 * 2 legs = $2 commission
    - Exit (not expired): $1 * 1 * 2 legs = $2 commission
    - Round-trip: $4 total commission

    If spread has $0.50 credit with $0.10 bid-ask spread on each leg:
    - Slippage at 66%: ~$0.066 * 100 * 2 legs = ~$13.20
    """

    def __init__(
        self,
        commission_per_contract: float = 1.00,
        slippage_by_legs: dict[int, float] | None = None,
    ):
        """Initialize cost calculator.

        Args:
            commission_per_contract: Commission per contract per leg
            slippage_by_legs: Optional custom slippage percentages
        """
        self.commission_per_contract = commission_per_contract
        self.slippage_by_legs = slippage_by_legs or DEFAULT_SLIPPAGE_BY_LEGS

    def get_slippage_pct(self, legs: int) -> float:
        """Get slippage percentage for given number of legs.

        Args:
            legs: Number of option legs (1-4+)

        Returns:
            Slippage as decimal (e.g., 0.66)
        """
        if legs < 1:
            raise ValueError(f"Invalid number of legs: {legs}")
        if legs > 4:
            return self.slippage_by_legs[4]
        return self.slippage_by_legs[legs]

    def calculate_entry_cost(
        self,
        contracts: int,
        legs: int = 2,
    ) -> float:
        """Calculate entry commission.

        Args:
            contracts: Number of contracts
            legs: Number of legs (default 2 for credit spread)

        Returns:
            Total entry commission in dollars
        """
        return self.commission_per_contract * contracts * legs

    def calculate_exit_cost(
        self,
        contracts: int,
        legs: int = 2,
        expired_otm: bool = False,
    ) -> float:
        """Calculate exit commission.

        Per ORATS: $0 if expired OTM, else $1.00 per contract per leg.

        Args:
            contracts: Number of contracts
            legs: Number of legs
            expired_otm: Whether position expired worthless

        Returns:
            Total exit commission in dollars
        """
        if expired_otm:
            return 0.0
        return self.commission_per_contract * contracts * legs

    def calculate_slippage(
        self,
        bid: float,
        ask: float,
        side: Literal["buy", "sell"],
        legs: int = 2,
    ) -> float:
        """Calculate fill price with ORATS slippage.

        Fill Price Formula:
        - Buy: Bid + (Ask - Bid) * slippage_pct
        - Sell: Ask - (Ask - Bid) * slippage_pct

        Args:
            bid: Bid price
            ask: Ask price
            side: "buy" or "sell"
            legs: Number of legs (affects slippage percentage)

        Returns:
            Simulated fill price
        """
        slippage_pct = self.get_slippage_pct(legs)
        spread = ask - bid

        if side == "buy":
            return bid + (spread * slippage_pct)
        else:
            return ask - (spread * slippage_pct)

    def calculate_slippage_cost(
        self,
        bid: float,
        ask: float,
        side: Literal["buy", "sell"],
        contracts: int,
        legs: int = 2,
    ) -> float:
        """Calculate slippage cost in dollars.

        Slippage cost = difference between mid price and fill price.

        Args:
            bid: Bid price
            ask: Ask price
            side: "buy" or "sell"
            contracts: Number of contracts
            legs: Number of legs

        Returns:
            Slippage cost in dollars (always positive)
        """
        mid = (bid + ask) / 2
        fill = self.calculate_slippage(bid, ask, side, legs)

        if side == "buy":
            # Paid more than mid
            slippage_per_contract = (fill - mid) * 100
        else:
            # Received less than mid
            slippage_per_contract = (mid - fill) * 100

        return max(0, slippage_per_contract * contracts)

    def calculate_spread_entry_costs(
        self,
        short_bid: float,
        short_ask: float,
        long_bid: float,
        long_ask: float,
        contracts: int,
        legs: int = 2,
    ) -> CostBreakdown:
        """Calculate entry costs for a credit spread.

        Entry: Sell short leg, Buy long leg.

        Args:
            short_bid/ask: Quotes for short leg
            long_bid/ask: Quotes for long leg
            contracts: Number of contracts
            legs: Number of legs

        Returns:
            CostBreakdown for entry
        """
        # Slippage on selling short leg
        short_slippage = self.calculate_slippage_cost(
            short_bid, short_ask, "sell", contracts, legs
        )

        # Slippage on buying long leg
        long_slippage = self.calculate_slippage_cost(
            long_bid, long_ask, "buy", contracts, legs
        )

        # Commission
        commission = self.calculate_entry_cost(contracts, legs)

        return CostBreakdown(
            slippage_cost=short_slippage + long_slippage,
            commission=commission,
            contracts=contracts,
            legs=legs,
        )

    def calculate_spread_exit_costs(
        self,
        short_bid: float,
        short_ask: float,
        long_bid: float,
        long_ask: float,
        contracts: int,
        legs: int = 2,
        expired_otm: bool = False,
    ) -> CostBreakdown:
        """Calculate exit costs for a credit spread.

        Exit: Buy back short leg, Sell long leg.

        Args:
            short_bid/ask: Quotes for short leg
            long_bid/ask: Quotes for long leg
            contracts: Number of contracts
            legs: Number of legs
            expired_otm: Whether position expired worthless

        Returns:
            CostBreakdown for exit
        """
        if expired_otm:
            # No slippage or commission on worthless expiration
            return CostBreakdown(
                slippage_cost=0.0,
                commission=0.0,
                contracts=contracts,
                legs=legs,
            )

        # Slippage on buying back short leg
        short_slippage = self.calculate_slippage_cost(
            short_bid, short_ask, "buy", contracts, legs
        )

        # Slippage on selling long leg
        long_slippage = self.calculate_slippage_cost(
            long_bid, long_ask, "sell", contracts, legs
        )

        # Commission
        commission = self.calculate_exit_cost(contracts, legs, expired_otm)

        return CostBreakdown(
            slippage_cost=short_slippage + long_slippage,
            commission=commission,
            contracts=contracts,
            legs=legs,
        )

    def calculate_round_trip(
        self,
        entry_short_bid: float,
        entry_short_ask: float,
        entry_long_bid: float,
        entry_long_ask: float,
        exit_short_bid: float,
        exit_short_ask: float,
        exit_long_bid: float,
        exit_long_ask: float,
        contracts: int,
        legs: int = 2,
        expired_otm: bool = False,
    ) -> RoundTripCosts:
        """Calculate complete round-trip costs for a trade.

        Args:
            entry_*: Entry quotes for both legs
            exit_*: Exit quotes for both legs
            contracts: Number of contracts
            legs: Number of legs
            expired_otm: Whether position expired worthless

        Returns:
            RoundTripCosts with entry and exit breakdown
        """
        entry = self.calculate_spread_entry_costs(
            entry_short_bid,
            entry_short_ask,
            entry_long_bid,
            entry_long_ask,
            contracts,
            legs,
        )

        exit_costs = self.calculate_spread_exit_costs(
            exit_short_bid,
            exit_short_ask,
            exit_long_bid,
            exit_long_ask,
            contracts,
            legs,
            expired_otm,
        )

        return RoundTripCosts(entry=entry, exit=exit_costs)
