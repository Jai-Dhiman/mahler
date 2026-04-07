"""Backtest configuration with ORATS-compliant slippage methodology.

ORATS Slippage by Number of Legs:
- 1 leg: 75% of bid-ask spread
- 2 legs: 66% (credit spreads)
- 3 legs: 56%
- 4 legs: 53% (iron condors)

Fill Price Formula:
- Buy: Bid + (Ask - Bid) * slippage_pct
- Sell: Ask - (Ask - Bid) * slippage_pct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class MarketRegime(str, Enum):
    """Market volatility regime for regime-specific analysis."""

    LOW_VOL = "low_vol"  # VIX < 15
    NORMAL = "normal"  # VIX 15-25
    ELEVATED = "elevated"  # VIX 25-35
    HIGH_VOL = "high_vol"  # VIX 35-50
    CRISIS = "crisis"  # VIX > 50


@dataclass
class BacktestConfig:
    """Configuration for backtesting with ORATS-compliant assumptions.

    Slippage values are based on ORATS backtester methodology, which assumes
    fills occur at a percentage of the bid-ask spread depending on trade complexity.
    """

    # Slippage by number of legs (ORATS methodology)
    # Higher legs = lower slippage due to market maker efficiency
    slippage_by_legs: dict[int, float] = field(
        default_factory=lambda: {
            1: 0.75,  # 75% for single leg
            2: 0.66,  # 66% for 2-leg spreads (credit spreads)
            3: 0.56,  # 56% for 3-leg
            4: 0.53,  # 53% for 4-leg (iron condors)
        }
    )

    # Commission structure (Alpaca-based)
    # Per ORATS: $1.00 per contract per leg on entry
    # Exit: $0 if expired OTM, else $1.00 per contract per leg
    commission_per_contract: float = 1.00
    commission_per_share: float = 0.01  # For underlying trades if needed

    # Entry criteria
    dte_min: int = 30
    dte_max: int = 45
    short_delta_min: float = 0.10  # Research: further OTM is more consistent
    short_delta_max: float = 0.15
    iv_percentile_min: float = 50.0

    # Exit criteria
    profit_target_pct: float = 0.50  # 50% of max credit
    stop_loss_pct: float = 1.25  # 125% of credit (loss = 1.25x credit received)
    dte_exit: int = 21  # Time-based exit at 21 DTE

    # Position sizing
    max_risk_per_trade_pct: float = 2.0  # 2% of account per trade
    max_portfolio_risk_pct: float = 10.0  # 10% total portfolio risk

    # Walk-forward settings
    train_months: int = 6
    validate_months: int = 1
    test_months: int = 1
    min_trades_per_period: int = 20

    def get_slippage(self, legs: int) -> float:
        """Get slippage percentage for given number of legs.

        Args:
            legs: Number of option legs in the trade (1-4)

        Returns:
            Slippage as decimal (e.g., 0.66 for 66%)
        """
        if legs < 1:
            raise ValueError(f"Invalid number of legs: {legs}")
        if legs > 4:
            # For complex trades, use 4-leg slippage
            return self.slippage_by_legs[4]
        return self.slippage_by_legs[legs]


@dataclass
class RegimeMetrics:
    """Performance metrics for a specific market regime."""

    regime: MarketRegime
    total_trades: int
    win_rate: float
    avg_pnl: float
    avg_pnl_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float


@dataclass
class BacktestResults:
    """Complete backtest results with per-regime breakdown."""

    total_trades: int
    win_rate: float  # Target: >= 70%
    profit_factor: float  # Target: >= 1.5
    sharpe_ratio: float  # Target: >= 1.0
    max_drawdown_pct: float  # Target: <= 15%
    avg_days_in_trade: float
    cumulative_return_pct: float
    annualized_return_pct: float

    # Breakdown by regime
    performance_by_regime: dict[MarketRegime, RegimeMetrics] = field(
        default_factory=dict
    )

    # Cost breakdown
    total_commissions: float = 0.0
    total_slippage_cost: float = 0.0
    net_pnl: float = 0.0
    gross_pnl: float = 0.0

    def cost_impact_pct(self) -> float:
        """Calculate percentage impact of costs on gross P/L."""
        if self.gross_pnl == 0:
            return 0.0
        total_costs = self.total_commissions + self.total_slippage_cost
        return total_costs / self.gross_pnl * 100
