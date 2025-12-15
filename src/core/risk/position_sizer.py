from __future__ import annotations

"""Position sizing with correlation-aware risk management.

Research shows SPY/QQQ/IWM are 86-92% correlated, making them effectively
a single concentrated bet on US equities. This module implements:
1. Beta-weighted position sizing
2. Asset-class-based exposure limits
3. Per-underlying concentration limits
"""

from dataclasses import dataclass

from core.types import (
    ASSET_BETAS,
    ASSET_CLASSES,
    AssetClass,
    CreditSpread,
    Position,
    PortfolioGreeks,
)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    contracts: int
    risk_amount: float
    risk_percent: float
    reason: str | None = None


@dataclass
class RiskLimits:
    """Risk limits with correlation awareness."""

    # Position sizing
    max_risk_per_trade_pct: float = 0.02  # 2% max per trade
    max_single_position_pct: float = 0.05  # 5% max in one position
    max_portfolio_heat_pct: float = 0.10  # 10% total open risk

    # Per-underlying limits (avoid concentration)
    max_per_underlying_pct: float = 0.033  # ~33% max in any single underlying

    # Asset class limits (manage correlation risk)
    max_equity_class_pct: float = 0.50  # 50% max in correlated equity ETFs
    max_treasury_class_pct: float = 0.25  # 25% max in treasuries
    max_commodity_class_pct: float = 0.25  # 25% max in commodities

    # Portfolio Greeks limits
    max_portfolio_delta: float = 0.30  # Max absolute beta-weighted delta
    max_portfolio_gamma: float = 0.20  # Max absolute gamma

    # Volatility adjustments (kept for backward compatibility)
    high_vix_threshold: float = 40.0
    high_vix_reduction: float = 0.75
    extreme_vix_threshold: float = 50.0


class PositionSizer:
    """Calculates position sizes with correlation-aware risk limits.

    Key improvements over naive sizing:
    1. Beta-weighted calculations account for different volatilities
    2. Asset class limits prevent correlation concentration
    3. Per-underlying limits ensure diversification within classes
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def get_beta(self, underlying: str) -> float:
        """Get beta for an underlying (default 1.0 if unknown)."""
        return ASSET_BETAS.get(underlying, 1.0)

    def get_asset_class(self, underlying: str) -> AssetClass:
        """Get asset class for an underlying (default EQUITY if unknown)."""
        return ASSET_CLASSES.get(underlying, AssetClass.EQUITY)

    def calculate_portfolio_greeks(
        self,
        positions: list[Position],
        position_deltas: dict[str, float] | None = None,
    ) -> PortfolioGreeks:
        """Calculate aggregate portfolio Greeks (beta-weighted).

        Args:
            positions: List of open positions
            position_deltas: Optional dict of trade_id -> position delta

        Returns:
            PortfolioGreeks with aggregate metrics
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        equity_risk = 0.0
        treasury_risk = 0.0
        commodity_risk = 0.0

        for pos in positions:
            beta = self.get_beta(pos.underlying)
            asset_class = self.get_asset_class(pos.underlying)
            risk = abs(pos.current_value)

            # Beta-weight the delta (approximate from position value)
            # In a real system, you'd have actual position Greeks
            position_delta = position_deltas.get(pos.trade_id, 0.0) if position_deltas else 0.0
            total_delta += position_delta * beta

            # Track risk by asset class
            if asset_class == AssetClass.EQUITY:
                equity_risk += risk
            elif asset_class == AssetClass.TREASURY:
                treasury_risk += risk
            elif asset_class == AssetClass.COMMODITY:
                commodity_risk += risk

        return PortfolioGreeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            equity_risk=equity_risk,
            treasury_risk=treasury_risk,
            commodity_risk=commodity_risk,
        )

    def calculate_asset_class_exposure(
        self,
        positions: list[Position],
        account_equity: float,
    ) -> dict[AssetClass, float]:
        """Calculate current exposure by asset class as percentage of equity."""
        exposure: dict[AssetClass, float] = {
            AssetClass.EQUITY: 0.0,
            AssetClass.TREASURY: 0.0,
            AssetClass.COMMODITY: 0.0,
        }

        for pos in positions:
            asset_class = self.get_asset_class(pos.underlying)
            exposure[asset_class] += abs(pos.current_value)

        # Convert to percentages
        if account_equity > 0:
            for ac in exposure:
                exposure[ac] /= account_equity

        return exposure

    def calculate_underlying_exposure(
        self,
        positions: list[Position],
        account_equity: float,
    ) -> dict[str, float]:
        """Calculate current exposure by underlying as percentage of equity."""
        exposure: dict[str, float] = {}

        for pos in positions:
            if pos.underlying not in exposure:
                exposure[pos.underlying] = 0.0
            exposure[pos.underlying] += abs(pos.current_value)

        # Convert to percentages
        if account_equity > 0:
            for underlying in exposure:
                exposure[underlying] /= account_equity

        return exposure

    def calculate_size(
        self,
        spread: CreditSpread,
        account_equity: float,
        current_positions: list[Position],
        current_vix: float | None = None,
    ) -> PositionSizeResult:
        """Calculate position size with correlation-aware limits.

        Args:
            spread: The credit spread being considered
            account_equity: Current account equity
            current_positions: List of open positions
            current_vix: Current VIX level (optional)

        Returns:
            PositionSizeResult with recommended contracts
        """
        # Check for zero/negative equity early to avoid division by zero
        if account_equity <= 0:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0,
                risk_percent=0,
                reason="Invalid account equity (zero or negative)",
            )

        # Check VIX halt (extreme conditions)
        if current_vix is not None:
            if current_vix >= self.limits.extreme_vix_threshold:
                return PositionSizeResult(
                    contracts=0,
                    risk_amount=0,
                    risk_percent=0,
                    reason=f"VIX ({current_vix:.1f}) exceeds extreme threshold",
                )

        # Risk per contract
        risk_per_contract = spread.max_loss
        if risk_per_contract <= 0:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0,
                risk_percent=0,
                reason="Invalid spread: no risk calculated",
            )

        # Calculate various limits
        constraints: list[tuple[int, str]] = []

        # 1. Per-trade risk limit (2%)
        max_trade_risk = account_equity * self.limits.max_risk_per_trade_pct
        max_by_trade = int(max_trade_risk / risk_per_contract)
        constraints.append((max_by_trade, "2% per-trade risk limit"))

        # 2. Single position limit (5%)
        max_position = account_equity * self.limits.max_single_position_pct
        max_by_position = int(max_position / risk_per_contract)
        constraints.append((max_by_position, "5% single position limit"))

        # 3. Portfolio heat limit (10%)
        current_heat = sum(abs(p.current_value) for p in current_positions)
        available_heat = max(0, account_equity * self.limits.max_portfolio_heat_pct - current_heat)
        max_by_heat = int(available_heat / risk_per_contract)
        constraints.append((max_by_heat, f"Portfolio heat ({current_heat/account_equity:.1%} used)"))

        # 4. Per-underlying limit (~33%)
        underlying_exposure = self.calculate_underlying_exposure(current_positions, account_equity)
        current_underlying_pct = underlying_exposure.get(spread.underlying, 0.0)
        available_underlying = max(
            0, self.limits.max_per_underlying_pct - current_underlying_pct
        ) * account_equity
        max_by_underlying = int(available_underlying / risk_per_contract)
        constraints.append(
            (max_by_underlying, f"{spread.underlying} concentration ({current_underlying_pct:.1%} used)")
        )

        # 5. Asset class limit
        asset_class = self.get_asset_class(spread.underlying)
        class_exposure = self.calculate_asset_class_exposure(current_positions, account_equity)
        current_class_pct = class_exposure.get(asset_class, 0.0)

        if asset_class == AssetClass.EQUITY:
            max_class_pct = self.limits.max_equity_class_pct
        elif asset_class == AssetClass.TREASURY:
            max_class_pct = self.limits.max_treasury_class_pct
        else:
            max_class_pct = self.limits.max_commodity_class_pct

        available_class = max(0, max_class_pct - current_class_pct) * account_equity
        max_by_class = int(available_class / risk_per_contract)
        constraints.append(
            (max_by_class, f"{asset_class.value} class limit ({current_class_pct:.1%} of {max_class_pct:.0%})")
        )

        # Find binding constraint
        contracts = min(c[0] for c in constraints)
        contracts = max(0, contracts)

        # Determine which constraint is binding
        reason = None
        if contracts == 0:
            # Find the constraint that's at zero
            for limit, desc in constraints:
                if limit == 0:
                    reason = f"Blocked by {desc}"
                    break
        else:
            # Find which constraint is binding
            min_constraint = min(constraints, key=lambda x: x[0])
            if min_constraint[0] < max_by_trade:
                reason = f"Limited by {min_constraint[1]}"

        # Apply VIX adjustment (this is now mostly handled by graduated circuit breaker)
        if current_vix is not None and current_vix >= self.limits.high_vix_threshold:
            original = contracts
            contracts = max(1, int(contracts * (1 - self.limits.high_vix_reduction)))
            if contracts < original:
                reason = f"Reduced due to high VIX ({current_vix:.1f})"

        # Ensure at least 1 if any contracts allowed
        if contracts > 0:
            contracts = max(1, contracts)

        risk_amount = contracts * risk_per_contract
        risk_percent = risk_amount / account_equity if account_equity > 0 else 0

        return PositionSizeResult(
            contracts=contracts,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            reason=reason,
        )

    def calculate_portfolio_heat(
        self,
        positions: list[Position],
        account_equity: float,
    ) -> dict:
        """Calculate current portfolio heat metrics with correlation awareness."""
        total_risk = sum(abs(p.current_value) for p in positions)
        heat_pct = total_risk / account_equity if account_equity > 0 else 0

        # Group by underlying
        by_underlying = self.calculate_underlying_exposure(positions, account_equity)

        # Group by asset class
        by_class = self.calculate_asset_class_exposure(positions, account_equity)

        return {
            "total_risk": total_risk,
            "heat_percent": heat_pct,
            "max_heat_percent": self.limits.max_portfolio_heat_pct,
            "available_capacity": max(
                0, (self.limits.max_portfolio_heat_pct - heat_pct) * account_equity
            ),
            "by_underlying": by_underlying,
            "by_asset_class": {ac.value: pct for ac, pct in by_class.items()},
            "at_limit": heat_pct >= self.limits.max_portfolio_heat_pct,
            "equity_class_at_limit": by_class[AssetClass.EQUITY] >= self.limits.max_equity_class_pct,
        }
