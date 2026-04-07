"""Mock ORATS client for testing without API key.

Generates realistic mock options chains based on:
- Historical underlying prices from yfinance
- Synthetic Greeks using Black-Scholes
- Realistic bid-ask spreads
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timedelta

import numpy as np

from integrations.orats.types import (
    BacktestJobStatus,
    BacktestResults,
    BacktestStatus,
    BacktestSubmission,
    BacktestTradeResult,
    OptionData,
    OptionsChain,
)


class MockORATSClient:
    """Mock ORATS client using synthetic data for testing.

    Generates realistic options chains without requiring ORATS API access.
    Uses Black-Scholes for theoretical pricing and Greeks.
    """

    def __init__(self, seed: int | None = 42):
        """Initialize mock client.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self._submitted_jobs: dict[str, BacktestSubmission] = {}
        self._job_results: dict[str, BacktestResults] = {}

    async def submit_backtest(self, config: BacktestSubmission) -> str:
        """Submit a mock backtest job."""
        job_id = f"mock_{uuid.uuid4().hex[:8]}"
        self._submitted_jobs[job_id] = config

        # Generate mock results immediately
        self._job_results[job_id] = self._generate_mock_results(job_id, config)

        return job_id

    async def get_backtest_status(self, job_id: str) -> BacktestStatus:
        """Get status of mock backtest (always completed)."""
        if job_id not in self._submitted_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")

        return BacktestStatus(
            job_id=job_id,
            status=BacktestJobStatus.COMPLETED,
            submitted_at=datetime.now(),
            completed_at=datetime.now(),
            progress_pct=100.0,
        )

    async def get_backtest_results(self, job_id: str) -> BacktestResults:
        """Get results of mock backtest."""
        if job_id not in self._job_results:
            raise ValueError(f"No results for job ID: {job_id}")
        return self._job_results[job_id]

    async def get_historical_chain(
        self, symbol: str, quote_date: date
    ) -> OptionsChain:
        """Generate realistic mock options chain.

        Uses synthetic data based on historical patterns.
        """
        # Get a synthetic underlying price
        underlying_price = self._get_mock_price(symbol, quote_date)

        # Generate expirations (weekly + monthly)
        expirations = self._generate_expirations(quote_date)

        # Generate IV metrics
        base_iv = 0.20 + self.rng.random() * 0.15  # 20-35% IV
        iv_percentile = self.rng.random() * 100

        options = []
        for exp in expirations:
            dte = (exp - quote_date).days
            if dte <= 0:
                continue

            # Generate strikes around ATM
            strikes = self._generate_strikes(underlying_price)

            for strike in strikes:
                # Generate call
                call = self._generate_option(
                    symbol=symbol,
                    underlying_price=underlying_price,
                    quote_date=quote_date,
                    expiration=exp,
                    strike=strike,
                    option_type="call",
                    base_iv=base_iv,
                )
                options.append(call)

                # Generate put
                put = self._generate_option(
                    symbol=symbol,
                    underlying_price=underlying_price,
                    quote_date=quote_date,
                    expiration=exp,
                    strike=strike,
                    option_type="put",
                    base_iv=base_iv,
                )
                options.append(put)

        return OptionsChain(
            underlying_symbol=symbol,
            quote_date=quote_date,
            underlying_price=underlying_price,
            options=options,
            iv_30d_atm=base_iv,
            iv_percentile=iv_percentile,
            iv_rank=iv_percentile * 0.9,  # IV rank typically lower
        )

    def _get_mock_price(self, symbol: str, quote_date: date) -> float:
        """Get mock underlying price.

        Uses approximate historical levels with some randomness.
        """
        # Base prices by symbol (approximate 2024 levels)
        base_prices = {
            "SPY": 500.0,
            "QQQ": 450.0,
            "IWM": 200.0,
            "TLT": 95.0,
            "GLD": 215.0,
        }

        base = base_prices.get(symbol, 100.0)

        # Add some random drift based on date
        days_offset = (quote_date - date(2024, 1, 1)).days
        drift = self.rng.normal(0, 0.01) * days_offset

        return base * (1 + drift)

    def _generate_expirations(self, quote_date: date) -> list[date]:
        """Generate realistic expiration dates."""
        expirations = []

        # Weekly expirations for next 8 weeks
        for i in range(1, 9):
            # Find next Friday
            days_until_friday = (4 - quote_date.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            exp = quote_date + timedelta(days=days_until_friday + (i - 1) * 7)
            expirations.append(exp)

        # Monthly expirations for next 3 months
        current_month = quote_date.month
        current_year = quote_date.year

        for i in range(1, 4):
            month = ((current_month - 1 + i) % 12) + 1
            year = current_year + ((current_month - 1 + i) // 12)

            # Third Friday of month
            first_day = date(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=days_until_friday + 14)

            if third_friday > quote_date and third_friday not in expirations:
                expirations.append(third_friday)

        return sorted(expirations)

    def _generate_strikes(self, underlying_price: float) -> list[float]:
        """Generate strike prices around ATM."""
        # Round to nearest $5 for strike spacing
        atm = round(underlying_price / 5) * 5

        # Generate strikes +/- 15%
        strikes = []
        for pct in range(-15, 16, 1):
            strike = atm + (pct * 5)
            if strike > 0:
                strikes.append(strike)

        return strikes

    def _generate_option(
        self,
        symbol: str,
        underlying_price: float,
        quote_date: date,
        expiration: date,
        strike: float,
        option_type: str,
        base_iv: float,
    ) -> OptionData:
        """Generate a single option with realistic Greeks."""
        dte = (expiration - quote_date).days
        t = dte / 365.0
        r = 0.05  # Risk-free rate

        # IV skew - OTM puts have higher IV
        moneyness = strike / underlying_price
        if option_type == "put":
            iv_adj = base_iv * (1 + 0.1 * max(0, 1 - moneyness))
        else:
            iv_adj = base_iv * (1 + 0.05 * max(0, moneyness - 1))

        # Black-Scholes pricing and Greeks
        price, delta, gamma, theta, vega = self._black_scholes(
            S=underlying_price,
            K=strike,
            t=t,
            r=r,
            sigma=iv_adj,
            option_type=option_type,
        )

        # Add realistic bid-ask spread
        # Tighter spreads for ATM, wider for OTM
        base_spread = 0.02 + 0.03 * abs(1 - moneyness)
        spread = max(0.01, price * base_spread)

        bid = max(0.01, price - spread / 2)
        ask = price + spread / 2
        mid = (bid + ask) / 2

        # Mock volume/OI
        volume = int(self.rng.exponential(500) * (1 - abs(1 - moneyness)))
        oi = int(self.rng.exponential(5000) * (1 - abs(1 - moneyness) * 0.5))

        return OptionData(
            underlying_symbol=symbol,
            underlying_price=underlying_price,
            quote_date=quote_date,
            expiration_date=expiration,
            strike=strike,
            option_type=option_type,
            bid=round(bid, 2),
            ask=round(ask, 2),
            mid=round(mid, 2),
            volume=max(0, volume),
            open_interest=max(0, oi),
            implied_volatility=round(iv_adj, 4),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(delta * 0.01 * t, 4),  # Approximate rho
        )

    def _black_scholes(
        self,
        S: float,
        K: float,
        t: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> tuple[float, float, float, float, float]:
        """Calculate Black-Scholes price and Greeks.

        Returns: (price, delta, gamma, theta, vega)
        """
        if t <= 0:
            # Expired
            if option_type == "call":
                intrinsic = max(0, S - K)
            else:
                intrinsic = max(0, K - S)
            return intrinsic, 0.0, 0.0, 0.0, 0.0

        sqrt_t = math.sqrt(t)

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        # Standard normal CDF and PDF
        def norm_cdf(x: float) -> float:
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        def norm_pdf(x: float) -> float:
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

        n_d1 = norm_cdf(d1)
        n_d2 = norm_cdf(d2)
        n_prime_d1 = norm_pdf(d1)

        if option_type == "call":
            price = S * n_d1 - K * math.exp(-r * t) * n_d2
            delta = n_d1
        else:
            price = K * math.exp(-r * t) * (1 - n_d2) - S * (1 - n_d1)
            delta = n_d1 - 1

        # Greeks
        gamma = n_prime_d1 / (S * sigma * sqrt_t)
        theta = (
            -S * n_prime_d1 * sigma / (2 * sqrt_t)
            - r * K * math.exp(-r * t) * (n_d2 if option_type == "call" else (1 - n_d2))
        ) / 365
        vega = S * n_prime_d1 * sqrt_t / 100  # Per 1% IV change

        return max(0.01, price), delta, gamma, theta, vega

    def _generate_mock_results(
        self, job_id: str, config: BacktestSubmission
    ) -> BacktestResults:
        """Generate mock backtest results."""
        # Simulate trades
        num_trades = self.rng.integers(50, 150)
        trades: list[BacktestTradeResult] = []

        current_date = config.start_date
        total_pnl = 0.0
        wins = 0
        losses = 0

        for i in range(num_trades):
            # Random entry within date range
            days_range = (config.end_date - config.start_date).days
            entry_offset = self.rng.integers(0, max(1, days_range))
            entry_date = config.start_date + timedelta(days=int(entry_offset))

            if entry_date >= config.end_date:
                break

            # Generate trade details
            underlying_price = self._get_mock_price(config.underlying, entry_date)
            short_strike = round(underlying_price * (1 - config.entry_delta_min), 0)
            long_strike = short_strike - 5  # $5 wide spread

            expiration = entry_date + timedelta(days=config.entry_dte_max)

            # Random outcome
            is_win = self.rng.random() > 0.3  # ~70% win rate
            if is_win:
                exit_reason = "profit_target"
                pnl = self.rng.uniform(30, 100)  # $30-100 profit
                wins += 1
            else:
                exit_reason = self.rng.choice(["stop_loss", "time_exit"])
                pnl = self.rng.uniform(-200, -50)  # $50-200 loss
                losses += 1

            days_in_trade = self.rng.integers(5, 30)
            exit_date = entry_date + timedelta(days=int(days_in_trade))

            entry_credit = self.rng.uniform(0.30, 0.80)
            exit_debit = entry_credit - (pnl / 100)

            trades.append(
                BacktestTradeResult(
                    trade_id=f"trade_{i}",
                    entry_date=entry_date,
                    exit_date=exit_date,
                    underlying=config.underlying,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    expiration=expiration,
                    contracts=config.contracts_per_trade,
                    entry_credit=round(entry_credit, 2),
                    exit_debit=round(max(0, exit_debit), 2),
                    gross_pnl=round(pnl, 2),
                    net_pnl=round(pnl - 4, 2),  # $4 round-trip costs
                    exit_reason=exit_reason,
                    entry_iv_percentile=self.rng.uniform(50, 80),
                    entry_vix=self.rng.uniform(15, 25),
                )
            )

            total_pnl += pnl

        # Calculate metrics
        total_trades = len(trades)
        if total_trades == 0:
            win_rate = 0.0
            profit_factor = 0.0
        else:
            win_rate = wins / total_trades
            gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
            gross_loss = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99

        # Approximate Sharpe
        if trades:
            returns = [t.net_pnl / 500 for t in trades]  # Normalize by ~max risk
            sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(12)
                if np.std(returns) > 0
                else 0
            )
        else:
            sharpe = 0.0

        return BacktestResults(
            job_id=job_id,
            strategy_name=config.strategy_name,
            underlying=config.underlying,
            start_date=config.start_date,
            end_date=config.end_date,
            total_trades=total_trades,
            win_count=wins,
            loss_count=losses,
            win_rate=win_rate,
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(float(sharpe), 2),
            max_drawdown=round(abs(total_pnl * 0.15), 2),  # Mock drawdown
            max_drawdown_pct=15.0,
            avg_days_in_trade=15.0,
            avg_dte_at_exit=config.dte_exit,
            total_slippage=round(total_trades * 13.20, 2),  # ~$13.20 per trade
            total_commission=round(total_trades * 4.0, 2),  # $4 round-trip
            trades=trades,
        )
