"""Data fetcher for model training.

Fetches training data from Alpaca (historical bars) and Cloudflare D1 (trades).
Falls back to yfinance if Alpaca fails.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import httpx


async def fetch_spy_bars_alpaca(days: int = 750) -> list[dict]:
    """Fetch SPY historical bars from Alpaca API.

    Args:
        days: Number of days of historical data to fetch (default 750 = ~3 years)

    Returns:
        List of bar dictionaries with keys: timestamp, open, high, low, close, volume

    Raises:
        Exception if Alpaca API fails
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required")

    # Use data API endpoint
    base_url = "https://data.alpaca.markets/v2"

    # Calculate date range (end yesterday to ensure complete data)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    # Alpaca paginates results, need to handle next_page_token
    bars = []
    next_page_token = None

    async with httpx.AsyncClient() as client:
        while True:
            params = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "timeframe": "1Day",
                "adjustment": "split",
                "limit": 10000,  # Max per request
            }
            if next_page_token:
                params["page_token"] = next_page_token

            response = await client.get(
                f"{base_url}/stocks/SPY/bars",
                headers=headers,
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()

            data = response.json()
            for bar in data.get("bars", []):
                bars.append({
                    "timestamp": bar["t"],
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                })

            # Check for more pages
            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

    print(f"Fetched {len(bars)} SPY bars from Alpaca ({days} days requested)")
    return bars


def fetch_spy_bars_yfinance(days: int = 750) -> list[dict]:
    """Fetch SPY historical bars from Yahoo Finance (fallback).

    Args:
        days: Number of days of historical data to fetch

    Returns:
        List of bar dictionaries with keys: timestamp, open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fetch data
    spy = yf.Ticker("SPY")
    df = spy.history(start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")

    bars = []
    for idx, row in df.iterrows():
        bars.append({
            "timestamp": idx.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        })

    print(f"Fetched {len(bars)} SPY bars from Yahoo Finance ({days} days requested)")
    return bars


async def fetch_spy_bars(days: int = 750) -> list[dict]:
    """Fetch SPY historical bars, with fallback to Yahoo Finance.

    Tries Alpaca first, falls back to yfinance if Alpaca fails.

    Args:
        days: Number of days of historical data to fetch (default 750 = ~3 years)

    Returns:
        List of bar dictionaries with keys: timestamp, open, high, low, close, volume
    """
    # Try Alpaca first
    try:
        return await fetch_spy_bars_alpaca(days)
    except Exception as e:
        print(f"Alpaca API failed: {e}")
        print("Falling back to Yahoo Finance...")

    # Fallback to yfinance (synchronous but wrapped)
    return fetch_spy_bars_yfinance(days)


async def fetch_trades_from_d1() -> list[dict]:
    """Fetch closed trades from Cloudflare D1.

    Uses Cloudflare API to query the D1 database directly.

    Returns:
        List of trade dictionaries
    """
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.environ.get("D1_DATABASE_ID")

    if not all([api_token, account_id, database_id]):
        print("Missing Cloudflare credentials, skipping D1 trade fetch")
        return []

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # Query for closed trades with regime and recommendation data
    query = """
        SELECT
            t.id,
            t.profit_loss,
            t.entry_credit,
            t.exit_debit,
            t.opened_at,
            t.closed_at,
            t.dte_at_exit,
            r.iv_rank,
            r.delta as short_delta,
            r.credit,
            r.expiration,
            mr.regime
        FROM trades t
        JOIN recommendations r ON t.recommendation_id = r.id
        LEFT JOIN market_regimes mr ON
            mr.symbol = r.underlying AND
            DATE(mr.detected_at) = DATE(t.opened_at)
        WHERE t.status = 'closed'
        ORDER BY t.closed_at DESC
        LIMIT 500
    """

    payload = {"sql": query}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()

        data = response.json()
        if not data.get("success"):
            print(f"D1 query failed: {data.get('errors')}")
            return []

        results = data.get("result", [{}])[0].get("results", [])
        print(f"Fetched {len(results)} closed trades from D1")
        return results


async def fetch_training_data(days: int = 750) -> dict:
    """Fetch all training data.

    Args:
        days: Number of days of historical bars to fetch (default 750 = ~3 years)

    Returns:
        Dictionary with 'bars' and 'trades' keys
    """
    bars = await fetch_spy_bars(days=days)
    trades = await fetch_trades_from_d1()

    return {
        "bars": bars,
        "trades": trades,
    }


if __name__ == "__main__":
    import asyncio
    import json

    async def main():
        data = await fetch_training_data()
        print(f"Bars: {len(data['bars'])}")
        print(f"Trades: {len(data['trades'])}")

        # Optionally save to file
        with open("/tmp/training_data.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        print("Saved to /tmp/training_data.json")

    asyncio.run(main())
