"""ORATS API integration for historical options data and backtesting.

This module provides:
- ORATSClient: Protocol interface for ORATS API
- MockORATSClient: Mock implementation for testing without API key
- Data types for options chains and backtest results

Note: User does not have ORATS API key yet. Use MockORATSClient for testing.
"""

from integrations.orats.client import ORATSClient, ORATSClientImpl
from integrations.orats.mock import MockORATSClient
from integrations.orats.types import (
    BacktestResults,
    BacktestStatus,
    BacktestSubmission,
    OptionData,
    OptionsChain,
)

__all__ = [
    "ORATSClient",
    "ORATSClientImpl",
    "MockORATSClient",
    "OptionsChain",
    "OptionData",
    "BacktestSubmission",
    "BacktestStatus",
    "BacktestResults",
]
