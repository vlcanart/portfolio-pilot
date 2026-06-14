"""Market-data abstraction.

The rest of the app depends only on the `MarketDataProvider` interface, so swapping
yfinance for Finnhub / Alpaca / Twelve Data later is a one-class change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """Interface every data source must implement."""

    @abstractmethod
    def latest_prices(self, tickers: list[str]) -> dict[str, float]:
        """Most recent price per ticker."""

    @abstractmethod
    def history(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        """Adjusted-close price history. Columns = tickers, index = dates."""


class YFinanceProvider(MarketDataProvider):
    """Free, no-key provider. Great for prototyping; unreliable for production —
    see ROADMAP for the Finnhub/Alpaca swap."""

    def __init__(self) -> None:
        import yfinance  # imported lazily so the package imports without it

        self._yf = yfinance

    def latest_prices(self, tickers: list[str]) -> dict[str, float]:
        hist = self.history(tickers, period="5d")
        if hist.empty:
            return {}
        last = hist.ffill().iloc[-1]
        return {t: float(last[t]) for t in hist.columns if pd.notna(last[t])}

    def history(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        raw = self._yf.download(
            tickers, period=period, auto_adjust=True, progress=False
        )
        if raw.empty:
            return pd.DataFrame()
        # With multiple tickers yfinance returns a column MultiIndex; pick Close.
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):  # single ticker
            close = close.to_frame(name=tickers[0])
        return close.dropna(how="all")


def get_default_provider() -> MarketDataProvider:
    return YFinanceProvider()
