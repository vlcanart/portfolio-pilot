"""Market-data abstraction.

The rest of the app depends only on the `MarketDataProvider` interface, so swapping
yfinance for Finnhub / Alpaca / Twelve Data later is a one-class change.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import pandas as pd

# Fundamental fields the app expects from any provider's fundamentals().
FUNDAMENTAL_COLS = ["name", "pe", "rev_growth", "profit_margin", "debt_to_equity", "market_cap"]


class MarketDataProvider(ABC):
    """Interface every data source must implement."""

    @abstractmethod
    def latest_prices(self, tickers: list[str]) -> dict[str, float]:
        """Most recent price per ticker."""

    @abstractmethod
    def history(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        """Adjusted-close price history. Columns = tickers, index = dates."""

    def fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Per-ticker fundamentals indexed by ticker (FUNDAMENTAL_COLS). Best-effort."""
        return pd.DataFrame(columns=FUNDAMENTAL_COLS)


class YFinanceProvider(MarketDataProvider):
    """Free, no-key provider. Good for prototyping. Price history is reliable; the
    intermittent SQLite 'database is locked' error is handled with a short retry."""

    def __init__(self, retries: int = 3) -> None:
        import yfinance  # imported lazily so the package imports without it

        self._yf = yfinance
        self._retries = retries

    def latest_prices(self, tickers: list[str]) -> dict[str, float]:
        hist = self.history(tickers, period="5d")
        if hist.empty:
            return {}
        last = hist.ffill().iloc[-1]
        return {t: float(last[t]) for t in hist.columns if pd.notna(last[t])}

    def history(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        raw = None
        for attempt in range(self._retries):
            try:
                raw = self._yf.download(
                    tickers, period=period, auto_adjust=True, progress=False,
                    threads=False,  # avoids the concurrent SQLite cache-lock on Windows
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:
                if attempt == self._retries - 1:
                    raise
            time.sleep(0.6 * (attempt + 1))
        if raw is None or raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):  # single ticker
            close = close.to_frame(name=tickers[0])
        return close.dropna(how="all")

    def fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Best-effort fundamentals via yfinance .info (one network call per ticker)."""
        rows = []
        for t in tickers:
            d = {}
            try:
                info = self._yf.Ticker(t).info
            except Exception:
                info = {}
            dte = info.get("debtToEquity")
            rows.append({
                "ticker": t,
                "name": info.get("shortName") or info.get("longName"),
                "pe": info.get("trailingPE"),
                "rev_growth": info.get("revenueGrowth"),
                "profit_margin": info.get("profitMargins"),
                # yfinance reports D/E as a percent (e.g. 38.0 = 0.38x); normalize to a ratio.
                "debt_to_equity": (dte / 100.0) if isinstance(dte, (int, float)) else None,
                "market_cap": info.get("marketCap"),
            })
            d.clear()
        df = pd.DataFrame(rows)
        return df.set_index("ticker") if not df.empty else pd.DataFrame(columns=FUNDAMENTAL_COLS)


def get_default_provider() -> MarketDataProvider:
    """Return the best available provider. Finnhub (reliable, real fundamentals) is used
    when FINNHUB_API_KEY is set; otherwise yfinance."""
    try:
        from .config import settings
        if settings.finnhub_api_key:
            return FinnhubProvider(settings.finnhub_api_key)
    except Exception:
        pass
    return YFinanceProvider()


class FinnhubProvider(MarketDataProvider):
    """Reliable provider with real fundamentals, used when FINNHUB_API_KEY is set.

    Uses Finnhub's /quote (latest price), /stock/candle (history), and /stock/metric
    (fundamentals) endpoints. Falls back to yfinance for history if the candle endpoint
    is unavailable on the free tier.
    """

    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str) -> None:
        import requests  # bundled via yfinance's deps

        self._requests = requests
        self._key = api_key
        self._fallback = YFinanceProvider()

    def _get(self, path: str, **params) -> dict | list:
        params["token"] = self._key
        r = self._requests.get(f"{self.BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def latest_prices(self, tickers: list[str]) -> dict[str, float]:
        out = {}
        for t in tickers:
            try:
                q = self._get("/quote", symbol=t)
                if q.get("c"):
                    out[t] = float(q["c"])
            except Exception:
                continue
        return out

    def history(self, tickers: list[str], period: str = "1y") -> pd.DataFrame:
        # Free-tier candle access is restricted; reuse yfinance for clean adjusted history.
        return self._fallback.history(tickers, period=period)

    def fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                m = self._get("/stock/metric", symbol=t, metric="all").get("metric", {})
            except Exception:
                m = {}
            rows.append({
                "ticker": t,
                "name": None,
                "pe": m.get("peTTM"),
                "rev_growth": (m.get("revenueGrowthTTMYoy") or 0) / 100.0 if m.get("revenueGrowthTTMYoy") else None,
                "profit_margin": (m.get("netProfitMarginTTM") or 0) / 100.0 if m.get("netProfitMarginTTM") else None,
                "debt_to_equity": m.get("totalDebt/totalEquityQuarterly"),
                "market_cap": (m.get("marketCapitalization") or 0) * 1e6 or None,
            })
        df = pd.DataFrame(rows)
        return df.set_index("ticker") if not df.empty else pd.DataFrame(columns=FUNDAMENTAL_COLS)
