"""Historical-performance comparison chart for all tracked + held names.

Pulls aligned history for the union of holdings and the thesis watchlist, then returns a
normalized (start=100) wide DataFrame suitable for st.line_chart. Supports three time
granularities — daily, monthly (month-end), and yearly (year-end) — and an optional
benchmark overlay (SPY by default).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_provider import MarketDataProvider
from .watchlist import load_watchlist


def universe_tickers(
    holdings_tickers: list[str],
    watchlist_path: str | Path = "data/watchlist_agentic.csv",
) -> list[str]:
    """Union of currently-held names and thesis watchlist (excluding CASH and unknowns)."""
    try:
        wl = load_watchlist(watchlist_path)["ticker"].tolist()
    except Exception:
        wl = []
    seen: dict[str, None] = {}
    for t in list(holdings_tickers) + wl:
        if t and t != "CASH":
            seen.setdefault(t, None)
    return list(seen)


def _resample(prices: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "daily":
        return prices
    if granularity == "monthly":
        return prices.resample("ME").last()    # month-end
    if granularity == "yearly":
        return prices.resample("YE").last()    # year-end
    raise ValueError(f"granularity must be daily|monthly|yearly, got {granularity!r}")


def normalized_history(
    provider: MarketDataProvider,
    tickers: list[str],
    period: str = "5y",
    granularity: str = "monthly",
    benchmark: str | None = "SPY",
) -> pd.DataFrame:
    """Each column = ticker, values = price normalized so the first non-NaN equals 100.

    Returns an empty DataFrame if no history is available. Includes the benchmark as a
    column if supplied. Use this directly with st.line_chart().
    """
    if not tickers:
        return pd.DataFrame()

    syms = list(tickers)
    if benchmark and benchmark not in syms:
        syms.append(benchmark)

    raw = provider.history(syms, period=period)
    if raw.empty:
        return pd.DataFrame()

    sampled = _resample(raw, granularity).dropna(how="all")
    base = sampled.bfill().iloc[0]               # first usable price per name
    norm = sampled.divide(base).multiply(100.0)
    return norm.dropna(how="all")


def rank_by_total_return(norm_df: pd.DataFrame) -> pd.DataFrame:
    """Rank tickers by total return over the displayed window (descending)."""
    if norm_df.empty:
        return pd.DataFrame(columns=["ticker", "total_return"])
    last = norm_df.ffill().iloc[-1]
    rows = [(t, float(last[t] / 100.0 - 1)) for t in norm_df.columns
            if pd.notna(last[t])]
    df = pd.DataFrame(rows, columns=["ticker", "total_return"])
    return df.sort_values("total_return", ascending=False).reset_index(drop=True)
