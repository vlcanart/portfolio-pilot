"""Watchlist monitor — live price, momentum, and entry signals for thesis names.

Rebuilds the 'Agentic Economy' screen as a live, ranked table instead of a stale
spreadsheet: trailing momentum across windows plus a pullback signal based on
distance from the trailing-1y high.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_provider import MarketDataProvider

DEFAULT_WATCHLIST = "data/watchlist_agentic.csv"


def load_watchlist(path: str | Path = DEFAULT_WATCHLIST) -> pd.DataFrame:
    return pd.read_csv(path)


def screen(
    provider: MarketDataProvider,
    watchlist: pd.DataFrame,
    held: set[str] | None = None,
    pullback_threshold: float = -0.15,
    with_fundamentals: bool = False,
) -> pd.DataFrame:
    """Return a ranked screen with momentum + a pullback entry flag.

    `pullback_threshold` (e.g. -0.15) flags names trading >=15% below their 1y high.
    `with_fundamentals=True` merges P/E, revenue growth, and margin (slower — one call
    per name).
    """
    held = held or set()
    tickers = watchlist["ticker"].tolist()
    hist = provider.history(tickers, period="1y")

    def _ret(s: pd.Series, days: int) -> float:
        s = s.dropna()
        if len(s) < days + 1:
            return np.nan
        return float(s.iloc[-1] / s.iloc[-days - 1] - 1)

    rows = []
    for _, r in watchlist.iterrows():
        t = r["ticker"]
        if t not in hist.columns:
            continue
        s = hist[t].dropna()
        if len(s) < 30:
            continue
        price = float(s.iloc[-1])
        from_high = price / float(s.max()) - 1
        rows.append({
            "ticker": t,
            "layer": r["layer"],
            "price": round(price, 2),
            "1M": _ret(s, 21),
            "3M": _ret(s, 63),
            "6M": _ret(s, 126),
            "from_1y_high": from_high,
            "held": "●" if t in held else "",
            "signal": "pullback" if from_high <= pullback_threshold else "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if with_fundamentals:
        fund = provider.fundamentals(df["ticker"].tolist())
        if not fund.empty:
            df = df.merge(
                fund[["pe", "rev_growth", "profit_margin"]],
                left_on="ticker", right_index=True, how="left",
            )

    return df.sort_values("6M", ascending=False, na_position="last").reset_index(drop=True)
