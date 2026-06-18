"""Daily market conditions — indices, sectors, macro, and 24h news digest."""
from __future__ import annotations

import time

import pandas as pd

_INDICES: dict[str, str] = {
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "DIA":  "Dow Jones",
    "IWM":  "Russell 2000",
}

# VIX fetched separately to avoid column-name edge cases with the ^ prefix
_VIX_TICKER = "^VIX"

_SECTORS: dict[str, str] = {
    "XLK": "Tech",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLC": "Comm Svcs",
    "XLI": "Industrials",
    "XLY": "Cons Disc",
    "XLP": "Cons Staples",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

_MACRO: dict[str, str] = {
    "TLT": "Bonds (20yr)",
    "GLD": "Gold",
    "USO": "Oil",
    "UUP": "USD",
}


def _batch_pct_change(tickers: list[str]) -> dict[str, float | None]:
    """% change from prior close to latest close for each ticker (batch download)."""
    import yfinance as yf
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            return {t: None for t in tickers}
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        close = close.dropna(how="all").ffill()
        out: dict[str, float | None] = {}
        for t in tickers:
            if t in close.columns:
                col = close[t].dropna()
                out[t] = float(col.iloc[-1] / col.iloc[-2] - 1) if len(col) >= 2 else None
            else:
                out[t] = None
        return out
    except Exception:
        return {t: None for t in tickers}


def _vix_level() -> tuple[float | None, float | None]:
    """Return (current VIX level, % change). Level matters more than % change for VIX."""
    import yfinance as yf
    try:
        hist = yf.Ticker(_VIX_TICKER).history(period="5d")
        if hist is None or hist.empty or len(hist) < 2:
            return None, None
        level = float(hist["Close"].iloc[-1])
        chg = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1)
        return level, chg
    except Exception:
        return None, None


def daily_market_snapshot() -> dict:
    """Fetch and return all daily market data.

    Returns a dict with keys:
      indices  — {label: pct_change}
      vix      — (level, pct_change)
      sectors  — {label: pct_change}
      macro    — {label: pct_change}
    """
    idx_chg = _batch_pct_change(list(_INDICES.keys()))
    sec_chg = _batch_pct_change(list(_SECTORS.keys()))
    mac_chg = _batch_pct_change(list(_MACRO.keys()))
    vix_level, vix_chg = _vix_level()

    return {
        "indices": {label: idx_chg.get(t) for t, label in _INDICES.items()},
        "vix":     (vix_level, vix_chg),
        "sectors": {label: sec_chg.get(t) for t, label in _SECTORS.items()},
        "macro":   {label: mac_chg.get(t) for t, label in _MACRO.items()},
    }


def recent_news(tickers: list[str], max_articles: int = 20) -> list[dict]:
    """Return recent news articles for the given tickers, sorted newest first.

    Each article: {title, publisher, link, time (unix), ticker}.
    """
    import yfinance as yf
    articles: list[dict] = []
    seen: set[str] = set()
    for t in tickers:
        try:
            news = yf.Ticker(t).news or []
            for item in news[:5]:
                title = item.get("title", "").strip()
                if title and title not in seen:
                    seen.add(title)
                    articles.append({
                        "title":     title,
                        "publisher": item.get("publisher", ""),
                        "link":      item.get("link", ""),
                        "time":      item.get("providerPublishTime", 0),
                        "ticker":    t,
                    })
        except Exception:
            pass
        time.sleep(0.08)
    articles.sort(key=lambda x: x["time"], reverse=True)
    return articles[:max_articles]
