"""Stock directory — names, business summaries, and analyst outlook for all tracked names.

Fetches company profiles via the provider's company_profile() method and merges them
with the Agentic Economy layer map so every name is annotated with its thesis layer.
"""
from __future__ import annotations

import pandas as pd

from .data_provider import MarketDataProvider
from .exposure import LAYER_MAP


_CONSENSUS_LABEL = {
    "strong_buy": "Strong Buy",
    "buy":        "Buy",
    "hold":       "Hold",
    "sell":       "Sell",
    "strong_sell":"Strong Sell",
    "underperform":"Underperform",
    "outperform": "Outperform",
}


def build_directory(
    tickers: list[str],
    provider: MarketDataProvider,
) -> pd.DataFrame:
    """Return a display-ready DataFrame for the stock directory section.

    Columns: Ticker, Layer, Name, Sector, Business, Consensus, # Analysts, Target, Upside.
    """
    profiles = provider.company_profile(tickers)

    rows = []
    for t in tickers:
        if t == "CASH":
            continue
        layer = LAYER_MAP.get(t, "—")
        if t in profiles.index:
            p = profiles.loc[t]
            raw_rec = str(p.get("recommendation") or "").strip()
            consensus = _CONSENSUS_LABEL.get(raw_rec, raw_rec.replace("_", " ").title())
            upside = p.get("upside")
            target = p.get("target_price")
            rows.append({
                "Ticker":      t,
                "Layer":       layer,
                "Name":        p.get("name") or t,
                "Sector":      p.get("sector") or "",
                "Business":    p.get("summary") or "",
                "Consensus":   consensus,
                "# Analysts":  int(p["analysts"]) if pd.notna(p.get("analysts")) else "",
                "Target ($)":  f"${target:.2f}" if pd.notna(target) and target else "",
                "Upside":      f"{upside * 100:+.0f}%" if upside is not None else "",
            })
        else:
            rows.append({
                "Ticker": t, "Layer": layer, "Name": t,
                "Sector": "", "Business": "", "Consensus": "",
                "# Analysts": "", "Target ($)": "", "Upside": "",
            })

    return pd.DataFrame(rows)
