"""Simplified Brinson-Fachler performance attribution.

Decomposes active return (portfolio vs SPY) into three effects:

  Allocation effect  — did over/underweighting thesis layers vs target add value?
  Selection effect   — did individual stock picks within each layer beat the layer?
  Interaction effect — joint effect of allocation + selection decisions.

Benchmark allocation = TARGET_LAYER_WEIGHTS (the thesis target).
Layer benchmark return = equal-weight return of all names tracked in that layer.
Portfolio layer return = weight-averaged return of held names in that layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_provider import MarketDataProvider
from .exposure import LAYER_MAP, TARGET_LAYER_WEIGHTS
from .portfolio import Portfolio


@dataclass
class BrinsonResult:
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    benchmark_return: float
    portfolio_return: float
    by_layer: pd.DataFrame


def compute_brinson(
    portfolio: Portfolio,
    provider: MarketDataProvider,
    period: str = "1y",
    benchmark: str = "SPY",
) -> BrinsonResult | None:
    held = [t for t in portfolio.tickers if t != "CASH"]
    all_layer_tickers = list(LAYER_MAP.keys())
    syms = list({*held, *all_layer_tickers, benchmark})

    raw = provider.history(syms, period=period)
    if raw.empty:
        return None

    def _period_return(col: pd.Series) -> float:
        c = col.dropna()
        return float(c.iloc[-1] / c.iloc[0] - 1) if len(c) >= 2 else float("nan")

    returns = {t: _period_return(raw[t]) for t in raw.columns}
    spy_ret = returns.get(benchmark, 0.0)

    # Reverse map: layer_name → [tickers tracked in that layer]
    layer_tickers: dict[str, list[str]] = {}
    for ticker, layer in LAYER_MAP.items():
        layer_tickers.setdefault(layer, []).append(ticker)

    pw = portfolio.weights  # {ticker: decimal weight}

    # Portfolio layer weights = sum of position weights in each layer
    port_layer_w: dict[str, float] = {}
    for t, w in pw.items():
        lyr = LAYER_MAP.get(t)
        if lyr:
            port_layer_w[lyr] = port_layer_w.get(lyr, 0.0) + w

    rows = []
    for layer, target_w in TARGET_LAYER_WEIGHTS.items():
        if layer == "Cash":
            continue

        actual_w = port_layer_w.get(layer, 0.0)

        # Layer benchmark return: equal-weight of all tracked names in this layer
        tracked = layer_tickers.get(layer, [])
        layer_rets = [
            returns[t] for t in tracked
            if t in returns and not pd.isna(returns[t])
        ]
        r_bi = float(sum(layer_rets) / len(layer_rets)) if layer_rets else spy_ret

        # Layer portfolio return: weight-averaged return of held names in this layer
        held_in = [(t, pw[t]) for t in tracked if t in pw and pw[t] > 0]
        if held_in:
            layer_w_sum = sum(w for _, w in held_in)
            r_pi = (
                sum(returns.get(t, r_bi) * w for t, w in held_in) / layer_w_sum
                if layer_w_sum > 0 else r_bi
            )
        else:
            r_pi = r_bi  # no position → assume we captured the layer benchmark

        alloc = (actual_w - target_w) * (r_bi - spy_ret)
        sel = target_w * (r_pi - r_bi)
        inter = (actual_w - target_w) * (r_pi - r_bi)

        rows.append({
            "Layer": layer,
            "Port. weight": actual_w,
            "Target weight": target_w,
            "Active weight": actual_w - target_w,
            "Port. layer rtn": r_pi,
            "Bmk. layer rtn": r_bi,
            "Allocation": alloc,
            "Selection": sel,
            "Interaction": inter,
            "Total": alloc + sel + inter,
        })

    df = pd.DataFrame(rows)

    port_return = sum(
        pw.get(t, 0.0) * returns.get(t, 0.0)
        for t in held
        if t in returns and not pd.isna(returns.get(t, float("nan")))
    )

    return BrinsonResult(
        allocation_effect=float(df["Allocation"].sum()),
        selection_effect=float(df["Selection"].sum()),
        interaction_effect=float(df["Interaction"].sum()),
        total_active_return=float(df["Total"].sum()),
        benchmark_return=spy_ret,
        portfolio_return=port_return,
        by_layer=df,
    )
