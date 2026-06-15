"""Compounding projections — Monte-Carlo the portfolio forward.

Estimates a weighted portfolio's annualized return/volatility from history, then runs a
Monte-Carlo simulation of its value over a horizon (with optional monthly contributions).
Used to compare the current allocation against the post-rebalance (thesis-aligned) one and
quantify what the rotation does to long-run compounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import settings
from .data_provider import MarketDataProvider


def weighted_return_vol(
    weights: dict[str, float],
    provider: MarketDataProvider,
    period: str = "2y",
    return_mode: str = "cma",
) -> tuple[float, float]:
    """Annualized (expected return, volatility) for a weighted basket.

    Volatility is estimated from historical daily returns (reliable). Expected return:
      - "cma" (default): rf + ERP * (portfolio_vol / market_vol). A capital-market
        assumption — avoids extrapolating a short-run bull market forward.
      - "historical": trailing annualized mean (over-optimistic; for an aggressive scenario).
    CASH is treated as risk-free. Tickers without history are dropped.
    """
    rf = settings.risk_free_rate
    td = settings.trading_days
    cash_w = weights.get("CASH", 0.0)
    risky = {t: w for t, w in weights.items() if t != "CASH"}
    if not risky:
        return rf, 0.0

    hist = provider.history(list(risky), period=period).dropna(axis=1, how="all")
    rets = hist.pct_change().dropna()
    cols = [t for t in risky if t in rets.columns]
    if not cols:
        return rf, 0.0

    w = np.array([risky[t] for t in cols])
    port = rets[cols].values @ w           # daily portfolio return (absolute weights)
    ann_vol = float(port.std() * np.sqrt(td))

    if return_mode == "historical":
        ann_return = float(port.mean() * td + cash_w * rf)
    else:  # capital-market assumption — price of risk, not extrapolated history
        ann_return = rf + settings.equity_risk_premium * (ann_vol / settings.market_vol)
    return ann_return, ann_vol


def monte_carlo(
    initial: float,
    ann_return: float,
    ann_vol: float,
    years: int = 10,
    monthly_contribution: float = 0.0,
    n_sims: int = 2000,
    seed: int = 7,
) -> dict:
    """Simulate value paths with monthly normal returns + contributions."""
    rng = np.random.default_rng(seed)
    months = int(years * 12)
    mu_m = (1 + ann_return) ** (1 / 12) - 1
    sig_m = ann_vol / np.sqrt(12)

    shocks = rng.normal(mu_m, sig_m, size=(n_sims, months))
    vals = np.empty((n_sims, months + 1))
    vals[:, 0] = initial
    for m in range(months):
        vals[:, m + 1] = np.maximum(vals[:, m] * (1 + shocks[:, m]) + monthly_contribution, 0.0)

    term = vals[:, -1]
    invested = initial + monthly_contribution * months
    pcts = np.percentile(vals, [10, 50, 90], axis=0)
    path = pd.DataFrame({"p10": pcts[0], "p50": pcts[1], "p90": pcts[2]})
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "invested": invested,
        "terminal_p10": float(np.percentile(term, 10)),
        "terminal_p50": float(np.percentile(term, 50)),
        "terminal_p90": float(np.percentile(term, 90)),
        "median_multiple": float(np.percentile(term, 50) / invested) if invested else 0.0,
        "path": path,
    }


def compare(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    provider: MarketDataProvider,
    initial: float,
    years: int = 10,
    monthly_contribution: float = 0.0,
) -> dict[str, dict]:
    """Run the projection for both allocations and return {'current':..., 'target':...}."""
    cr, cv = weighted_return_vol(current_weights, provider)
    tr, tv = weighted_return_vol(target_weights, provider)
    return {
        "current": monte_carlo(initial, cr, cv, years, monthly_contribution),
        "target": monte_carlo(initial, tr, tv, years, monthly_contribution),
    }
