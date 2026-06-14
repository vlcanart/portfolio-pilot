"""Target-allocation engine built on PyPortfolioOpt.

Produces a recommended set of weights (max Sharpe or min volatility) from historical
returns. This is the deterministic 'recommend' core — the advisor explains its output.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_provider import MarketDataProvider


@dataclass
class OptimizationResult:
    weights: dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    sharpe: float
    objective: str


def optimize(
    tickers: list[str],
    provider: MarketDataProvider,
    objective: str = "max_sharpe",
    period: str = "2y",
    max_weight: float = 0.30,
) -> OptimizationResult | None:
    """Compute target weights. objective: 'max_sharpe' or 'min_volatility'.

    ``max_weight`` caps any single position (default 30%) so the optimizer produces a
    diversified target instead of dumping everything into the highest-Sharpe name.
    Cash (CASH_TICKER) is excluded — it has no price history.
    """
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    tickers = [t for t in tickers if t != "CASH"]
    prices = provider.history(tickers, period=period)
    # Need at least two assets with usable history for a covariance matrix.
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[1] < 2:
        return None

    # A cap below 1/N is infeasible (weights must sum to 1); floor it.
    n = prices.shape[1]
    bound = max(max_weight, 1.0 / n)

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, bound))
    if objective == "min_volatility":
        ef.min_volatility()
    else:
        objective = "max_sharpe"
        ef.max_sharpe()

    cleaned = ef.clean_weights()
    exp_ret, vol, sharpe = ef.portfolio_performance()

    return OptimizationResult(
        weights={k: float(v) for k, v in cleaned.items() if v > 0},
        expected_annual_return=float(exp_ret),
        annual_volatility=float(vol),
        sharpe=float(sharpe),
        objective=objective,
    )
