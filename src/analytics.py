"""Performance and risk analytics for a portfolio's historical value series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import settings
from .data_provider import MarketDataProvider
from .portfolio import Portfolio


@dataclass
class Metrics:
    total_return: float        # over the lookback window
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    period: str


def portfolio_value_series(
    portfolio: Portfolio, provider: MarketDataProvider, period: str = "1y"
) -> pd.Series:
    """Reconstruct the portfolio's market value over time using current share counts.

    Note: this uses *today's* share counts back over history (it does not model past
    buys/sells). It answers "how would my current basket have behaved?" — a clean,
    honest approximation for a point-in-time holdings file.
    """
    hist = provider.history(portfolio.tickers, period=period)
    if hist.empty:
        return pd.Series(dtype=float)
    shares = {p.ticker: p.shares for p in portfolio.positions}
    cols = [t for t in hist.columns if t in shares]
    valued = hist[cols].mul(pd.Series({t: shares[t] for t in cols}), axis=1)
    return valued.sum(axis=1).dropna()


def compute_metrics(value_series: pd.Series, period: str = "1y") -> Metrics | None:
    if value_series is None or len(value_series) < 2:
        return None

    daily_ret = value_series.pct_change().dropna()
    if daily_ret.empty:
        return None

    total_return = float(value_series.iloc[-1] / value_series.iloc[0] - 1)

    td = settings.trading_days
    ann_return = float((1 + daily_ret.mean()) ** td - 1)
    ann_vol = float(daily_ret.std() * np.sqrt(td))

    rf = settings.risk_free_rate
    sharpe = float((ann_return - rf) / ann_vol) if ann_vol else 0.0

    running_max = value_series.cummax()
    drawdown = value_series / running_max - 1
    max_dd = float(drawdown.min())

    return Metrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        period=period,
    )
