"""Institutional-grade risk metrics beyond Sharpe + max-drawdown. v2

Adds Sortino, Calmar, Beta vs SPY, Information Ratio, Jensen's Alpha,
rolling Sharpe (30/90/365d), historical VaR / CVaR, MTD return, and a
multi-scenario macro stress table. All derived from the portfolio value
series — no new data dependencies beyond the SPY benchmark history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import settings
from .data_provider import MarketDataProvider


@dataclass
class StressScenario:
    name: str
    description: str
    spy_shock: float           # assumed SPY move (e.g. -0.20)
    portfolio_pct: float       # estimated portfolio move (beta × shock)
    portfolio_dollars: float   # dollar P&L impact


@dataclass
class AdvancedMetrics:
    sortino: float
    calmar: float
    beta_spy: float
    info_ratio: float
    jensens_alpha: float
    rolling_sharpe: dict[str, float]   # {"30d": ..., "90d": ..., "365d": ...}
    var_99_1d: float                   # historical 1-day VaR (loss as negative)
    cvar_99_1d: float                  # mean loss beyond VaR
    mtd_return: float
    stress_spy_minus_20: float         # legacy single stress
    stress_scenarios: list[StressScenario] = field(default_factory=list)


# Scenario definitions: (display_name, description, assumed_spy_shock)
_SCENARIO_DEFS: list[tuple[str, str, float]] = [
    (
        "Rate Shock +100bps",
        "Fed surprise hike; tech multiple compression",
        -0.08,
    ),
    (
        "Stagflation",
        "Growth rotation; defensives outperform AI names",
        -0.15,
    ),
    (
        "Deep Recession",
        "Broad market capitulation; risk-off",
        -0.35,
    ),
    (
        "AI Sector Unwind",
        "AI valuation de-rate; high-beta tech leads down",
        -0.25,
    ),
]
# AI-heavy portfolio carries extra sensitivity in the AI-specific scenario
_AI_BETA_MULTIPLIER = 1.5


def compute_advanced(
    value_series: pd.Series,
    provider: MarketDataProvider,
    benchmark: str = "SPY",
    portfolio_value: float = 0.0,
) -> AdvancedMetrics | None:
    if value_series is None or len(value_series) < 30:
        return None

    daily = value_series.pct_change().dropna()
    if daily.empty:
        return None

    td = settings.trading_days
    rf = settings.risk_free_rate

    # --- Sortino ---
    downside = daily[daily < 0]
    downside_vol = float(downside.std() * np.sqrt(td)) if len(downside) > 1 else 0.0
    ann_return = float(daily.mean() * td)
    sortino = float((ann_return - rf) / downside_vol) if downside_vol > 0 else 0.0

    # --- Calmar ---
    cumulative = (1 + daily).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    # --- Benchmark-linked: Beta, Info Ratio, Jensen's Alpha ---
    bench_hist = provider.history([benchmark], period="2y")
    beta = info = jensens_alpha = 0.0
    if not bench_hist.empty and benchmark in bench_hist.columns:
        bench_daily = bench_hist[benchmark].pct_change().dropna()
        aligned = pd.concat([daily, bench_daily], axis=1, join="inner").dropna()
        aligned.columns = ["p", "b"]
        if len(aligned) > 20 and aligned["b"].var() > 0:
            beta = float(aligned["p"].cov(aligned["b"]) / aligned["b"].var())
            active = aligned["p"] - aligned["b"]
            te = float(active.std() * np.sqrt(td))
            info = float((active.mean() * td) / te) if te > 0 else 0.0
            bench_ann = float(aligned["b"].mean() * td)
            jensens_alpha = ann_return - (rf + beta * (bench_ann - rf))

    # --- Rolling Sharpe ---
    def _rs(window: int) -> float:
        if len(daily) < window:
            return 0.0
        w = daily.tail(window)
        sd = float(w.std() * np.sqrt(td))
        return float((w.mean() * td - rf) / sd) if sd > 0 else 0.0
    rolling = {"30d": _rs(30), "90d": _rs(90), "365d": _rs(252)}

    # --- Historical VaR / CVaR (99%, 1-day) ---
    var_99 = float(np.percentile(daily, 1))
    cvar_99 = float(daily[daily <= var_99].mean()) if (daily <= var_99).any() else var_99

    # --- MTD return ---
    month_start = pd.Timestamp(value_series.index[-1]).normalize().replace(day=1)
    mtd_slice = value_series[value_series.index >= month_start]
    mtd = float(mtd_slice.iloc[-1] / mtd_slice.iloc[0] - 1) if len(mtd_slice) > 1 else 0.0

    # --- Legacy single stress (SPY -20%) ---
    stress_legacy = float(beta * -0.20)

    # --- Multi-scenario stress table ---
    pv = portfolio_value
    scenarios: list[StressScenario] = []
    for name, desc, spy_shock in _SCENARIO_DEFS:
        b = beta * _AI_BETA_MULTIPLIER if "AI" in name else beta
        pct = b * spy_shock
        scenarios.append(StressScenario(
            name=name,
            description=desc,
            spy_shock=spy_shock,
            portfolio_pct=pct,
            portfolio_dollars=pv * pct,
        ))

    return AdvancedMetrics(
        sortino=sortino,
        calmar=calmar,
        beta_spy=beta,
        info_ratio=info,
        jensens_alpha=jensens_alpha,
        rolling_sharpe=rolling,
        var_99_1d=var_99,
        cvar_99_1d=cvar_99,
        mtd_return=mtd,
        stress_spy_minus_20=stress_legacy,
        stress_scenarios=scenarios,
    )
