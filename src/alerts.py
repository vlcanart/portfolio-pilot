"""Alert engine — turn the monitoring signals into prioritized, actionable flags.

Checks concentration, layer drift vs thesis target, portfolio drawdown, underwater
positions, idle cash, and single-name pullbacks. Returns a severity-sorted list.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .exposure import ExposureReport, analyze_exposure
from .portfolio import Portfolio

# Default thresholds (override per call if desired).
THRESHOLDS = {
    "concentration": 0.25,    # single-name weight
    "layer_warn": 0.10,       # |gap| vs target for a WARN
    "layer_high": 0.15,       # |gap| vs target for a HIGH
    "drawdown_warn": -0.10,   # portfolio below recent peak
    "drawdown_high": -0.20,
    "position_loss": -0.20,   # position below cost basis
    "position_from_high": -0.25,  # position off its 1y high
    "cash_idle": 0.10,        # cash weight considered idle
}

_SEV_ORDER = {"HIGH": 0, "WARN": 1, "INFO": 2}


@dataclass
class Alert:
    severity: str   # HIGH | WARN | INFO
    category: str
    message: str


def check_alerts(
    portfolio: Portfolio,
    exposure: ExposureReport | None = None,
    value_series: pd.Series | None = None,
    holdings_history: pd.DataFrame | None = None,
    th: dict | None = None,
) -> list[Alert]:
    th = th or THRESHOLDS
    out: list[Alert] = []

    # 1. Single-name concentration
    top = max(portfolio.positions, key=lambda p: p.weight)
    if top.weight >= th["concentration"]:
        out.append(Alert("HIGH", "concentration",
                         f"{top.ticker} is {top.weight * 100:.0f}% of the account "
                         f"(> {th['concentration'] * 100:.0f}% limit)."))

    # 2. Layer drift vs thesis target
    exp = exposure or analyze_exposure(portfolio)
    for layer, row in exp.by_layer.iterrows():
        gap = row["gap"]
        if abs(gap) >= th["layer_high"]:
            sev = "HIGH"
        elif abs(gap) >= th["layer_warn"]:
            sev = "WARN"
        else:
            continue
        direction = "overweight" if gap > 0 else "underweight"
        out.append(Alert(sev, "drift",
                         f"{layer} {direction} by {abs(gap) * 100:.0f}pp vs thesis target."))

    # 3. Portfolio drawdown from recent peak
    if value_series is not None and len(value_series) > 1:
        cur_dd = float(value_series.iloc[-1] / value_series.cummax().iloc[-1] - 1)
        if cur_dd <= th["drawdown_high"]:
            out.append(Alert("HIGH", "drawdown",
                             f"Portfolio is {cur_dd * 100:.0f}% below its recent peak."))
        elif cur_dd <= th["drawdown_warn"]:
            out.append(Alert("WARN", "drawdown",
                             f"Portfolio is {cur_dd * 100:.0f}% below its recent peak."))

    # 4. Positions underwater vs cost
    for p in portfolio.positions:
        if p.ticker == "CASH" or p.cost_basis <= 0:
            continue
        pl = p.unrealized_pl / (p.shares * p.cost_basis)
        if pl <= th["position_loss"]:
            out.append(Alert("WARN", "position",
                             f"{p.ticker} is {pl * 100:.0f}% below cost basis."))

    # 5. Positions far off their 1y high (deterioration or add opportunity)
    if holdings_history is not None and not holdings_history.empty:
        for p in portfolio.positions:
            if p.ticker not in holdings_history.columns:
                continue
            s = holdings_history[p.ticker].dropna()
            if len(s) > 30:
                from_high = float(s.iloc[-1] / s.max() - 1)
                if from_high <= th["position_from_high"]:
                    out.append(Alert("INFO", "pullback",
                                     f"{p.ticker} is {from_high * 100:.0f}% off its 1y high."))

    # 6. Idle cash
    cash_w = portfolio.weights.get("CASH", 0.0)
    if cash_w >= th["cash_idle"]:
        out.append(Alert("INFO", "cash",
                         f"{cash_w * 100:.0f}% in cash — idle capital to deploy."))

    return sorted(out, key=lambda a: _SEV_ORDER[a.severity])
