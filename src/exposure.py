"""Theme / layer exposure analysis and thesis-gap reporting.

Maps holdings onto the user's 'Agentic Economy' framework layers, measures
concentration, and compares current layer weights to a thesis-aligned target so
the gap between research conviction and actual capital is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .portfolio import Portfolio

# Which framework layer each ticker belongs to. Extend as holdings/watchlist change.
LAYER_MAP: dict[str, str] = {
    # Compute & hardware (incl. connectivity / optical)
    "NVDA": "Compute & Semis", "AVGO": "Compute & Semis", "TSM": "Compute & Semis",
    "MRVL": "Compute & Semis", "MU": "Compute & Semis", "AMD": "Compute & Semis",
    "ALAB": "Compute & Semis", "CRDO": "Compute & Semis", "COHR": "Compute & Semis",
    "LITE": "Compute & Semis", "DELL": "Compute & Semis", "SMH": "Compute & Semis",
    "GLW": "Compute & Semis",
    # Software & platforms
    "MSFT": "Software & Platforms", "GOOGL": "Software & Platforms",
    "GOOG": "Software & Platforms", "META": "Software & Platforms",
    "AMZN": "Software & Platforms", "ORCL": "Software & Platforms",
    "PLTR": "Software & Platforms", "NOW": "Software & Platforms",
    "CRM": "Software & Platforms", "SNOW": "Software & Platforms",
    "NET": "Software & Platforms", "CRWD": "Software & Platforms",
    "PANW": "Software & Platforms",
    # AI power & infrastructure
    "CEG": "AI Power & Infra", "VST": "AI Power & Infra", "GEV": "AI Power & Infra",
    "VRT": "AI Power & Infra", "ANET": "AI Power & Infra", "ETN": "AI Power & Infra",
    "CCJ": "AI Power & Infra", "ICLN": "AI Power & Infra",
    # Robotics & automation
    "TSLA": "Robotics", "SYM": "Robotics", "TER": "Robotics", "ISRG": "Robotics",
    # Materials & critical minerals
    "COPX": "Materials & Critical Minerals", "GNR": "Materials & Critical Minerals",
    "MP": "Materials & Critical Minerals", "ALB": "Materials & Critical Minerals",
    # Frontier / quantum (speculative: quantum + small-modular nuclear)
    "IONQ": "Frontier / Quantum", "RGTI": "Frontier / Quantum",
    "OKLO": "Frontier / Quantum", "SMR": "Frontier / Quantum",
    # Crypto & digital assets
    "BTC": "Crypto", "COIN": "Crypto", "IBIT": "Crypto",
    # Precious-metals hedge
    "GLD": "Precious Metals (hedge)", "PHYS": "Precious Metals (hedge)",
    "PSLV": "Precious Metals (hedge)",
    # Other buckets
    "VEA": "Diversified / Intl", "CASH": "Cash",
}

# Illustrative thesis-aligned target weights by layer (editable). Sums to 1.0.
# Growth-tilted 'own your thesis' stance: a small speculative Frontier sleeve, a modest
# crypto allocation, and a reduced real-asset / metals hedge.
TARGET_LAYER_WEIGHTS: dict[str, float] = {
    "Compute & Semis": 0.22,
    "Software & Platforms": 0.20,
    "AI Power & Infra": 0.16,
    "Robotics": 0.10,
    "Materials & Critical Minerals": 0.10,
    "Frontier / Quantum": 0.05,
    "Precious Metals (hedge)": 0.08,
    "Diversified / Intl": 0.04,
    "Crypto": 0.03,
    "Cash": 0.02,
}


@dataclass
class ExposureReport:
    by_layer: pd.DataFrame          # current vs target weight + gap per layer
    top_position: tuple[str, float]  # largest single name and its weight
    concentration_flag: bool         # any single name > 25%


def _layer_of(ticker: str) -> str:
    return LAYER_MAP.get(ticker, "Unclassified")


def analyze_exposure(
    portfolio: Portfolio,
    target: dict[str, float] | None = None,
) -> ExposureReport:
    target = target or TARGET_LAYER_WEIGHTS

    rows = [(p.ticker, _layer_of(p.ticker), p.weight) for p in portfolio.positions]
    df = pd.DataFrame(rows, columns=["ticker", "layer", "weight"])

    current = df.groupby("layer")["weight"].sum()
    layers = sorted(set(current.index) | set(target.keys()))
    out = pd.DataFrame(index=layers)
    out["current"] = current.reindex(layers).fillna(0.0)
    out["target"] = pd.Series(target).reindex(layers).fillna(0.0)
    out["gap"] = out["current"] - out["target"]  # + = overweight, - = underweight
    out = out.sort_values("current", ascending=False)

    top = max(portfolio.positions, key=lambda p: p.weight)
    return ExposureReport(
        by_layer=out,
        top_position=(top.ticker, top.weight),
        concentration_flag=top.weight > 0.25,
    )
