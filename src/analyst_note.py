"""Recurring AI analyst note — Claude synthesizes the full monitoring picture.

Assembles portfolio state, alerts, thesis-gap, watchlist pullbacks, and the change since
the last snapshot into one payload, then has Claude write a dated, prioritized analyst
note. As with the advisor, Claude reasons only over numbers the app computed.
"""
from __future__ import annotations

import datetime as _dt
import json

import pandas as pd

from .alerts import Alert
from .analytics import Metrics
from .config import settings
from .exposure import ExposureReport
from .portfolio import Portfolio

SYSTEM_PROMPT = (
    "You are a senior portfolio analyst writing a recurring monitoring note for a single "
    "client's retirement (IRA) account. You are given numbers the application computed: "
    "current positions and weights, risk metrics, thesis-vs-actual layer gaps, fired alerts, "
    "watchlist names in pullback, and the change since the last snapshot.\n\n"
    "Write a concise dated note with these sections:\n"
    "1. Headline — one line on the account's state and the single most important action.\n"
    "2. What changed — since the last snapshot (skip if no prior snapshot).\n"
    "3. Risks — translate the HIGH/WARN alerts into plain language; lead with concentration.\n"
    "4. Opportunities — watchlist pullbacks that fit underweight thesis layers.\n"
    "5. Actions — a short prioritized list (trim/add/hold), naming tickers.\n\n"
    "Rules: work only from the provided figures; never invent prices or returns. Remember "
    "this is a tax-free account, so rebalancing has no tax cost. Decision support, not "
    "financial advice — one short caveat at the end. Be skimmable: short sections, bullets."
)


def build_note_payload(
    portfolio: Portfolio,
    metrics: Metrics | None,
    exposure: ExposureReport | None,
    alerts: list[Alert],
    watchlist_screen: pd.DataFrame | None = None,
    change: dict | None = None,
    today: str | None = None,
) -> dict:
    gaps = {}
    if exposure is not None:
        gaps = {layer: round(row["gap"], 4) for layer, row in exposure.by_layer.iterrows()
                if abs(row["gap"]) >= 0.05}

    pullbacks = []
    if watchlist_screen is not None and not watchlist_screen.empty and "signal" in watchlist_screen:
        pb = watchlist_screen[watchlist_screen["signal"] == "pullback"]
        pullbacks = [
            {"ticker": r["ticker"], "layer": r["layer"],
             "off_high": round(float(r["from_1y_high"]), 3),
             "mom_6m": round(float(r["6M"]), 3) if pd.notna(r["6M"]) else None}
            for _, r in pb.iterrows()
        ]

    return {
        "date": today or _dt.date.today().isoformat(),
        "portfolio": {
            "total_value": round(portfolio.total_value, 2),
            "total_pl_pct": round(portfolio.total_pl_pct, 4),
            "weights": {t: round(w, 4) for t, w in portfolio.weights.items()},
        },
        "metrics": None if metrics is None else {
            "annualized_return": round(metrics.annualized_return, 4),
            "annualized_volatility": round(metrics.annualized_volatility, 4),
            "sharpe": round(metrics.sharpe, 3),
            "max_drawdown": round(metrics.max_drawdown, 4),
        },
        "thesis_gaps": gaps,
        "alerts": [{"severity": a.severity, "category": a.category, "message": a.message}
                   for a in alerts],
        "watchlist_pullbacks": pullbacks,
        "change_since_last": change,
    }


def generate_note(payload: dict) -> str:
    """Call Claude to write the analyst note. Requires ANTHROPIC_API_KEY."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env / secrets to enable the analyst note."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_msg = (
        "Write today's monitoring note from these computed figures:\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )
    response = client.messages.create(
        model=settings.advisor_model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.advisor_effort},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
