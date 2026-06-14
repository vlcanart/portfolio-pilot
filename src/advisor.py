"""AI advisor: turns computed metrics + optimizer output into a plain-English brief.

Claude never sees raw market data or makes up numbers — it reasons over the figures
*we* computed (current weights, metrics, target weights) and explains the gap. This keeps
the quantitative work deterministic and uses the model only for synthesis and explanation.
"""
from __future__ import annotations

import json

from .analytics import Metrics
from .config import settings
from .optimizer import OptimizationResult
from .portfolio import Portfolio

SYSTEM_PROMPT = (
    "You are a portfolio analysis assistant embedded in a personal investing tool. "
    "You are given numbers the application already computed: current holdings and weights, "
    "performance/risk metrics, and a target allocation from a mean-variance optimizer. "
    "Your job is to explain what the numbers mean and propose concrete, prioritized "
    "rebalancing steps to move from current to target weights.\n\n"
    "Rules:\n"
    "- Work only from the figures provided. Never invent prices, returns, or tickers.\n"
    "- Be specific: name tickers and the direction/rough size of each suggested change.\n"
    "- Call out concentration, under-diversification, and notable drawdown/volatility.\n"
    "- This is decision support, not financial advice. Do not guarantee returns. Keep a "
    "brief, plain caveat at the end — one sentence, no boilerplate.\n"
    "- Be concise and skimmable. Use short sections and bullets."
)


def build_briefing_payload(
    portfolio: Portfolio,
    metrics: Metrics | None,
    target: OptimizationResult | None,
) -> dict:
    """Assemble the exact figures handed to the model (also useful for logging/audit)."""
    return {
        "portfolio": {
            "total_value": round(portfolio.total_value, 2),
            "total_cost": round(portfolio.total_cost, 2),
            "total_pl": round(portfolio.total_pl, 2),
            "total_pl_pct": round(portfolio.total_pl_pct, 4),
            "current_weights": {t: round(w, 4) for t, w in portfolio.weights.items()},
            "positions": [
                {
                    "ticker": p.ticker,
                    "weight": round(p.weight, 4),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pl": round(p.unrealized_pl, 2),
                }
                for p in portfolio.positions
            ],
        },
        "metrics": None
        if metrics is None
        else {
            "period": metrics.period,
            "total_return": round(metrics.total_return, 4),
            "annualized_return": round(metrics.annualized_return, 4),
            "annualized_volatility": round(metrics.annualized_volatility, 4),
            "sharpe": round(metrics.sharpe, 3),
            "max_drawdown": round(metrics.max_drawdown, 4),
        },
        "target_allocation": None
        if target is None
        else {
            "objective": target.objective,
            "weights": {t: round(w, 4) for t, w in target.weights.items()},
            "expected_annual_return": round(target.expected_annual_return, 4),
            "annual_volatility": round(target.annual_volatility, 4),
            "sharpe": round(target.sharpe, 3),
        },
    }


def generate_briefing(payload: dict) -> str:
    """Call Claude to produce the rebalancing brief. Requires ANTHROPIC_API_KEY."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env to enable the AI advisor."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    user_msg = (
        "Here are the computed figures for my portfolio as JSON. Give me a rebalancing "
        "brief: a one-line summary, key observations (risk, concentration, performance), "
        "and a prioritized list of suggested moves from current → target weights.\n\n"
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
