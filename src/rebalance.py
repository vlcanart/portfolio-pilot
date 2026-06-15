"""Rebalance planner — turn the thesis-gap into concrete buy/sell orders.

Works at the layer level: trims overweight layers proportionally across the names you
hold, deploys excess cash, and funds underweight layers with a curated candidate set
from the watchlist. Because this is a tax-advantaged (IRA) account, rebalancing has no
capital-gains cost, so the plan optimizes purely for target alignment.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .data_provider import MarketDataProvider
from .exposure import TARGET_LAYER_WEIGHTS, _layer_of
from .portfolio import Portfolio

# Curated names used to fund each underweight layer (skips ones you already hold).
NEW_CANDIDATES: dict[str, list[str]] = {
    "Compute & Semis": ["SMH", "TSM", "AVGO"],
    "Software & Platforms": ["MSFT", "GOOGL"],
    "AI Power & Infra": ["CEG", "VST", "VRT"],
}


@dataclass
class Order:
    action: str    # SELL | BUY | DEPLOY_CASH
    ticker: str
    shares: float
    dollars: float
    layer: str
    reason: str


def plan_rebalance(
    portfolio: Portfolio,
    provider: MarketDataProvider,
    target: dict[str, float] | None = None,
    candidates: dict[str, list[str]] | None = None,
    drift_tolerance: float = 0.01,
) -> list[Order]:
    target = target or TARGET_LAYER_WEIGHTS
    candidates = candidates or NEW_CANDIDATES
    total = portfolio.total_value

    by_layer: dict[str, list] = defaultdict(list)
    for p in portfolio.positions:
        by_layer[_layer_of(p.ticker)].append(p)
    current_dollars = {L: sum(p.market_value for p in ps) for L, ps in by_layer.items()}

    held = {p.ticker for p in portfolio.positions}
    held_price = {p.ticker: p.price for p in portfolio.positions}
    new_tickers = [t for names in candidates.values() for t in names if t not in held]
    new_prices = provider.latest_prices(new_tickers)

    orders: list[Order] = []
    for layer in sorted(set(current_dollars) | set(target)):
        tgt = target.get(layer, 0.0) * total
        cur = current_dollars.get(layer, 0.0)
        delta = tgt - cur
        if abs(delta) < drift_tolerance * total:
            continue

        if layer == "Cash":
            if delta < 0:  # too much cash → deploy it
                orders.append(Order("DEPLOY_CASH", "CASH", abs(delta), abs(delta),
                                    layer, "Deploy idle cash into target layers"))
            continue

        if delta < 0:  # overweight → trim held names proportionally
            names = [p for p in by_layer[layer] if p.ticker != "CASH"]
            lsum = sum(p.market_value for p in names) or 1.0
            for p in names:
                d = abs(delta) * (p.market_value / lsum)
                orders.append(Order("SELL", p.ticker, d / p.price, d, layer,
                                    f"Trim — {layer} overweight vs thesis"))
        else:  # underweight → buy curated candidates evenly
            cands = candidates.get(layer, [])
            if not cands:
                orders.append(Order("BUY", "(no candidate)", 0.0, delta, layer,
                                    f"{layer} underweight — pick names to fund"))
                continue
            each = delta / len(cands)
            for c in cands:
                price = held_price.get(c) or new_prices.get(c)
                if not price:
                    continue
                orders.append(Order("BUY", c, each / price, each, layer,
                                    f"Add to underweight {layer}"))
    return orders


def summarize(orders: list[Order]) -> dict[str, float]:
    sells = sum(o.dollars for o in orders if o.action == "SELL")
    buys = sum(o.dollars for o in orders if o.action == "BUY")
    cash = sum(o.dollars for o in orders if o.action == "DEPLOY_CASH")
    return {"sell_total": sells, "buy_total": buys, "cash_deployed": cash,
            "net": sells + cash - buys}
