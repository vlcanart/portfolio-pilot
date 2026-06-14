"""Load holdings and value the portfolio against live prices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_provider import MarketDataProvider

# Synthetic ticker for cash / money-market holdings. Priced at $1; excluded from
# price history and optimization.
CASH_TICKER = "CASH"


@dataclass
class Position:
    ticker: str
    shares: float
    cost_basis: float          # per-share cost
    price: float               # current per-share price
    market_value: float
    unrealized_pl: float
    weight: float              # fraction of total portfolio value


@dataclass
class Portfolio:
    positions: list[Position]
    total_value: float
    total_cost: float

    @property
    def total_pl(self) -> float:
        return self.total_value - self.total_cost

    @property
    def total_pl_pct(self) -> float:
        return (self.total_pl / self.total_cost) if self.total_cost else 0.0

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @property
    def weights(self) -> dict[str, float]:
        return {p.ticker: p.weight for p in self.positions}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.__dict__ for p in self.positions])


def load_holdings(csv_path: str | Path) -> pd.DataFrame:
    """Read a holdings CSV with columns: ticker, shares, cost_basis."""
    df = pd.read_csv(csv_path)
    required = {"ticker", "shares", "cost_basis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"holdings file missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df


def build_portfolio(
    holdings: pd.DataFrame, provider: MarketDataProvider
) -> Portfolio:
    """Combine holdings with current prices into a valued Portfolio.

    A row with ticker ``CASH`` is treated specially: priced at $1, so ``shares`` is the
    dollar amount held. Cash counts toward total value and allocation but is excluded
    from price history and optimization (see CASH_TICKER).
    """
    tickers = [t for t in holdings["ticker"].tolist() if t != CASH_TICKER]
    prices = provider.latest_prices(tickers)

    rows: list[Position] = []
    total_value = 0.0
    total_cost = 0.0
    for _, h in holdings.iterrows():
        t = h["ticker"]
        if t == CASH_TICKER:
            price = 1.0
        else:
            price = prices.get(t)
        if price is None:
            # Skip unpriceable tickers but keep going.
            continue
        shares = float(h["shares"])
        cost_basis = float(h["cost_basis"])
        mv = shares * price
        rows.append(
            Position(
                ticker=t,
                shares=shares,
                cost_basis=cost_basis,
                price=price,
                market_value=mv,
                unrealized_pl=mv - shares * cost_basis,
                weight=0.0,  # filled in below once we know the total
            )
        )
        total_value += mv
        total_cost += shares * cost_basis

    for p in rows:
        p.weight = (p.market_value / total_value) if total_value else 0.0

    return Portfolio(positions=rows, total_value=total_value, total_cost=total_cost)
