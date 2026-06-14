"""Command-line entry point: track → analyze → (optional) advise.

Usage:
    python -m src.cli --holdings data/holdings.csv
    python -m src.cli --holdings data/holdings.csv --advise --objective max_sharpe
"""
from __future__ import annotations

import argparse

from .advisor import build_briefing_payload, generate_briefing
from .analytics import compute_metrics, portfolio_value_series
from .config import settings
from .data_provider import get_default_provider
from .optimizer import optimize
from .portfolio import build_portfolio, load_holdings


def _fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Pilot")
    parser.add_argument("--holdings", required=True, help="Path to holdings CSV")
    parser.add_argument("--period", default="1y", help="History window (e.g. 6mo, 1y, 2y)")
    parser.add_argument(
        "--objective",
        default="max_sharpe",
        choices=["max_sharpe", "min_volatility"],
        help="Optimizer target",
    )
    parser.add_argument(
        "--advise",
        action="store_true",
        help="Also generate a Claude rebalancing brief (needs ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    provider = get_default_provider()

    # --- Track ---
    holdings = load_holdings(args.holdings)
    portfolio = build_portfolio(holdings, provider)
    if not portfolio.positions:
        print("No priceable positions found. Check your tickers.")
        return

    print(f"\n{'=' * 56}\nPORTFOLIO\n{'=' * 56}")
    print(f"Total value: ${portfolio.total_value:,.2f}")
    print(f"Total cost:  ${portfolio.total_cost:,.2f}")
    print(f"P/L:         ${portfolio.total_pl:,.2f} ({_fmt_pct(portfolio.total_pl_pct)})")
    print(f"\n{'Ticker':<8}{'Weight':>10}{'Value':>14}{'Unreal. P/L':>16}")
    for p in sorted(portfolio.positions, key=lambda x: -x.weight):
        print(f"{p.ticker:<8}{_fmt_pct(p.weight):>10}{p.market_value:>14,.2f}{p.unrealized_pl:>16,.2f}")

    # --- Analyze ---
    series = portfolio_value_series(portfolio, provider, period=args.period)
    metrics = compute_metrics(series, period=args.period)
    if metrics:
        print(f"\n{'=' * 56}\nMETRICS ({metrics.period})\n{'=' * 56}")
        print(f"Total return:        {_fmt_pct(metrics.total_return)}")
        print(f"Annualized return:   {_fmt_pct(metrics.annualized_return)}")
        print(f"Annualized vol:      {_fmt_pct(metrics.annualized_volatility)}")
        print(f"Sharpe (rf={settings.risk_free_rate:.1%}):    {metrics.sharpe:.2f}")
        print(f"Max drawdown:        {_fmt_pct(metrics.max_drawdown)}")

    # --- Recommend (target allocation) ---
    target = optimize(portfolio.tickers, provider, objective=args.objective)
    if target:
        print(f"\n{'=' * 56}\nTARGET ALLOCATION ({target.objective})\n{'=' * 56}")
        for t, w in sorted(target.weights.items(), key=lambda kv: -kv[1]):
            print(f"{t:<8}{_fmt_pct(w):>10}")
        print(f"\nExpected return: {_fmt_pct(target.expected_annual_return)}  "
              f"Vol: {_fmt_pct(target.annual_volatility)}  Sharpe: {target.sharpe:.2f}")

    # --- Advise (Claude) ---
    if args.advise:
        payload = build_briefing_payload(portfolio, metrics, target)
        print(f"\n{'=' * 56}\nAI REBALANCING BRIEF\n{'=' * 56}")
        try:
            print(generate_briefing(payload))
        except RuntimeError as e:
            print(f"(skipped) {e}")


if __name__ == "__main__":
    main()
