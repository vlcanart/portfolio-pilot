"""Command-line entry point: track → analyze → (optional) advise.

Usage:
    python -m src.cli --holdings data/holdings.csv
    python -m src.cli --holdings data/holdings.csv --advise --objective max_sharpe
"""
from __future__ import annotations

import argparse
import sys

from .advisor import build_briefing_payload, generate_briefing
from .alerts import check_alerts
from .analytics import compute_metrics, portfolio_value_series
from .config import settings
from .data_provider import get_default_provider
from .exposure import analyze_exposure
from .optimizer import optimize
from .portfolio import build_portfolio, load_holdings
from .projections import compare as project_compare
from .rebalance import plan_rebalance, resulting_weights, summarize
from .watchlist import load_watchlist, screen


def _fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so symbols/arrows don't crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    parser.add_argument(
        "--rebalance", action="store_true",
        help="Print concrete buy/sell orders to move toward the thesis target",
    )
    parser.add_argument(
        "--screen", action="store_true",
        help="Print the live thesis watchlist monitor",
    )
    parser.add_argument(
        "--fundamentals", action="store_true",
        help="Include P/E, revenue growth, margin in the watchlist (slower)",
    )
    parser.add_argument(
        "--project", action="store_true",
        help="Monte-Carlo compounding projection: current vs post-rebalance allocation",
    )
    parser.add_argument("--years", type=int, default=10, help="Projection horizon")
    parser.add_argument("--monthly", type=float, default=0.0, help="Monthly contribution ($)")
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

    # --- Alerts (monitoring headline) ---
    exp = analyze_exposure(portfolio)
    holdings_hist = provider.history(portfolio.tickers, period=args.period)
    alerts = check_alerts(portfolio, exposure=exp, value_series=series,
                          holdings_history=holdings_hist)
    if alerts:
        print(f"\n{'=' * 56}\nALERTS\n{'=' * 56}")
        for a in alerts:
            print(f"[{a.severity:<4}] {a.category:<13} {a.message}")

    # --- Exposure / thesis gap ---
    print(f"\n{'=' * 56}\nTHEME EXPOSURE vs THESIS TARGET\n{'=' * 56}")
    print(f"{'Layer':<32}{'Current':>9}{'Target':>9}{'Gap':>8}")
    for layer, r in exp.by_layer.iterrows():
        print(f"{layer:<32}{_fmt_pct(r['current']):>9}{_fmt_pct(r['target']):>9}"
              f"{r['gap'] * 100:>+7.1f}%")
    if exp.concentration_flag:
        t, w = exp.top_position
        print(f"\n[!] Concentration: {t} is {_fmt_pct(w)} of the account (>25%).")

    # --- Recommend (target allocation) ---
    target = optimize(portfolio.tickers, provider, objective=args.objective)
    if target:
        print(f"\n{'=' * 56}\nTARGET ALLOCATION ({target.objective})\n{'=' * 56}")
        for t, w in sorted(target.weights.items(), key=lambda kv: -kv[1]):
            print(f"{t:<8}{_fmt_pct(w):>10}")
        print(f"\nExpected return: {_fmt_pct(target.expected_annual_return)}  "
              f"Vol: {_fmt_pct(target.annual_volatility)}  Sharpe: {target.sharpe:.2f}")

    # --- Rebalance planner ---
    if args.rebalance:
        orders = plan_rebalance(portfolio, provider)
        s = summarize(orders)
        print(f"\n{'=' * 56}\nREBALANCE PLAN → thesis target\n{'=' * 56}")
        print(f"{'Action':<12}{'Ticker':<10}{'Shares':>10}{'$ Amount':>14}  Layer")
        for o in sorted(orders, key=lambda x: (x.action, -x.dollars)):
            print(f"{o.action:<12}{o.ticker:<10}{o.shares:>10,.1f}{o.dollars:>14,.0f}  {o.layer}")
        print(f"\nSell ${s['sell_total']:,.0f} + deploy ${s['cash_deployed']:,.0f} cash "
              f"→ buy ${s['buy_total']:,.0f}  (net ${s['net']:,.0f})")
        print("Tax-free rebalance (IRA). Sizes are starting points, not targets.")

    # --- Watchlist monitor ---
    if args.screen:
        wl = load_watchlist()
        sc = screen(provider, wl, held=set(portfolio.tickers),
                    with_fundamentals=args.fundamentals)
        print(f"\n{'=' * 56}\nTHESIS WATCHLIST MONITOR\n{'=' * 56}")
        disp = sc.copy()
        for c in ["1M", "3M", "6M", "from_1y_high"]:
            disp[c] = (disp[c] * 100).round(1)
        print(disp.to_string(index=False))
        flags = sc[sc["signal"] == "pullback"]["ticker"].tolist()
        if flags:
            print(f"\nPullback (>15% off 1y high): {', '.join(flags)}")

    # --- Compounding projection ---
    if args.project:
        current_w = portfolio.weights
        target_w = resulting_weights(portfolio, plan_rebalance(portfolio, provider))
        res = project_compare(current_w, target_w, provider, portfolio.total_value,
                              years=args.years, monthly_contribution=args.monthly)
        print(f"\n{'=' * 64}\nCOMPOUNDING PROJECTION — {args.years}yr"
              f"{f', +${args.monthly:,.0f}/mo' if args.monthly else ''}\n{'=' * 64}")
        print(f"{'Allocation':<14}{'Ret/Vol':>14}{'p10':>14}{'p50 (median)':>16}{'p90':>14}")
        for label, r in [("Current", res["current"]), ("Rebalanced", res["target"])]:
            rv = f"{r['ann_return']*100:.0f}%/{r['ann_vol']*100:.0f}%"
            print(f"{label:<14}{rv:>14}{r['terminal_p10']:>14,.0f}"
                  f"{r['terminal_p50']:>16,.0f}{r['terminal_p90']:>14,.0f}")
        inv = res["current"]["invested"]
        print(f"\nInvested over horizon: ${inv:,.0f}. Returns use a capital-market assumption "
              "(rf + ERP x vol/market_vol); vol from 2y history. Gaussian model understates "
              "single-name tail risk, so it flatters concentration. Illustrative, not a forecast.")

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
