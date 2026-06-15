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
from .analyst_note import build_note_payload, generate_note
from .analytics import compute_metrics, portfolio_value_series
from .history import latest_change, load_history, record_snapshot
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
    parser.add_argument(
        "--snapshot", action="store_true",
        help="Record a point-in-time snapshot to the SQLite history",
    )
    parser.add_argument(
        "--note", action="store_true",
        help="Generate the recurring AI analyst note (needs ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--email", action="store_true",
        help="Email the daily digest (charts + note) via SMTP (needs SMTP creds)",
    )
    parser.add_argument(
        "--email-preview", action="store_true",
        help="Write the digest to a local HTML file instead of sending",
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

    sc = None  # watchlist screen, computed on demand below

    # --- Snapshot to history ---
    if args.snapshot:
        ts = record_snapshot(portfolio, metrics=metrics, exposure=exp)
        hist = load_history()
        print(f"\n{'=' * 56}\nSNAPSHOT RECORDED ({ts})\n{'=' * 56}")
        print(f"History now holds {len(hist)} snapshot(s).")
        chg = latest_change()
        if chg:
            print(f"Change since {chg['from']}: ${chg['value_change']:,.0f} "
                  f"({chg['value_change_pct'] * 100:+.1f}%)")

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

    # --- AI analyst note (also used by the email digest) ---
    note_text = None
    if args.note or args.email or args.email_preview:
        if sc is None:
            sc = screen(provider, load_watchlist(), held=set(portfolio.tickers))
        payload = build_note_payload(portfolio, metrics, exp, alerts,
                                     watchlist_screen=sc, change=latest_change())
        try:
            note_text = generate_note(payload)
        except RuntimeError as e:
            print(f"(note skipped) {e}")
        if args.note and note_text:
            print(f"\n{'=' * 56}\nAI ANALYST NOTE\n{'=' * 56}")
            print(note_text)

    # --- Email digest ---
    if args.email or args.email_preview:
        import datetime as _dt
        from .emailer import compose_digest, render_preview, send_digest
        subject, html, charts = compose_digest(
            portfolio, metrics, exp, alerts, sc, load_history(), note_text,
            date_str=_dt.date.today().isoformat(),
        )
        if args.email_preview:
            path = render_preview(html, charts, "notes/digest_preview.html")
            print(f"\nDigest preview written: {path}")
        if args.email:
            try:
                send_digest(subject, html, charts)
                print(f"\nEmailed digest to {settings.email_to}")
            except Exception as e:
                print(f"\n(email failed) {e}")

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
