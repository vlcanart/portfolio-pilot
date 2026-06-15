"""Streamlit dashboard. Run locally with:  streamlit run src/app.py

Deployed on Streamlit Community Cloud, this app reads three things from st.secrets
(set in the app's Secrets manager, never committed):
  - app_password   : gates the whole dashboard behind a password
  - ANTHROPIC_API_KEY : enables the AI advisor
  - holdings_csv   : your portfolio as a CSV string (keeps real $ amounts out of the repo)
"""
from __future__ import annotations

import hmac
import io
import os
import sys
from pathlib import Path

# Ensure the repo root is importable. Streamlit Cloud runs this file with src/ on the
# path (not the repo root), so `import src.*` would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st


def _secret(key: str, default=None):
    """Read a Streamlit secret without crashing when no secrets file exists (local dev)."""
    try:
        return st.secrets[key]
    except Exception:
        return default


# Hydrate the API key into the environment BEFORE importing modules that read it at
# import time (src.config builds its settings object on import).
_api_key = _secret("ANTHROPIC_API_KEY")
if _api_key:
    os.environ["ANTHROPIC_API_KEY"] = _api_key

from src.advanced_metrics import compute_advanced                   # noqa: E402
from src.attribution import compute_brinson                         # noqa: E402
from src.advisor import build_briefing_payload, generate_briefing  # noqa: E402
from src.alerts import check_alerts                                 # noqa: E402
from src.analyst_note import build_note_payload, generate_note      # noqa: E402
from src.analytics import compute_metrics, portfolio_value_series   # noqa: E402
from src.history import latest_change, load_history, record_snapshot  # noqa: E402
from src.config import settings                                     # noqa: E402
from src.data_provider import get_default_provider                  # noqa: E402
from src.exposure import analyze_exposure                           # noqa: E402
from src.perf_chart import (                                        # noqa: E402
    normalized_history, rank_by_total_return, universe_tickers,
)
from src.optimizer import optimize                                  # noqa: E402
from src.portfolio import build_portfolio, load_holdings            # noqa: E402
from src.projections import compare as project_compare             # noqa: E402
from src.rebalance import plan_rebalance, resulting_weights, summarize  # noqa: E402
from src.watchlist import load_watchlist, screen                    # noqa: E402

st.set_page_config(page_title="Portfolio Pilot", layout="wide")


def check_password() -> bool:
    """Gate the app behind a password (official Streamlit pattern).

    If no `app_password` secret is set (local dev), access is allowed so you can
    iterate without a password. On the deployed app, set the secret to lock it.
    """
    expected = _secret("app_password")
    if not expected:
        return True  # local dev — no gate configured

    if st.session_state.get("password_ok", False):
        return True

    def _entered():
        if hmac.compare_digest(st.session_state.get("pw", ""), str(expected)):
            st.session_state["password_ok"] = True
            del st.session_state["pw"]
        else:
            st.session_state["password_ok"] = False

    st.title("📈 Portfolio Pilot")
    st.text_input("Password", type="password", on_change=_entered, key="pw")
    if st.session_state.get("password_ok") is False:
        st.error("😕 Incorrect password")
    return False


if not check_password():
    st.stop()

st.title("📈 Portfolio Pilot")
st.caption("Track · Analyze · Recommend — decision support, not financial advice.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Holdings CSV (ticker, shares, cost_basis)", type="csv")
    period = st.selectbox("History window", ["6mo", "1y", "2y", "5y"], index=1)
    objective = st.selectbox("Optimizer objective", ["max_sharpe", "min_volatility"])
    max_weight = st.slider("Max weight per position (optimizer)", 0.10, 1.0, 0.30, 0.05)
    show_watchlist = st.checkbox("Show thesis watchlist monitor", value=False)
    wl_fundamentals = st.checkbox("…with fundamentals (slower)", value=False)
    show_projection = st.checkbox("Show compounding projection", value=False)
    proj_years = st.slider("Projection horizon (yrs)", 5, 30, 15)
    proj_monthly = st.number_input("Monthly contribution ($)", 0, 50000, 1500, step=500)
    show_attribution = st.checkbox("Show performance attribution (Brinson)", value=False)
    want_note = st.checkbox("Generate AI analyst note", value=False)
    want_advice = st.checkbox("Generate AI brief (quick)", value=False)
    st.markdown("---")
    do_snapshot = st.button("📸 Record snapshot")
    st.caption("Snapshots persist locally. On Streamlit Cloud the filesystem is "
               "ephemeral, so cloud history resets on reboot.")


def _load_holdings_df():
    """Source holdings: uploaded file → st.secrets['holdings_csv'] → local CSV."""
    if uploaded is not None:
        return load_holdings(uploaded)
    secret_csv = _secret("holdings_csv")
    if secret_csv:
        return load_holdings(io.StringIO(secret_csv))
    try:
        return load_holdings("data/holdings.csv")
    except FileNotFoundError:
        st.info("Upload a holdings CSV, or set `holdings_csv` in the app secrets.")
        st.stop()


holdings = _load_holdings_df()
provider = get_default_provider()
portfolio = build_portfolio(holdings, provider)

if not portfolio.positions:
    st.error("No priceable positions found. Check your tickers.")
    st.stop()

# --- Track ---
c1, c2, c3 = st.columns(3)
c1.metric("Total value", f"${portfolio.total_value:,.0f}")
c2.metric("Total cost", f"${portfolio.total_cost:,.0f}")
c3.metric("P/L", f"${portfolio.total_pl:,.0f}", f"{portfolio.total_pl_pct * 100:.1f}%")

left, right = st.columns([3, 2])
with left:
    st.subheader("Positions")
    df = portfolio.to_frame()[
        ["ticker", "shares", "price", "market_value", "unrealized_pl", "weight"]
    ].sort_values("weight", ascending=False)

    # Per-position alpha contribution vs SPY over the selected period
    _alpha_hist = provider.history(
        [t for t in portfolio.tickers if t != "CASH"] + ["SPY"], period=period
    )
    _spy_pr = (
        float(_alpha_hist["SPY"].dropna().iloc[-1] / _alpha_hist["SPY"].dropna().iloc[0] - 1)
        if not _alpha_hist.empty and "SPY" in _alpha_hist.columns
        else 0.0
    )
    def _pos_return(t: str) -> float:
        if t in _alpha_hist.columns:
            c = _alpha_hist[t].dropna()
            if len(c) >= 2:
                return float(c.iloc[-1] / c.iloc[0] - 1)
        return float("nan")

    df["period_rtn %"] = df["ticker"].apply(_pos_return) * 100
    df["alpha_contrib %"] = df.apply(
        lambda r: r["weight"] * (_pos_return(r["ticker"]) - _spy_pr) * 100
        if not pd.isna(r["period_rtn %"]) else float("nan"),
        axis=1,
    ).round(2)
    df["weight"] = (df["weight"] * 100).round(1)
    df["period_rtn %"] = df["period_rtn %"].round(1)
    st.dataframe(df, width="stretch", hide_index=True)
with right:
    st.subheader("Allocation")
    st.bar_chart(pd.Series(portfolio.weights).sort_values(ascending=False))

# --- Analyze ---
series = portfolio_value_series(portfolio, provider, period=period)
metrics = compute_metrics(series, period=period)
if metrics:
    st.subheader(f"Performance & risk ({metrics.period})")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total return", f"{metrics.total_return * 100:.1f}%")
    m2.metric("Ann. return", f"{metrics.annualized_return * 100:.1f}%")
    m3.metric("Ann. volatility", f"{metrics.annualized_volatility * 100:.1f}%")
    m4.metric("Sharpe", f"{metrics.sharpe:.2f}")
    m5.metric("Max drawdown", f"{metrics.max_drawdown * 100:.1f}%")
    st.line_chart(series, height=240)

    # Institutional-grade extensions
    try:
        adv = compute_advanced(series, provider, portfolio_value=portfolio.total_value)
    except TypeError:
        adv = None  # old cached .pyc — user should reboot the Streamlit app
    if adv:
        st.markdown("**Advanced risk metrics**")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Sortino", f"{adv.sortino:.2f}",
                  help="Annual excess return / downside-only vol")
        a2.metric("Calmar", f"{adv.calmar:.2f}",
                  help="Annual return / |max drawdown|")
        a3.metric("Beta vs SPY", f"{adv.beta_spy:.2f}",
                  help="Sensitivity to S&P 500 — 1.0 = market-like, >1 = amplified")
        a4.metric("Info Ratio", f"{adv.info_ratio:.2f}",
                  help="Active return vs SPY / tracking error")
        a5.metric("Jensen's α", f"{adv.jensens_alpha * 100:+.1f}%",
                  help="Excess return above CAPM expectation (R_p − [R_f + β(R_m − R_f)])")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Rolling Sharpe 30d", f"{adv.rolling_sharpe['30d']:.2f}")
        b2.metric("Rolling Sharpe 90d", f"{adv.rolling_sharpe['90d']:.2f}")
        b3.metric("Rolling Sharpe 365d", f"{adv.rolling_sharpe['365d']:.2f}")
        b4.metric("VaR 99% (1d)", f"{adv.var_99_1d * 100:.1f}%",
                  help="Historical 1-day loss exceeded 1% of the time")
        b5.metric("CVaR 99% (1d)", f"{adv.cvar_99_1d * 100:.1f}%",
                  help="Mean loss on the worst 1% of days")

        st.metric("MTD", f"{adv.mtd_return * 100:+.1f}%")

        if adv.stress_scenarios:
            st.markdown("**Macro stress scenarios** (beta-linear approximation)")
            s_rows = [
                {
                    "Scenario": sc.name,
                    "Description": sc.description,
                    "SPY shock": f"{sc.spy_shock * 100:.0f}%",
                    "Est. portfolio move": f"{sc.portfolio_pct * 100:+.1f}%",
                    "Est. P&L": f"${sc.portfolio_dollars:,.0f}",
                }
                for sc in adv.stress_scenarios
            ]
            st.dataframe(pd.DataFrame(s_rows), hide_index=True, width="stretch")
            st.caption(
                "Uses portfolio beta vs SPY; AI Sector Unwind applies 1.5× beta "
                "to reflect the AI-heavy tilt. Linear approximation only."
            )

# --- Historical performance chart ---
st.subheader("Historical performance — all tracked names")
_perf_col1, _perf_col2, _perf_col3 = st.columns([2, 2, 4])
with _perf_col1:
    perf_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=3,
                               key="perf_period")
with _perf_col2:
    perf_gran = st.radio("Granularity", ["daily", "monthly", "yearly"],
                         index=1, horizontal=True, key="perf_gran")
with _perf_col3:
    show_benchmark = st.checkbox("Show SPY benchmark", value=True, key="perf_spy")

with st.spinner("Loading history for all tracked names…"):
    _uni = universe_tickers(portfolio.tickers)
    _bench = "SPY" if show_benchmark else None
    _norm = normalized_history(provider, _uni, period=perf_period,
                               granularity=perf_gran, benchmark=_bench)

if not _norm.empty:
    _all_names = sorted(_norm.columns.tolist())
    _default = (
        [t for t in portfolio.tickers if t != "CASH" and t in _norm.columns]
        + (["SPY"] if show_benchmark and "SPY" in _norm.columns else [])
    )
    _sel = st.multiselect(
        "Filter names (leave blank = show all)",
        options=_all_names,
        default=_default,
        key="perf_sel",
    )
    _chart_df = _norm[_sel] if _sel else _norm
    st.line_chart(_chart_df, height=380)
    st.caption(
        "Normalized to 100 at first data point. "
        f"{perf_gran.capitalize()} closes, {perf_period} window. "
        "Includes held positions + 46-name Agentic Economy watchlist."
    )

    _ranked = rank_by_total_return(_norm[_sel] if _sel else _norm)
    if not _ranked.empty:
        _ranked["total_return"] = (_ranked["total_return"] * 100).round(1)
        _ranked.columns = ["Ticker", "Total return (%)"]
        with st.expander("Return ranking over selected window"):
            st.dataframe(_ranked, hide_index=True, width="stretch")
else:
    st.info("No price history available — check your data provider or try a shorter period.")

# --- Brinson attribution ---
if show_attribution:
    st.subheader("Performance attribution (Brinson-Fachler)")
    with st.spinner("Computing attribution — fetching layer returns…"):
        brinson = compute_brinson(portfolio, provider, period=period)
    if brinson:
        bx1, bx2, bx3, bx4 = st.columns(4)
        bx1.metric(
            "Portfolio return",
            f"{brinson.portfolio_return * 100:.1f}%",
            help=f"Weighted return of held positions ({period})",
        )
        bx2.metric(
            "Benchmark (SPY)",
            f"{brinson.benchmark_return * 100:.1f}%",
        )
        bx3.metric(
            "Active return",
            f"{brinson.total_active_return * 100:+.1f}%",
            help="Portfolio − SPY, decomposed below",
        )
        bx4.metric(
            "Allocation effect",
            f"{brinson.allocation_effect * 100:+.2f}%",
            help="Value added by over/underweighting thesis layers vs target",
        )
        by1, by2, by3, _ = st.columns(4)
        by1.metric(
            "Selection effect",
            f"{brinson.selection_effect * 100:+.2f}%",
            help="Value added by stock picks within each layer vs equal-weight layer",
        )
        by2.metric(
            "Interaction effect",
            f"{brinson.interaction_effect * 100:+.2f}%",
            help="Joint allocation × selection effect",
        )

        detail = brinson.by_layer.copy()
        pct_cols = [
            "Port. weight", "Target weight", "Active weight",
            "Port. layer rtn", "Bmk. layer rtn",
            "Allocation", "Selection", "Interaction", "Total",
        ]
        for c in pct_cols:
            detail[c] = (detail[c] * 100).round(2)
        with st.expander("Attribution by layer"):
            st.dataframe(detail, hide_index=True, width="stretch")
        st.caption(
            f"Benchmark allocation = thesis target weights; "
            "layer benchmark = equal-weight of all 46 tracked names per layer. "
            f"Period: {period}."
        )
    else:
        st.info("Not enough history to compute attribution.")

# --- Alerts ---
exp = analyze_exposure(portfolio)
_alert_hist = provider.history(portfolio.tickers, period=period)
alerts = check_alerts(portfolio, exposure=exp, value_series=series,
                      holdings_history=_alert_hist)
if alerts:
    st.subheader("Alerts")
    _render = {"HIGH": st.error, "WARN": st.warning, "INFO": st.info}
    for a in alerts:
        _render[a.severity](f"**{a.severity}** · {a.category} — {a.message}")

sc = None  # watchlist screen, computed if the monitor is shown

# --- History ---
if do_snapshot:
    ts = record_snapshot(portfolio, metrics=metrics, exposure=exp)
    st.success(f"Snapshot recorded at {ts}.")
_hist = load_history()
if len(_hist) > 1:
    st.subheader("Value history")
    st.line_chart(_hist.set_index("ts")["total_value"], height=220)
    chg = latest_change()
    if chg:
        st.caption(f"Since {chg['from']}: ${chg['value_change']:,.0f} "
                   f"({chg['value_change_pct'] * 100:+.1f}%)")

# --- Exposure / thesis gap ---
st.subheader("Theme exposure vs thesis target")
if exp.concentration_flag:
    t, w = exp.top_position
    st.warning(f"Concentration risk: **{t}** is {w * 100:.0f}% of the account (>25%).")
ex_df = exp.by_layer.copy()
ex_df.columns = ["Current", "Target", "Gap"]
st.bar_chart(ex_df[["Current", "Target"]])
st.dataframe(
    (ex_df * 100).round(1).rename(columns=lambda c: f"{c} %"),
    width="stretch",
)
st.caption("Gap = current − target (positive = overweight vs thesis, negative = underweight).")

# --- Rebalance plan ---
st.subheader("Rebalance plan → thesis target")
orders = plan_rebalance(portfolio, provider)
s = summarize(orders)
od = pd.DataFrame([o.__dict__ for o in orders])
if not od.empty:
    od["shares"] = od["shares"].round(1)
    od["dollars"] = od["dollars"].round(0)
    st.dataframe(
        od[["action", "ticker", "shares", "dollars", "layer", "reason"]],
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Sell ${s['sell_total']:,.0f} + deploy ${s['cash_deployed']:,.0f} cash → "
        f"buy ${s['buy_total']:,.0f}. Tax-free rebalance (IRA). Sizes are starting points, not targets."
    )

# --- Recommend ---
target = optimize(portfolio.tickers, provider, objective=objective, max_weight=max_weight)
if target:
    st.subheader(f"Target allocation ({target.objective})")
    comp = pd.DataFrame(
        {
            "current": pd.Series(portfolio.weights),
            "target": pd.Series(target.weights),
        }
    ).fillna(0.0)
    st.bar_chart(comp)
    st.caption(
        f"Expected return {target.expected_annual_return * 100:.1f}% · "
        f"Vol {target.annual_volatility * 100:.1f}% · Sharpe {target.sharpe:.2f} · "
        "expected return is extrapolated from history — treat as directional, not a forecast."
    )

# --- Thesis watchlist monitor ---
if show_watchlist:
    st.subheader("Thesis watchlist monitor")
    with st.spinner("Loading live watchlist…"):
        sc = screen(provider, load_watchlist(), held=set(portfolio.tickers),
                    with_fundamentals=wl_fundamentals)
    if not sc.empty:
        disp = sc.copy()
        for c in ["1M", "3M", "6M", "from_1y_high"]:
            disp[c] = (disp[c] * 100).round(1)
        st.dataframe(disp, width="stretch", hide_index=True)
        pulls = sc[sc["signal"] == "pullback"]["ticker"].tolist()
        if pulls:
            st.caption("Pullback (>15% off 1y high) — possible entry points: " + ", ".join(pulls))

# --- Compounding projection ---
if show_projection:
    st.subheader(f"Compounding projection — {proj_years}yr, +${proj_monthly:,.0f}/mo")
    with st.spinner("Running Monte-Carlo…"):
        target_w = resulting_weights(portfolio, orders)
        res = project_compare(portfolio.weights, target_w, provider,
                              portfolio.total_value, years=proj_years,
                              monthly_contribution=float(proj_monthly))
    rows = []
    for label, r in [("Current", res["current"]), ("Rebalanced", res["target"])]:
        rows.append({
            "Allocation": label,
            "Exp. return": f"{r['ann_return'] * 100:.0f}%",
            "Volatility": f"{r['ann_vol'] * 100:.0f}%",
            "p10": f"${r['terminal_p10']:,.0f}",
            "Median": f"${r['terminal_p50']:,.0f}",
            "p90": f"${r['terminal_p90']:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    paths = pd.DataFrame({
        "Current (median)": res["current"]["path"]["p50"],
        "Rebalanced (median)": res["target"]["path"]["p50"],
    })
    st.line_chart(paths, height=240)
    st.caption(
        f"Invested over horizon: ${res['current']['invested']:,.0f}. Returns use a "
        "capital-market assumption (rf + ERP×vol/market); volatility from 2y history. "
        "Gaussian model understates single-name tail risk. Illustrative, not a forecast."
    )

# --- AI analyst note ---
if want_note:
    st.subheader("AI analyst note")
    if not settings.anthropic_api_key:
        st.warning("Set ANTHROPIC_API_KEY in the app secrets to enable the analyst note.")
    else:
        with st.spinner("Writing analyst note…"):
            note_payload = build_note_payload(
                portfolio, metrics, exp, alerts, watchlist_screen=sc, change=latest_change(),
            )
            st.markdown(generate_note(note_payload))

# --- Advise (quick brief) ---
if want_advice:
    st.subheader("AI rebalancing brief")
    if not settings.anthropic_api_key:
        st.warning("Set ANTHROPIC_API_KEY in the app secrets to enable the AI advisor.")
    else:
        with st.spinner("Asking Claude…"):
            payload = build_briefing_payload(portfolio, metrics, target)
            st.markdown(generate_briefing(payload))

st.caption("⚠️ Not financial advice. Backtests and optimizers can overstate real-world returns.")
