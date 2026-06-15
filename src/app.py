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

from src.advisor import build_briefing_payload, generate_briefing  # noqa: E402
from src.analytics import compute_metrics, portfolio_value_series   # noqa: E402
from src.config import settings                                     # noqa: E402
from src.data_provider import get_default_provider                  # noqa: E402
from src.exposure import analyze_exposure                           # noqa: E402
from src.optimizer import optimize                                  # noqa: E402
from src.portfolio import build_portfolio, load_holdings            # noqa: E402
from src.rebalance import plan_rebalance, summarize                 # noqa: E402
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
    want_advice = st.checkbox("Generate AI brief", value=False)


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
    df["weight"] = (df["weight"] * 100).round(1)
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

# --- Exposure / thesis gap ---
exp = analyze_exposure(portfolio)
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
        sc = screen(provider, load_watchlist(), held=set(portfolio.tickers))
    if not sc.empty:
        disp = sc.copy()
        for c in ["1M", "3M", "6M", "from_1y_high"]:
            disp[c] = (disp[c] * 100).round(1)
        st.dataframe(disp, width="stretch", hide_index=True)
        pulls = sc[sc["signal"] == "pullback"]["ticker"].tolist()
        if pulls:
            st.caption("Pullback (>15% off 1y high) — possible entry points: " + ", ".join(pulls))

# --- Advise ---
if want_advice:
    st.subheader("AI rebalancing brief")
    if not settings.anthropic_api_key:
        st.warning("Set ANTHROPIC_API_KEY in the app secrets to enable the AI advisor.")
    else:
        with st.spinner("Asking Claude…"):
            payload = build_briefing_payload(portfolio, metrics, target)
            st.markdown(generate_briefing(payload))

st.caption("⚠️ Not financial advice. Backtests and optimizers can overstate real-world returns.")
