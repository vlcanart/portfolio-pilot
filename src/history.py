"""Portfolio history — persist point-in-time snapshots to SQLite for trend tracking.

Each snapshot records total value/cost/PL, key risk metrics, the top position, and a JSON
payload of position + layer weights. Enables charting the IRA over time and lets the AI
analyst note describe what changed since last run.

Note: on Streamlit Community Cloud the filesystem is ephemeral, so snapshots written there
do not persist across reboots. Run snapshots locally (or point at an external DB) for a
durable history.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3

import pandas as pd

from .analytics import Metrics
from .exposure import ExposureReport
from .portfolio import Portfolio

DEFAULT_DB = "data/history.db"

_SCALAR_COLS = ["ts", "total_value", "total_cost", "total_pl", "sharpe", "ann_vol",
                "top_ticker", "top_weight"]


def _conn(db_path: str | os.PathLike) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS snapshots ("
        "ts TEXT PRIMARY KEY, total_value REAL, total_cost REAL, total_pl REAL, "
        "sharpe REAL, ann_vol REAL, top_ticker TEXT, top_weight REAL, payload TEXT)"
    )
    return con


def record_snapshot(
    portfolio: Portfolio,
    metrics: Metrics | None = None,
    exposure: ExposureReport | None = None,
    db_path: str | os.PathLike = DEFAULT_DB,
    ts: str | None = None,
) -> str:
    """Append (or replace same-timestamp) a snapshot. Returns the timestamp used."""
    ts = ts or _dt.datetime.now().isoformat(timespec="seconds")
    payload = {
        "weights": {t: round(w, 4) for t, w in portfolio.weights.items()},
        "positions": {p.ticker: round(p.market_value, 2) for p in portfolio.positions},
        "layers": {k: round(v, 4) for k, v in
                   (exposure.by_layer["current"].to_dict().items() if exposure else [])},
    }
    con = _conn(db_path)
    con.execute(
        "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)",
        (ts, portfolio.total_value, portfolio.total_cost, portfolio.total_pl,
         metrics.sharpe if metrics else None,
         metrics.annualized_volatility if metrics else None,
         exposure.top_position[0] if exposure else None,
         exposure.top_position[1] if exposure else None,
         json.dumps(payload)),
    )
    con.commit()
    con.close()
    return ts


def load_history(db_path: str | os.PathLike = DEFAULT_DB) -> pd.DataFrame:
    """Return all snapshots (scalar columns) as a DataFrame, oldest first."""
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=_SCALAR_COLS)
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f"SELECT {', '.join(_SCALAR_COLS)} FROM snapshots ORDER BY ts",
            con, parse_dates=["ts"],
        )
    finally:
        con.close()
    return df


def latest_change(db_path: str | os.PathLike = DEFAULT_DB) -> dict | None:
    """Value/PL delta between the two most recent snapshots, or None if <2 exist."""
    df = load_history(db_path)
    if len(df) < 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    return {
        "from": str(prev["ts"]),
        "to": str(cur["ts"]),
        "value_change": float(cur["total_value"] - prev["total_value"]),
        "value_change_pct": float(cur["total_value"] / prev["total_value"] - 1)
        if prev["total_value"] else 0.0,
    }
