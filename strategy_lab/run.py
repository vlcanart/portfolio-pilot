"""Run the strategy lab end-to-end.

    python -m strategy_lab.run [--period 3y] [--timesteps 30000] [--window 20]

Fetches the thesis universe, trains a PPO allocator, evaluates it strictly out-of-sample
vs equal-weight buy-and-hold, prints results, and saves strategy_lab/last_run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from src.data_provider import get_default_provider
from strategy_lab.lab import train_and_backtest


def _pct(x: float) -> str:
    return f"{x * 100:,.1f}%"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Strategy Lab (RL allocator)")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--watchlist", default="data/watchlist_agentic.csv")
    args = ap.parse_args()

    tickers = pd.read_csv(args.watchlist)["ticker"].tolist()
    provider = get_default_provider()
    # dropna(axis=1) keeps only names with full aligned history over the window.
    prices = provider.history(tickers, period=args.period).dropna(axis=1, how="any")
    returns = prices.pct_change().dropna()
    if returns.shape[1] < 3 or len(returns) < 200:
        print("Not enough aligned history to train. Try a shorter --period.")
        return

    print(f"Universe: {returns.shape[1]} names with full {args.period} history, "
          f"{len(returns)} trading days.")
    print("Training PPO allocator (this runs on CPU; ~1-3 min)…\n")
    res = train_and_backtest(returns, window=args.window, timesteps=args.timesteps)

    s, b = res["oos_strategy"], res["oos_equal_weight"]
    print(f"{'=' * 62}\nOUT-OF-SAMPLE BACKTEST ({res['n_test_days']} days held out)\n{'=' * 62}")
    print(f"{'':<16}{'CAGR':>10}{'Vol':>10}{'Sharpe':>10}{'MaxDD':>10}")
    print(f"{'RL strategy':<16}{_pct(s['cagr']):>10}{_pct(s['vol']):>10}"
          f"{s['sharpe']:>10.2f}{_pct(s['max_dd']):>10}")
    print(f"{'Equal-weight':<16}{_pct(b['cagr']):>10}{_pct(b['vol']):>10}"
          f"{b['sharpe']:>10.2f}{_pct(b['max_dd']):>10}")

    print(f"\n{'=' * 62}\nRL SUGGESTED ALLOCATION (avg OOS weights, top 10)\n{'=' * 62}")
    for t, w in list(res["rl_weights"].items())[:10]:
        print(f"{t:<8}{_pct(w):>8}")

    os.makedirs("strategy_lab", exist_ok=True)
    with open("strategy_lab/last_run.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved strategy_lab/last_run.json")
    print("\nREAD THIS: RL backtests overfit. One regime (a mostly-up market), no live "
          "trading costs/slippage beyond a turnover proxy, short sample. Treat the weights "
          "as a discussion input for the analyst note — NOT a trade list.")


if __name__ == "__main__":
    main()
