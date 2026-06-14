# Roadmap

## v1 — Decision-support core (this scaffold)
- [x] Holdings ingestion (CSV)
- [x] Market-data provider abstraction (yfinance)
- [x] Portfolio valuation (positions, market value, P/L)
- [x] Analytics (returns, volatility, Sharpe, max drawdown, allocation)
- [x] Optimizer (PyPortfolioOpt — max Sharpe / min volatility target weights)
- [x] Claude advisor (plain-English rebalancing brief from computed numbers)
- [x] CLI + Streamlit dashboard

## v2 — Production data + strategy lab
- [ ] **Finnhub / Alpaca provider** — implement `MarketDataProvider` subclasses for reliable
      data (and, via Alpaca, a path from "recommend" to "execute" later).
- [ ] **FinRL strategy lab** (`strategy_lab/`) — an *optional, sandboxed* reinforcement-learning
      module. Train an RL agent on historical data, **backtest it honestly** (out-of-sample,
      no look-ahead), and feed its output to the advisor as one more signal alongside the
      deterministic optimizer. Kept isolated so its instability never touches the core.
- [ ] Persist snapshots over time (SQLite) to chart portfolio performance, not just a point-in-time view.

## Why FinRL is v2, not v1
FinRL learns a numerical trading *policy* from price history. It's powerful but research-grade:
heavy (PyTorch), slow to train, and easy to overfit — backtested policies frequently fall
apart on live data. The v1 core (optimizer + metrics + Claude) is deterministic and
trustworthy. FinRL is complementary, not redundant: it *generates* a candidate strategy;
Claude *explains and contextualizes* it. Add it once the core is solid.
