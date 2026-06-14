# Portfolio Pilot

A personal investment portfolio tracker, analyzer, and AI-assisted advisor.

**What it does**
1. **Track** — load your holdings, pull live/EOD prices, value your portfolio.
2. **Analyze** — performance + risk metrics (returns, volatility, Sharpe, drawdown, allocation).
3. **Recommend** — a target allocation via mean-variance optimization, then a plain-English
   rebalancing brief written by Claude from *your* computed numbers.

> ⚠️ **Not financial advice.** This is a decision-support tool. It surfaces metrics and
> suggestions; you make the calls. Backtests and optimizers routinely overstate real-world
> returns (look-ahead bias, overfitting). Treat every recommendation as a prompt for your
> own judgment.

## Architecture

```
src/
  config.py         # settings, env loading
  data_provider.py  # market-data abstraction (yfinance default; swap to Finnhub/Alpaca later)
  portfolio.py      # load holdings, compute positions & current value
  analytics.py      # returns / volatility / Sharpe / drawdown / allocation
  optimizer.py      # PyPortfolioOpt: target weights (max Sharpe / min vol)
  advisor.py        # Claude turns metrics + target weights into a rebalancing brief
  app.py            # Streamlit dashboard
  cli.py            # command-line entry point
strategy_lab/       # (v2) optional FinRL reinforcement-learning sandbox — see ROADMAP
data/
  holdings.example.csv
```

The data layer is deliberately behind an interface (`MarketDataProvider`) so you can move
from `yfinance` (great for prototyping, unreliable for production) to Finnhub / Alpaca /
Twelve Data by writing one new class — nothing else changes.

## Setup (Windows / PowerShell)

```powershell
cd portfolio-pilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then put your ANTHROPIC_API_KEY in .env (only needed for the advisor)
copy data\holdings.example.csv data\holdings.csv   # then edit with your real positions
```

## Run

```powershell
# Command line — track + analyze + (optional) advise
python -m src.cli --holdings data\holdings.csv

# Dashboard
streamlit run src\app.py
```

## Roadmap
See [ROADMAP.md](ROADMAP.md) — v2 adds the FinRL strategy lab and a brokerage data provider.
