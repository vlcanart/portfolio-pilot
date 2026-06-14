"""Central configuration. Loads .env if present; everything has a sane default."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    finnhub_api_key: str | None = os.getenv("FINNHUB_API_KEY") or None
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.045"))

    # Trading days per year, used to annualize daily return statistics.
    trading_days: int = 252

    # Claude model + effort for the advisor. Adaptive thinking is set in advisor.py.
    advisor_model: str = "claude-opus-4-8"
    advisor_effort: str = "high"


settings = Settings()
