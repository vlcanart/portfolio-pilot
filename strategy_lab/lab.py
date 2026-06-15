"""Train + out-of-sample backtest for the strategy lab."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .env import PortfolioEnv

TRADING_DAYS = 252


def _metrics(equity: np.ndarray) -> dict:
    equity = np.asarray(equity, dtype=np.float64)
    rets = np.diff(equity) / equity[:-1]
    n = max(len(rets), 1)
    cagr = float(equity[-1] / equity[0]) ** (TRADING_DAYS / n) - 1
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((rets.mean() * TRADING_DAYS) / (vol + 1e-9))
    max_dd = float((equity / np.maximum.accumulate(equity) - 1).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def _run_policy(model, returns: np.ndarray, window: int):
    env = PortfolioEnv(returns, window)
    obs, _ = env.reset()
    equity = [1.0]
    wsum = np.zeros(env.N, dtype=np.float64)
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        equity.append(equity[-1] * (1.0 + info["port_return"]))
        wsum += info["weights"]
        steps += 1
        done = term or trunc
    return np.array(equity), (wsum / max(steps, 1))


def _equal_weight_equity(returns_df: pd.DataFrame) -> np.ndarray:
    w = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    port = returns_df.values @ w
    return np.concatenate([[1.0], np.cumprod(1.0 + port)])


def train_and_backtest(
    returns_df: pd.DataFrame,
    window: int = 20,
    train_frac: float = 0.7,
    timesteps: int = 30000,
    seed: int = 7,
) -> dict:
    """Train PPO on the first `train_frac` of history; evaluate OOS on the remainder."""
    from stable_baselines3 import PPO

    R = returns_df.values.astype(np.float32)
    split = int(len(R) * train_frac)
    train_R = R[:split]
    test_R = R[split - window:]          # include window lead-in for the first obs
    test_df = returns_df.iloc[split:]

    env = PortfolioEnv(train_R, window)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                n_steps=512, batch_size=128, gamma=0.99, gae_lambda=0.95, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)

    strat_eq, avg_w = _run_policy(model, test_R, window)
    bh_eq = _equal_weight_equity(test_df)

    order = np.argsort(avg_w)[::-1]
    return {
        "universe": list(returns_df.columns),
        "n_train_days": split,
        "n_test_days": len(test_df),
        "timesteps": timesteps,
        "oos_strategy": _metrics(strat_eq),
        "oos_equal_weight": _metrics(bh_eq),
        "rl_weights": {returns_df.columns[i]: round(float(avg_w[i]), 4) for i in order},
    }
