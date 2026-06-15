"""Gymnasium portfolio-allocation environment.

Observation: the trailing `window` days of returns for each asset (flattened).
Action: a real vector per asset, softmax-normalized to long-only weights summing to 1.
Reward: next-day portfolio log-return minus a turnover penalty (proxy for trading cost).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, window: int = 20, turnover_penalty: float = 0.001):
        super().__init__()
        self.returns = np.asarray(returns, dtype=np.float32)
        self.T, self.N = self.returns.shape
        self.window = window
        self.turnover_penalty = turnover_penalty
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window * self.N,), dtype=np.float32
        )
        self._t = self.window
        self._w = np.ones(self.N, dtype=np.float32) / self.N

    def _obs(self) -> np.ndarray:
        return self.returns[self._t - self.window:self._t].flatten().astype(np.float32)

    @staticmethod
    def _to_weights(action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        a = a - a.max()
        e = np.exp(a)
        return (e / e.sum()).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.window
        self._w = np.ones(self.N, dtype=np.float32) / self.N
        return self._obs(), {}

    def step(self, action):
        w = self._to_weights(action)
        r = self.returns[self._t]
        port = float(np.dot(w, r))
        turnover = float(np.abs(w - self._w).sum())
        reward = float(np.log1p(port) - self.turnover_penalty * turnover)
        self._w = w
        self._t += 1
        terminated = self._t >= self.T
        obs = (self._obs() if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))
        return obs, reward, terminated, False, {"weights": w, "port_return": port}
