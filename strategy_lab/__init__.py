"""Strategy Lab — an isolated reinforcement-learning research sandbox (FinRL-style).

Trains a PPO agent to allocate across the thesis universe, then evaluates it strictly
out-of-sample against an equal-weight buy-and-hold. Kept separate from the core app
because RL backtests overfit easily — its output is ONE signal for the analyst to weigh,
never a directive. Heavy deps (torch, stable-baselines3) live in requirements-lab.txt.
"""
