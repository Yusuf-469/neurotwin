"""Generalised Advantage Estimation with per-step discounts (semi-Markov; RESEARCH.md §6.18).

Standard GAE (Schulman et al., 2016) assumes a constant discount γ per step. Here decisions
happen at irregular times, so step t has its own discount γ_t = γ_hour ** Δt_t. With
d_t = 1 if the episode ended at step t:

    δ_t = r_t + γ_t · V(s_{t+1}) · (1 - d_t) - V(s_t)
    Â_t = δ_t + γ_t · λ · (1 - d_t) · Â_{t+1}

Atomic placements within an epoch have Δt = 0, so γ_t = 1: no time passes and nothing is
discounted. λ stays per decision, a bias-variance knob over steps rather than over time.
Truncated episodes are bootstrapped by the trainer, which adds γ_t · V(s_final) to the last
reward before marking the step done (Huang et al. 2022, "37 details": time-limit handling).
"""

from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, gammas: np.ndarray, dones: np.ndarray,
                last_values: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Arrays of shape (T, N) (steps x envs); ``last_values`` (N,) = V(s_T) after the rollout.
    Returns (advantages, returns), returns = advantages + values."""
    t_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in reversed(range(t_steps)):
        next_values = last_values if t == t_steps - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gammas[t] * next_values * nonterminal - values[t]
        last = delta + gammas[t] * lam * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values
