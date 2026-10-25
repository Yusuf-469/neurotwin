"""Closed-form queueing results used to validate the engine (RESEARCH.md v2 §6.17).

Notation: c servers (beds), offered load a = lambda / mu (in Erlangs), utilisation rho = a / c.

Why M/M/c/K rather than a pure loss system
------------------------------------------
In NeuroTwin an emergency arrival is diverted when the ED (waiting room) is full, so a reduced
configuration with c beds and ED capacity Q_max is exactly an M/M/c/K queue with
K = c + Q_max (boarding patients are the queue). A pure M/M/c/c loss system (Erlang B) would
need an ED with zero waiting room, which the model does not allow. Erlang B and Erlang C are
the K = c and K -> infinity limits of M/M/c/K, and the tests check the formulas against both
limits before the simulator is compared with M/M/c/K.
"""

from __future__ import annotations

import math


def erlang_b(c: int, a: float) -> float:
    """Blocking probability of M/M/c/c (numerically stable recursion)."""
    if c < 0 or a < 0:
        raise ValueError("need c >= 0 and a >= 0")
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Probability that an arrival waits in M/M/c (requires a < c)."""
    if not 0 <= a < c:
        raise ValueError("Erlang C needs 0 <= a < c (stable queue)")
    b = erlang_b(c, a)
    return c * b / (c - a * (1.0 - b))


def mmc_mean_wait(c: int, lam: float, mu: float) -> float:
    """Mean time in queue W_q of M/M/c."""
    a = lam / mu
    return erlang_c(c, a) / (c * mu - lam)


def mmck_distribution(c: int, k: int, lam: float, mu: float) -> list[float]:
    """Stationary distribution p_0..p_K of M/M/c/K (computed in log space for stability)."""
    if not 1 <= c <= k:
        raise ValueError("need 1 <= c <= K")
    a = lam / mu
    log_terms = []
    for n in range(k + 1):
        if n <= c:
            log_terms.append(n * math.log(a) - math.lgamma(n + 1) if a > 0 else (0.0 if n == 0 else -math.inf))
        else:
            log_terms.append(c * math.log(a) - math.lgamma(c + 1) + (n - c) * math.log(a / c))
    m = max(log_terms)
    weights = [math.exp(t - m) for t in log_terms]
    total = sum(weights)
    return [w / total for w in weights]


def mmck_blocking(c: int, k: int, lam: float, mu: float) -> float:
    """Probability an arrival is blocked (diverted) in M/M/c/K (PASTA: equals p_K)."""
    return mmck_distribution(c, k, lam, mu)[k]


def mmck_mean_wait(c: int, k: int, lam: float, mu: float) -> float:
    """Mean time in queue W_q for *admitted* customers of M/M/c/K (Little's law)."""
    p = mmck_distribution(c, k, lam, mu)
    lq = sum((n - c) * p[n] for n in range(c + 1, k + 1))
    lam_eff = lam * (1.0 - p[k])
    return lq / lam_eff


def mmck_wait_probability(c: int, k: int, lam: float, mu: float) -> float:
    """Probability an *admitted* customer waits (finds all c servers busy, queue not full)."""
    p = mmck_distribution(c, k, lam, mu)
    return sum(p[c:k]) / (1.0 - p[k])
