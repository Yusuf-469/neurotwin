"""Statistical methodology (RESEARCH.md v2 §6.14), fixed before any test-seed run.

* Primary: **hierarchical paired bootstrap** of the mean difference PPO - baseline. Each
  replicate resamples PPO agents (K, with replacement) and then test seeds (n, with
  replacement); the PPO value for a seed is the mean over the resampled agents. Seeds are
  paired across policies by CRN. B = 10,000, percentile 95% CI. The two-sided p-value is
  2 * min(P*(Δ <= 0), P*(Δ >= 0)), floored at 1/B (a bootstrap test that the CI excludes 0 at
  the given level).
* Multiplicity: Holm-Bonferroni over the primary family (PPO vs 5 baselines x 5 scenarios x
  4 primary metrics = 100 comparisons).
* Effect sizes: d_z = mean(d) / sd(d) of per-seed paired differences (agent-averaged PPO), and
  Cliff's delta between agent-averaged PPO values and baseline values (unpaired dominance:
  P(X < Y) - P(X > Y); negative favours PPO for lower-is-better metrics).
* Sensitivity: Wilcoxon signed-rank on the per-seed paired differences, always reported, never
  chosen conditionally on a normality pre-test.

The bootstrap RNG seed is fixed (BOOTSTRAP_SEED) so the reported numbers are reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

BOOTSTRAP_SEED = 20261026
N_BOOT = 10_000


@dataclass
class Comparison:
    scenario: str
    metric: str
    baseline: str
    n_seeds: int
    n_agents: int
    ppo_mean: float
    baseline_mean: float
    diff_mean: float
    ci_low: float
    ci_high: float
    p_boot: float
    p_holm: float = float("nan")
    significant_holm: bool = False
    d_z: float = float("nan")
    cliffs_delta: float = float("nan")
    p_wilcoxon: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def hierarchical_paired_bootstrap(ppo: np.ndarray, baseline: np.ndarray, n_boot: int = N_BOOT,
                                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float, float]:
    """ppo: (K agents, n seeds); baseline: (n seeds,). Returns (diff, ci_low, ci_high, p)."""
    ppo = np.asarray(ppo, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    k, n = ppo.shape
    if baseline.shape != (n,):
        raise ValueError("baseline must have one value per seed")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(ppo.mean(axis=0) - baseline))
    agent_idx = rng.integers(0, k, size=(n_boot, k))
    seed_idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        per_seed_ppo = ppo[agent_idx[b]].mean(axis=0)            # resampled agents, all seeds
        s = seed_idx[b]
        boot[b] = np.mean(per_seed_ppo[s] - baseline[s])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
    return observed, float(lo), float(hi), float(min(1.0, max(p, 1.0 / n_boot)))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X < Y) - P(X > Y) over all pairs."""
    x, y = np.asarray(x, float)[:, None], np.asarray(y, float)[None, :]
    return float(np.mean(x < y) - np.mean(x > y))


def d_z(differences: np.ndarray) -> float:
    d = np.asarray(differences, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def compare(scenario: str, metric: str, baseline_name: str, ppo: np.ndarray,
            baseline: np.ndarray) -> Comparison:
    """One PPO-vs-baseline comparison. ppo: (K, n); baseline: (n,)."""
    diff, lo, hi, p = hierarchical_paired_bootstrap(ppo, baseline)
    agent_mean = np.asarray(ppo, float).mean(axis=0)
    d = agent_mean - np.asarray(baseline, float)
    try:
        p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = float("nan")
    return Comparison(scenario, metric, baseline_name, len(baseline), ppo.shape[0],
                      float(agent_mean.mean()), float(np.mean(baseline)), diff, lo, hi, p,
                      d_z=d_z(d), cliffs_delta=cliffs_delta(agent_mean, baseline), p_wilcoxon=p_w)


def apply_holm(comparisons: list[Comparison], alpha: float = 0.05) -> None:
    adjusted = holm([c.p_boot for c in comparisons])
    for c, p in zip(comparisons, adjusted):
        c.p_holm = p
        c.significant_holm = p < alpha
