"""Statistical methodology (RESEARCH.md §6.14) checked on synthetic data before any test run."""

import numpy as np
import pytest

from neurotwin.experiments.evaluate import analyse, build_jobs, run_job
from neurotwin.experiments.statistics import (cliffs_delta, compare, d_z, hierarchical_paired_bootstrap,
                                              holm)


def test_holm_matches_manual_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_cliffs_delta_and_dz_known_values():
    assert cliffs_delta([1, 2], [3, 4]) == 1.0          # X always smaller
    assert cliffs_delta([3, 4], [1, 2]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    assert d_z([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_bootstrap_ci_covers_true_difference_at_nominal_rate():
    """Coverage check: with a true difference of -1, ~95% of CIs must contain it."""
    rng = np.random.default_rng(0)
    covered, trials = 0, 200
    for t in range(trials):
        seed_effect = rng.normal(10, 3, size=40)                     # shared by CRN
        base = seed_effect + rng.normal(0, 1, 40)
        ppo = seed_effect - 1.0 + rng.normal(0, 1, (5, 40)) + rng.normal(0, 0.3, (5, 1))
        _, lo, hi, _ = hierarchical_paired_bootstrap(ppo, base, n_boot=1000, seed=t)
        covered += lo <= -1.0 <= hi
    assert 0.88 <= covered / trials <= 0.99


def test_bootstrap_detects_effect_and_not_null():
    rng = np.random.default_rng(1)
    s = rng.normal(10, 3, 50)
    base = s + rng.normal(0, 1, 50)
    _, lo, hi, p = hierarchical_paired_bootstrap(s - 2 + rng.normal(0, 1, (5, 50)), base, 2000)
    assert hi < 0 and p < 0.01
    _, lo, hi, p = hierarchical_paired_bootstrap(s + rng.normal(0, 1, (5, 50)), base, 2000)
    assert lo < 0 < hi and p > 0.05


def test_hierarchical_bootstrap_widens_with_agent_variance():
    """Between-agent variability must widen the CI (that is why agents are resampled)."""
    rng = np.random.default_rng(2)
    s = rng.normal(10, 3, 50)
    base = s + rng.normal(0, 1, 50)
    noise = rng.normal(0, 1, (5, 50))
    tight = hierarchical_paired_bootstrap(s + noise, base, 2000)
    loose = hierarchical_paired_bootstrap(s + noise + np.array([[-3], [-1], [0], [1], [3]]), base, 2000)
    assert (loose[2] - loose[1]) > 1.5 * (tight[2] - tight[1])


def test_boarding_metric_counts_patients_still_waiting():
    """A policy that never places anyone must get the worst boarding, not an undefined one."""
    from neurotwin.experiments.metrics import episode_metrics
    from neurotwin.experiments.objective import CostWeights
    from neurotwin.simulation import SCENARIOS, HospitalSimulation, load_default_config, run_episode
    from neurotwin.simulation.entities import RequestType

    class NeverPlace:
        def decide(self, sim):
            return None

    # Short warm-up: with nobody placed, the ED fills within ~1.5 days and every later arrival
    # is diverted, so requests are only created early.
    cfg = load_default_config().with_overrides(horizon_hours=300.0, warmup_hours=1.0)
    sim = HospitalSimulation(cfg, SCENARIOS["A"], seed=1)
    run_episode(sim, NeverPlace())
    m = episode_metrics(sim, CostWeights.load())
    expected = [cfg.horizon_hours - r.created_at for r in sim.request_log()
                if r.created_at >= cfg.warmup_hours and r.from_ed
                and r.rtype in (RequestType.ADMIT_GENERAL, RequestType.ADMIT_ICU)]
    assert m["mean_boarding_h"] == pytest.approx(np.mean(expected))
    assert m["censored_ed_requests"] == len(expected) > 10
    assert np.isnan(m["mean_boarding_placed_only_h"])          # the flawed version is undefined


def test_end_to_end_on_validation_seeds(tmp_path):
    """Runner + analysis on 3 validation seeds, short episodes, untrained-free policies only."""
    from neurotwin.experiments.statistics import Comparison
    overrides = {"horizon_hours": 300.0, "warmup_hours": 100.0}
    jobs = build_jobs([100_000, 100_001, 100_002], ["B"], agents=1, agent_dir=tmp_path,
                      overrides=overrides, policies=("priority", "fcfs"))
    rows = [run_job(j) for j in jobs]
    # Treat 'priority' as the comparison policy to exercise analyse() without a trained agent.
    for r in rows:
        if r["policy"] == "priority":
            r["policy"], r["agent"] = "ppo", 0
    comps = analyse(rows, baselines=("fcfs",))
    assert len(comps) == 4 and all(isinstance(c, Comparison) for c in comps)
    assert all(c.n_seeds == 3 for c in comps)
