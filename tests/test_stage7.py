"""Stage 7 analysis components."""

import numpy as np
import pytest

from neurotwin.experiments.robustness import POLICIES, RELATIVE, STRESS, bootstrap_degradation
from neurotwin.experiments.structure import isotonic_decreasing
from neurotwin.rl.env import HospitalAllocationEnv
from neurotwin.simulation import SCENARIOS, load_default_config

SHORT = load_default_config().with_overrides(horizon_hours=300.0, warmup_hours=100.0)


def test_dropped_features_are_always_zero():
    env = HospitalAllocationEnv(SHORT, SCENARIOS["B"], drop_features=(0, 13, 20))
    obs, info = env.reset(seed=3)
    rng = np.random.default_rng(0)
    seen_nonzero_kept = False
    for _ in range(300):
        assert np.all(obs[[0, 13, 20]] == 0.0)
        seen_nonzero_kept |= bool(obs[1] != 0.0)
        obs, _, _, trunc, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
        if trunc:
            break
    assert seen_nonzero_kept  # other features are untouched
    with pytest.raises(ValueError):
        HospitalAllocationEnv(SHORT, drop_features=(21,))


def test_ppo_dr_range_is_milder_than_every_test_scenario():
    from neurotwin.experiments.ppo_dr import check_training_range, sample_scenario

    guard = check_training_range()
    assert guard["worst_corner"]["rho_general"] < guard["mildest_test"]["rho_general"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample_scenario(rng)
        assert 0.95 <= s.arrival_multiplier <= 1.04 and 0.97 <= s.staff_multiplier <= 1.0
        assert 0.97 <= s.capacity_multiplier <= 1.0


def test_warm_start_with_sampled_scenario():
    from neurotwin.digital_twin import WarmStartPool
    from neurotwin.experiments.ppo_dr import sample_scenario
    from neurotwin.simulation.validation.reduced import GreedyPlacement

    pool = WarmStartPool.generate(SHORT, SCENARIOS["A"], [0, 1], GreedyPlacement())
    env = HospitalAllocationEnv(
        SHORT, start_mode="warm", pool=pool, scenario_sampler=sample_scenario
    )
    env.reset(seed=5)
    first = env.sim.scenario
    assert first.name == "dr_train" and 0.95 <= first.arrival_multiplier <= 1.04
    env.reset(seed=6)
    assert env.sim.scenario != first  # a new draw per episode
    env.reset(seed=5)
    assert env.sim.scenario == first  # reproducible from the episode seed


def test_isotonic_decreasing_fit():
    y = np.array([0.1, 0.12, 0.05, 0.06, 0.01])
    fit = isotonic_decreasing(y, np.ones(5))
    assert np.all(np.diff(fit) <= 1e-12)
    assert fit == pytest.approx([0.11, 0.11, 0.055, 0.055, 0.01])
    already = np.array([0.5, 0.3, 0.1])
    assert isotonic_decreasing(already, np.ones(3)) == pytest.approx(already)


def test_degradation_on_a_known_case():
    """Policy whose metrics exactly double under stress -> +100% degradation, tight CI."""
    rng = np.random.default_rng(0)
    m = {}
    for p in POLICIES:
        k = 5 if p == "ppo" else 1
        base = rng.uniform(5, 10, size=(k, 30))
        m[p] = {
            s: {
                metric: base * (1 if s == "A" else 2)
                for metric in RELATIVE + ("diversion_per_100",)
            }
            for s in ("A",) + STRESS
        }
    res = bootstrap_degradation(m, n_boot=500)
    for p in POLICIES:
        assert res["robustness_index"][p]["estimate"] == pytest.approx(100.0)
        assert res["robustness_index"][p]["ci"] == pytest.approx([100.0, 100.0])
        d = res["degradation"][p]["B:diversion_per_100"]
        assert d["unit"].startswith("abs") and d["estimate"] > 0
    assert res["ppo_minus_baseline_RI"]["fcfs"]["estimate"] == pytest.approx(0.0)
