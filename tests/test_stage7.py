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


def test_dual_ascent_step_moves_toward_budget_and_is_projected():
    from neurotwin.rl.constrained import BUDGET_PER_100, ETA, LAMBDA_MAX, dual_ascent_step

    assert dual_ascent_step(20.0, BUDGET_PER_100) == 20.0  # on budget: unchanged
    assert dual_ascent_step(20.0, BUDGET_PER_100 + 2) == 20.0 + 2 * ETA
    assert dual_ascent_step(20.0, 0.0) == 20.0 - BUDGET_PER_100 * ETA
    assert dual_ascent_step(1.0, 0.0) == 0.0  # projected onto [0, LAMBDA_MAX]
    assert dual_ascent_step(LAMBDA_MAX - 1, 50.0) == LAMBDA_MAX


def test_lambda_conditioned_env_obs_and_reward():
    from neurotwin.rl.constrained import LAMBDA_MAX
    from neurotwin.rl.features import N_FEATURES

    env = HospitalAllocationEnv(SHORT, SCENARIOS["E"], lambda_conditioned=True)
    assert env.observation_space.shape == (N_FEATURES + 1,)
    obs, info = env.reset(seed=7)
    lam = env._lam
    assert 0.0 <= lam <= LAMBDA_MAX and obs.shape == (N_FEATURES + 1,)
    assert obs[-1] == pytest.approx(lam / LAMBDA_MAX)
    env.reset(seed=8)
    assert env._lam != lam  # a new draw per episode
    env.reset(seed=7)
    assert env._lam == lam  # reproducible from the episode seed

    # Reward uses λ in place of β: total return equals -(αW + λD + κO) over the episode.
    w = env.weights
    total, truncated = 0.0, False
    while not truncated:  # always HOLD: forces diversions, so λ enters the return
        obs, r, _, truncated, info = env.step(0)
        assert obs[-1] == pytest.approx(lam / LAMBDA_MAX)
        total += r
    acc = env.sim.acc
    expected = (
        -(
            w.alpha * acc.weighted_request_hours
            + lam * acc.diversions
            + w.kappa * acc.off_service_placements
        )
        * env.reward_scale
    )
    assert acc.diversions > 0  # λ actually matters
    assert total == pytest.approx(expected, rel=1e-6)


class _RecordingAgent:
    """Stub agent: always HOLD, records observations."""

    meta: dict = {}

    def __init__(self):
        self.obs = []

    def act(self, obs, mask=None, deterministic=True):
        self.obs.append(obs)
        return 0


def test_constrained_policy_updates_lambda_daily():
    from neurotwin.policies.constrained_ppo import ConstrainedPPOPolicy
    from neurotwin.rl.constrained import LAMBDA0, LAMBDA_MAX
    from neurotwin.rl.features import N_FEATURES
    from neurotwin.simulation.hospital import HospitalSimulation, run_episode

    agent = _RecordingAgent()
    policy = ConstrainedPPOPolicy(agent)
    sim = HospitalSimulation(SHORT, SCENARIOS["E"], seed=2)
    run_episode(sim, policy)  # always HOLD -> many diversions -> λ must rise
    assert all(o.shape == (N_FEATURES + 1,) for o in agent.obs)
    times = [t for t, _ in policy.lambda_trace]
    assert policy.lambda_trace[0][1] == LAMBDA0
    assert len(times) >= SHORT.horizon_hours // 24 - 1
    assert policy.lam > LAMBDA0 and policy.lam <= LAMBDA_MAX
    assert agent.obs[-1][-1] == pytest.approx(policy.lam / LAMBDA_MAX)


def test_shield_mask_removes_hold_only_when_ed_full_and_placement_feasible():
    from types import SimpleNamespace

    from neurotwin.policies.ppo_policy import PPOPolicy

    mask = np.array([True, True, False, False, False, False, False])
    full = SimpleNamespace(config=SimpleNamespace(ed_capacity=30), ed_count=30)
    not_full = SimpleNamespace(config=SimpleNamespace(ed_capacity=30), ed_count=29)
    unlimited = SimpleNamespace(config=SimpleNamespace(ed_capacity=None), ed_count=99)
    shielded, applied = PPOPolicy.shield_mask(full, mask)
    assert applied and not shielded[0] and shielded[1] and mask[0]  # original mask untouched
    for sim in (not_full, unlimited):
        same, applied = PPOPolicy.shield_mask(sim, mask)
        assert not applied and same[0]
    only_hold = np.array([True] + [False] * 6)
    same, applied = PPOPolicy.shield_mask(full, only_hold)
    assert not applied and same[0]  # nothing feasible: HOLD stays the only option


def test_shielded_agent_never_holds_while_ed_full():
    from neurotwin.experiments.evaluate import DEFAULT_AGENTS
    from neurotwin.policies.ppo_policy import PPOPolicy
    from neurotwin.rl.actions import action_mask
    from neurotwin.simulation.hospital import HospitalSimulation, run_episode

    policy = PPOPolicy.load(DEFAULT_AGENTS / "agent_0.pt", shield=True)
    violations = []
    decide = policy.decide

    def checked(sim):
        full = sim.ed_count >= sim.config.ed_capacity and action_mask(sim)[1:].any()
        result = decide(sim)
        if full and result is None:
            violations.append(sim.now)
        return result

    policy.decide = checked
    from neurotwin.simulation.config import SEVERE_SCENARIOS

    sim = HospitalSimulation(SHORT, SEVERE_SCENARIOS["E_severe"], seed=4)
    run_episode(sim, policy)
    assert policy.shield_activations > 0  # the overload actually triggered the shield
    assert violations == []


def test_v3_training_range_covers_every_stress_scenario():
    from neurotwin.experiments.v3 import RANGE, sample_wide

    for key in "BCDE":
        sc = SCENARIOS[key]
        assert RANGE["arrival"][0] <= sc.arrival_multiplier <= RANGE["arrival"][1]
        assert RANGE["staff"][0] <= sc.staff_multiplier <= RANGE["staff"][1]
        assert RANGE["capacity"][0] <= sc.capacity_multiplier <= RANGE["capacity"][1]
    rng = np.random.default_rng(0)
    draws = [sample_wide(rng) for _ in range(500)]
    assert min(d.arrival_multiplier for d in draws) >= 0.95 and max(d.arrival_multiplier for d in draws) <= 1.15
    assert min(d.staff_multiplier for d in draws) < 0.87  # actually reaches crisis staffing
