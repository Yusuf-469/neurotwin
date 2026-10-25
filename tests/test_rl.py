"""RL layer: actions, features, environment semantics and in-house PPO components."""

from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from neurotwin.experiments.objective import CostWeights
from neurotwin.rl.actions import N_ACTIONS, Action, action_mask, action_to_placement
from neurotwin.rl.env import HospitalAllocationEnv
from neurotwin.rl.features import FEATURE_NAMES, N_FEATURES
from neurotwin.rl.ppo import PPOAgent, PPOConfig, train
from neurotwin.rl.ppo.gae import compute_gae
from neurotwin.rl.ppo.losses import ppo_loss
from neurotwin.rl.ppo.masking import masked_distribution
from neurotwin.rl.ppo.networks import ActorCritic
from neurotwin.rl.ppo.normalize import RunningMeanStd
from neurotwin.simulation import SCENARIOS, HospitalSimulation, load_default_config, run_episode

SHORT = load_default_config().with_overrides(horizon_hours=400.0, warmup_hours=100.0)


# ---------------------------------------------------------------- actions and features

def test_masks_cover_exactly_the_feasible_placements():
    sim = HospitalSimulation(SHORT, SCENARIOS["E"], seed=1)
    seen = set()
    rng = np.random.default_rng(0)
    while sim.next_epoch():
        while True:
            mask = action_mask(sim)
            assert mask[Action.HOLD]
            assert mask[1:].any() == sim.has_feasible_placement()
            for a in range(1, N_ACTIONS):
                placement = action_to_placement(sim, a)
                assert (placement is not None) == mask[a]
                if placement is not None:
                    rid, unit = placement
                    view = next(r for r in sim.waiting_requests() if r.rid == rid)
                    assert unit in view.eligible_units and sim.free_staffed(unit) > 0
                    seen.add(a)
            valid = np.flatnonzero(mask[1:]) + 1
            if len(valid) == 0 or rng.random() < 0.2:
                break
            sim.place(*action_to_placement(sim, int(rng.choice(valid))))
    assert seen >= {1, 2, 4, 5}                    # common actions were exercised


def test_off_service_action_only_when_primary_full():
    from neurotwin.policies import PriorityAllocator
    sim = HospitalSimulation(SHORT, SCENARIOS["B"], seed=3)
    policy, checked = PriorityAllocator(off_service_wait_hours=1e9), 0   # never overflows itself
    while sim.next_epoch():
        while sim.has_feasible_placement():
            p = action_to_placement(sim, Action.ADMIT_OFF_SERVICE)
            if p is not None:
                view = next(r for r in sim.waiting_requests() if r.rid == p[0])
                assert sim.free_staffed(view.primary_unit) == 0 and p[1] is not view.primary_unit
                checked += 1
            decision = policy.decide(sim)
            if decision is None:
                break
            sim.place(*decision)
    assert checked > 0


def test_features_shape_bounds_and_names():
    env = HospitalAllocationEnv(SHORT, SCENARIOS["C"])
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(500):
        assert obs.shape == (N_FEATURES,) and np.all(np.isfinite(obs))
        assert env.observation_space.contains(obs)
        valid = np.flatnonzero(env.action_masks())
        obs, _, _, trunc, _ = env.step(int(rng.choice(valid)))
        if trunc:
            break
    assert len(FEATURE_NAMES) == N_FEATURES


# ---------------------------------------------------------------- environment semantics

def _random_masked_episode(env, seed, rng):
    obs, info = env.reset(seed=seed)
    total, steps, zero_dt_after_place = 0.0, 0, 0
    while True:
        valid = np.flatnonzero(info["action_mask"])
        a = int(rng.choice(valid))
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        if a != Action.HOLD and info["dt_hours"] == 0:
            zero_dt_after_place += 1
        if term or trunc:
            return total, steps, zero_dt_after_place


def test_episode_return_equals_minus_cost():
    """Sum of rewards = -(alpha W + beta D + kappa O) over the episode, exactly (§6.10)."""
    env = HospitalAllocationEnv(SHORT, SCENARIOS["B"], reward_scale=0.1)
    total, steps, zero_dt = _random_masked_episode(env, 7, np.random.default_rng(0))
    w, acc = CostWeights.load(), env.sim.acc
    cost = w.alpha * acc.weighted_request_hours + w.beta * acc.diversions + w.kappa * acc.off_service_placements
    assert total == pytest.approx(-0.1 * cost, rel=1e-9)
    assert steps > 100 and zero_dt > 0           # atomic placements take no time


def test_env_is_equivalent_to_direct_simulation():
    """Replaying the env's decisions through run_episode gives the identical run."""
    env = HospitalAllocationEnv(SHORT, SCENARIOS["B"])
    obs, info = env.reset(seed=11)
    rng = np.random.default_rng(2)
    decisions = []
    while True:
        valid = np.flatnonzero(info["action_mask"])
        a = int(rng.choice(valid))
        decisions.append(action_to_placement(env.sim, a))
        obs, r, term, trunc, info = env.step(a)
        if trunc:
            break

    replay = iter(decisions)

    class Replay:
        def decide(self, sim):
            return next(replay)

    direct = run_episode(HospitalSimulation(SHORT, SCENARIOS["B"], seed=11), Replay())
    assert direct.boarding_times == env.sim.result().boarding_times
    assert next(replay, "done") == "done"


def test_masked_action_is_rejected_and_passive_checks_pass():
    env = HospitalAllocationEnv(SHORT)
    env_reset_passive_checker(env, seed=3)
    obs, info = env.reset(seed=3)
    env_step_passive_checker(env, int(np.flatnonzero(info["action_mask"])[0]))
    invalid = np.flatnonzero(~env.action_masks())
    if len(invalid):
        with pytest.raises(ValueError):
            env.step(int(invalid[0]))


def test_warm_mode_rejects_non_training_seeds():
    from neurotwin.digital_twin import WarmStartPool
    from neurotwin.simulation.validation.reduced import GreedyPlacement
    pool = WarmStartPool.generate(SHORT, SCENARIOS["A"], [0], GreedyPlacement())
    env = HospitalAllocationEnv(SHORT, start_mode="warm", pool=pool)
    env.reset(seed=1)
    assert env.sim.now >= SHORT.warmup_hours
    with pytest.raises(ValueError):
        env.reset(seed=200_000)


# ---------------------------------------------------------------- PPO components

def test_gae_matches_hand_computation_with_variable_dt():
    gamma, lam = 0.9, 0.8
    dts = np.array([[1.0], [0.0], [2.0]])
    g = gamma ** dts
    r = np.array([[1.0], [0.5], [-1.0]])
    v = np.array([[0.2], [0.4], [0.1]])
    done = np.array([[0.0], [0.0], [0.0]])
    last_v = np.array([0.3])
    adv, ret = compute_gae(r, v, g, done, last_v, lam)
    d2 = -1.0 + 0.81 * 0.3 - 0.1
    d1 = 0.5 + 1.0 * 0.1 - 0.4                      # dt = 0 -> no discount
    d0 = 1.0 + 0.9 * 0.4 - 0.2
    a2 = d2
    a1 = d1 + 1.0 * lam * a2
    a0 = d0 + 0.9 * lam * a1
    assert adv[:, 0] == pytest.approx([a0, a1, a2])
    assert ret[:, 0] == pytest.approx(np.array([a0, a1, a2]) + v[:, 0])


def test_gae_stops_at_episode_boundary():
    r = np.array([[1.0], [5.0]])
    v = np.zeros((2, 1))
    adv, _ = compute_gae(r, v, np.full((2, 1), 0.99), np.array([[1.0], [0.0]]), np.array([0.0]), 0.95)
    assert adv[0, 0] == pytest.approx(1.0)          # nothing leaks across the boundary


def test_clipped_loss_equals_plain_surrogate_inside_trust_region():
    torch.manual_seed(0)
    old = torch.randn(64)
    new = old + 0.05 * torch.randn(64).clamp(-1, 1)          # |ratio - 1| < 0.2
    adv, vals = torch.randn(64), torch.zeros(64)
    loss, info = ppo_loss(new, old, adv, vals, vals, vals, torch.zeros(64), clip=0.2,
                          vf_clip=None, ent_coef=0.0, vf_coef=0.0)
    assert loss.item() == pytest.approx((-adv * (new - old).exp()).mean().item(), rel=1e-6)
    assert info.clip_fraction == 0.0


def test_clipping_removes_gradient_beyond_trust_region():
    old = torch.zeros(1)
    new = torch.tensor([0.5], requires_grad=True)             # ratio e^0.5 > 1.2
    loss, _ = ppo_loss(new, old, torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                       torch.zeros(1), clip=0.2, vf_clip=None, ent_coef=0.0, vf_coef=0.0)
    loss.backward()
    assert new.grad.item() == 0.0


def test_masked_actions_have_zero_probability_and_gradient():
    logits = torch.randn(4, N_ACTIONS, requires_grad=True)
    mask = torch.ones(4, N_ACTIONS, dtype=torch.bool)
    mask[:, 3] = False
    mask[1, 5] = False
    dist = masked_distribution(logits, mask)
    assert torch.all(dist.probs[~mask] == 0)
    assert torch.isfinite(dist.entropy()).all()
    actions = dist.sample((1000,))
    assert not (actions == 3).any()
    dist.entropy().sum().backward()
    assert torch.all(logits.grad[~mask] == 0)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 5))
    rms = RunningMeanStd((5,), epsilon=0.0)
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    assert rms.mean == pytest.approx(data.mean(axis=0))
    assert rms.var == pytest.approx(data.var(axis=0))


def test_agent_save_load_roundtrip(tmp_path):
    agent = PPOAgent(ActorCritic(N_FEATURES, N_ACTIONS), RunningMeanStd((N_FEATURES,)))
    agent.obs_rms.update(np.random.default_rng(0).normal(size=(50, N_FEATURES)))
    agent.save(tmp_path / "a.pt")
    loaded = PPOAgent.load(tmp_path / "a.pt")
    x = np.random.default_rng(1).normal(size=(20, N_FEATURES)).astype(np.float32)
    mask = np.ones(N_ACTIONS, dtype=bool)
    assert [agent.act(o, mask) for o in x] == [loaded.act(o, mask) for o in x]


def test_ppo_policy_makes_only_valid_decisions():
    from neurotwin.policies.ppo_policy import PPOPolicy
    agent = PPOAgent(ActorCritic(N_FEATURES, N_ACTIONS), RunningMeanStd((N_FEATURES,)))

    class Checked(PPOPolicy):
        def decide(self, sim):
            d = super().decide(sim)
            if d is not None:
                view = next(r for r in sim.waiting_requests() if r.rid == d[0])
                assert d[1] in view.eligible_units and sim.free_staffed(d[1]) > 0
            return d

    result = run_episode(HospitalSimulation(SHORT, SCENARIOS["B"], seed=2), Checked(agent))
    assert result.boarding_times


def test_training_episode_seeds_stay_in_train_range_and_are_disjoint():
    from neurotwin.experiments.seeds import TRAIN
    from neurotwin.experiments.train_ppo import EPISODE_SEED_BLOCK, episode_seeds
    blocks = [set(itertools.islice(episode_seeds(k), EPISODE_SEED_BLOCK)) for k in range(5)]
    assert all(min(b) in TRAIN and max(b) in TRAIN for b in blocks)
    assert all(not (a & b) for a, b in itertools.combinations(blocks, 2))


class _TwoHourEnv(gym.Env):
    """Dummy env whose every step lasts dt = 2 hours (truncated after 5 steps)."""
    observation_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(3, np.float32), 1.0, False, self.t >= 5, {"dt_hours": 2.0}


@pytest.mark.parametrize("semi_markov,expected", [(True, 0.9 ** 2), (False, 0.9)])
def test_trainer_discounts_by_elapsed_time(monkeypatch, semi_markov, expected):
    import neurotwin.rl.ppo.trainer as trainer_module
    captured = {}
    real_gae = trainer_module.compute_gae

    def spy(rewards, values, gammas, dones, last_values, lam):
        captured["gammas"] = gammas.copy()
        return real_gae(rewards, values, gammas, dones, last_values, lam)

    monkeypatch.setattr(trainer_module, "compute_gae", spy)
    config = PPOConfig(total_steps=16, n_envs=1, n_steps=16, n_minibatches=2, n_epochs=1,
                       gamma=0.9, semi_markov=semi_markov)
    train([_TwoHourEnv], config, itertools.count(0))
    assert np.allclose(captured["gammas"], expected)


@pytest.mark.slow
def test_ppo_learns_cartpole():
    """Smoke test: the in-house PPO must clearly learn CartPole (full comparison with
    stable-baselines3 is in neurotwin.rl.ppo.validate)."""
    config = PPOConfig(total_steps=60_000, n_envs=4, n_steps=128, n_epochs=4, n_minibatches=4,
                       learning_rate=2.5e-4, vf_clip=None, normalize_obs=False, seed=1)
    seeds = itertools.count(0)
    _, log = train([lambda: gym.make("CartPole-v1")] * 4, config, seeds)
    final = np.mean([r for _, r in log.episode_returns[-20:]])
    assert final > 150, final
