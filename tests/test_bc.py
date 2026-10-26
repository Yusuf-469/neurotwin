"""Behaviour cloning, ensembles and PPO warm starts (RESEARCH.md v2.8 §6.26)."""

import numpy as np
import pytest
import torch

from neurotwin.policies import load_baseline_params, make_policy
from neurotwin.rl.actions import Action
from neurotwin.simulation import SCENARIOS, load_default_config

SHORT = load_default_config().with_overrides(horizon_hours=240.0, warmup_hours=48.0)


@pytest.fixture(scope="module")
def demos():
    from neurotwin.rl.bc import collect

    teacher = make_policy("load_balancing", load_baseline_params()["load_balancing"])
    return collect(teacher, SHORT, [11, 12])


def test_demonstrations_reproduce_the_teacher(demos):
    from neurotwin.rl.actions import N_ACTIONS

    assert len(demos.action) > 200
    assert demos.obs.shape == (len(demos.action), 21) and demos.mask.shape[1] == N_ACTIONS
    assert all(demos.mask[i, a] for i, a in enumerate(demos.action))  # recorded action is valid
    assert demos.skipped < 0.1 * len(demos.action)


def test_collect_refuses_non_training_seeds():
    from neurotwin.rl.bc import collect

    teacher = make_policy("load_balancing", load_baseline_params()["load_balancing"])
    with pytest.raises(ValueError):
        collect(teacher, SHORT, [200_000])  # a test seed


def test_clone_learns_the_teacher(demos):
    from neurotwin.rl.bc import clone

    agent, stats = clone(demos, seed=0, epochs=40)
    majority = np.bincount(demos.action).max() / len(demos.action)
    assert stats["train_accuracy"] > majority + 0.05  # better than always predicting the commonest
    assert stats["holdout_accuracy"] > 0.8
    assert agent.obs_rms.frozen


def test_ensemble_of_identical_agents_matches_single_agent():
    from neurotwin.experiments.evaluate import DEFAULT_AGENTS
    from neurotwin.policies.ensemble_ppo import EnsemblePPOPolicy
    from neurotwin.policies.ppo_policy import PPOPolicy
    from neurotwin.simulation.hospital import HospitalSimulation, run_until

    path = DEFAULT_AGENTS / "agent_1.pt"
    single, ens = PPOPolicy.load(path), EnsemblePPOPolicy.load([path, path, path])
    sim_a = HospitalSimulation(SHORT, SCENARIOS["A"], seed=5)
    sim_b = HospitalSimulation(SHORT, SCENARIOS["A"], seed=5)
    run_until(sim_a, single, until=120.0)
    run_until(sim_b, ens, until=120.0)
    assert sim_a.acc == sim_b.acc  # identical decisions give identical trajectories


def test_warm_start_and_critic_warmup_keep_the_initial_policy(demos):
    from neurotwin.digital_twin.warm_start import WarmStartPool
    from neurotwin.rl.bc import clone
    from neurotwin.rl.env import HospitalAllocationEnv
    from neurotwin.rl.ppo import PPOConfig, train
    from neurotwin.simulation.validation.reduced import GreedyPlacement

    init, _ = clone(demos, seed=0, epochs=5)
    pool = WarmStartPool.generate(SHORT, SCENARIOS["A"], [0, 1], GreedyPlacement())
    cfg = PPOConfig(
        total_steps=256,
        n_envs=2,
        n_steps=64,
        critic_warmup_updates=10,
        freeze_obs_norm=True,
        learning_rate=1e-3,
        gamma=0.95,
        hidden=(64, 64),
    )
    agent, _ = train(
        [lambda: HospitalAllocationEnv(SHORT, start_mode="warm", pool=pool)] * 2,
        cfg,
        iter(range(10, 1000)),
        init_agent=init,
    )
    for p0, p1 in zip(init.model.actor.parameters(), agent.model.actor.parameters()):
        assert torch.equal(p0, p1)  # actor untouched during the critic-only warm-up
    assert not all(
        torch.equal(p0, p1)
        for p0, p1 in zip(init.model.critic.parameters(), agent.model.critic.parameters())
    )
    assert np.allclose(agent.obs_rms.mean, init.obs_rms.mean)  # normaliser frozen
    assert Action.HOLD == 0
