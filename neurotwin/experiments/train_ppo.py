"""Train K independent PPO agents and evaluate them on validation seeds (RESEARCH.md v2 §6.12).

* Training: scenario A, warm starts from a pool of MIN_TRAINING_POOL distinct states generated
  from TRAIN seeds; each agent k draws its episode seeds from its own disjoint block inside the
  TRAIN range. Agent seed k sets network initialisation, sampling and minibatch order.
* Evaluation: deterministic policy, full protocol from an empty start (49 days), scenario A,
  VALIDATION seeds only. This is used for hyperparameter selection; test seeds are never touched
  here (asserted).
* Outputs (results/ppo/<name>/, git-ignored): agent_k.pt checkpoints, training logs and a
  summary JSON with J per validation seed for every agent and for the tuned Priority baseline.

Run: python -m neurotwin.experiments.train_ppo --name g90 [--agents 5] [--set gamma=0.9 ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..digital_twin.warm_start import MIN_TRAINING_POOL, WarmStartPool
from ..policies import load_baseline_params, make_policy
from ..policies.ppo_policy import PPOPolicy
from ..rl.env import HospitalAllocationEnv
from ..rl.ppo import PPOConfig, train
from ..simulation.config import SCENARIOS, load_default_config
from ..simulation.hospital import HospitalSimulation, run_episode
from ..simulation.validation.reduced import GreedyPlacement
from .objective import CostWeights, measured_cost
from .seeds import TRAIN, VALIDATION, require_split

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "ppo"
PPO_CONFIG = ROOT / "config" / "ppo_default.json"
EPISODE_SEED_BLOCK = 15_000  # episode seeds per agent (k-th block starts at 10_000 + k*block)


def load_ppo_config(overrides: dict[str, Any] | None = None, seed: int = 0) -> PPOConfig:
    """PPOConfig from config/ppo_default.json with optional overrides."""
    data = {
        k: v
        for k, v in json.loads(PPO_CONFIG.read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }  # keys starting with '_' are comments
    data.update(overrides or {})
    data["hidden"] = tuple(data["hidden"])
    return PPOConfig(seed=seed, **data)


def episode_seeds(agent: int):
    """Disjoint block of TRAIN-range episode seeds for agent ``agent``."""
    start = 10_000 + agent * EPISODE_SEED_BLOCK
    block = range(start, start + EPISODE_SEED_BLOCK)
    require_split([block.start, block.stop - 1], "train")
    return iter(block)


def evaluate(policy, seeds=VALIDATION) -> list[float]:
    """J per measured day for ``policy`` on validation seeds (scenario A)."""
    seeds = list(seeds)
    require_split(seeds, "validation")
    config, weights = load_default_config(), CostWeights.load()
    costs = []
    for s in seeds:
        sim = HospitalSimulation(config, SCENARIOS["A"], seed=s)
        run_episode(sim, policy)
        costs.append(measured_cost(sim, weights))
    return costs


def train_agent(args: tuple) -> dict[str, Any]:
    """args = (name, agent_seed, ppo_overrides[, env_opts]).

    env_opts (ablations, RESEARCH.md §6.16): ``drop_features`` (0-based indices), ``beta``,
    ``kappa`` (reward weights) and ``uniform_omega`` (all priority weights 1 in the *training*
    reward). Evaluation always uses the standard cost J.
    """
    name, agent_seed, overrides = args[:3]
    env_opts = args[3] if len(args) > 3 else {}
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    config = load_ppo_config(overrides, seed=agent_seed)
    sim_config = load_default_config()
    if env_opts.get("uniform_omega"):
        sim_config = sim_config.with_overrides(priority_weights=[1.0] * 5)
    base_w = CostWeights.load()
    weights = CostWeights(
        base_w.alpha, env_opts.get("beta", base_w.beta), env_opts.get("kappa", base_w.kappa)
    )
    drop = tuple(env_opts.get("drop_features", ()))
    sampler = None
    if env_opts.get("domain_randomization"):
        from .ppo_dr import check_training_range, sample_scenario  # local: avoids import cycle

        check_training_range()
        sampler = sample_scenario
    pool = WarmStartPool.generate(
        sim_config, SCENARIOS["A"], range(MIN_TRAINING_POOL), GreedyPlacement()
    )
    t0 = time.perf_counter()
    agent, log = train(
        [
            lambda: HospitalAllocationEnv(
                sim_config,
                start_mode="warm",
                pool=pool,
                weights=weights,
                drop_features=drop,
                scenario_sampler=sampler,
                lambda_conditioned=bool(env_opts.get("lambda_conditioned")),
            )
        ]
        * config.n_envs,
        config,
        episode_seeds(agent_seed),
    )
    train_s = time.perf_counter() - t0
    agent.meta.update(
        {
            "name": name,
            "agent_seed": agent_seed,
            "train_seconds": train_s,
            "drop_features": list(drop),
            "env_opts": env_opts,
        }
    )
    agent.save(out / f"agent_{agent_seed}.pt")
    (out / f"log_{agent_seed}.json").write_text(
        json.dumps({"updates": log.updates, "episode_returns": log.episode_returns}),
        encoding="utf-8",
    )
    if env_opts.get("lambda_conditioned"):
        from ..policies.constrained_ppo import ConstrainedPPOPolicy  # 22-input policy + controller

        costs = evaluate(ConstrainedPPOPolicy(agent))
    else:
        costs = evaluate(PPOPolicy(agent))
    return {
        "agent_seed": agent_seed,
        "validation_J": costs,
        "mean_J": float(np.mean(costs)),
        "train_seconds": train_s,
    }


def _baseline_job(name: str) -> list[float]:
    return evaluate(make_policy(name, load_baseline_params()[name]))


def main() -> None:
    """Command-line entry point: train K agents, evaluate on validation seeds."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--set", nargs="*", default=[], help="override, e.g. gamma=0.95")
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    require_split(range(MIN_TRAINING_POOL), "train")

    with ProcessPoolExecutor(args.workers) as ex:
        base = ex.submit(_baseline_job, "priority")
        runs = list(ex.map(train_agent, [(args.name, k, overrides) for k in range(args.agents)]))
        priority = base.result()
    agent_means = [r["mean_J"] for r in runs]
    paired = [float(np.mean(np.array(r["validation_J"]) - np.array(priority))) for r in runs]
    summary = {
        "name": args.name,
        "overrides": overrides,
        "config": vars(load_ppo_config(overrides))
        | {"hidden": list(load_ppo_config(overrides).hidden)},
        "runs": runs,
        "priority_validation_J": priority,
        "mean_J_over_agents": float(np.mean(agent_means)),
        "paired_diff_vs_priority_per_agent": paired,
    }
    (RESULTS / args.name / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"{args.name} {overrides}: agent mean J {[round(m, 1) for m in agent_means]} "
        f"(overall {np.mean(agent_means):.1f}); Priority {np.mean(priority):.1f}; "
        f"paired PPO-Priority per agent {[round(p, 1) for p in paired]}"
    )


if __name__ == "__main__":
    main()
