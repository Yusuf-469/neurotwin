"""Validation gates for the in-house PPO against stable-baselines3 (RESEARCH.md v2 §6.18).

Gate 2 (``cartpole``): CartPole-v1, 5 seeds each, matched hyperparameters (the standard
CartPole settings used in CleanRL and the "37 details" post): 4 envs, 128-step rollouts, 4
epochs, 4 minibatches, lr 2.5e-4 annealed, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01,
no value clipping, no observation normalisation, tanh 64-64 separate actor and critic.

Gate 3 (``hospital``): the hospital environment (scenario A, warm starts from training seeds),
in-house PPO vs sb3-contrib MaskablePPO with matched hyperparameters and ordinary per-step
discounting (semi_markov=False), because the reference does not support gamma ** dt.

Both gates compare learning curves (mean episode return in step bins) and final performance
(mean of each run's last 20 episodes) with 95% CIs over seeds.

Run: python -m neurotwin.rl.ppo.validate cartpole|hospital [--steps N] [--seeds 5]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy import stats

from .trainer import PPOConfig, train

RESULTS = Path(__file__).resolve().parents[3] / "results"
CARTPOLE = dict(n_envs=4, n_steps=128, n_epochs=4, n_minibatches=4, learning_rate=2.5e-4,
                gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, vf_clip=None, normalize_obs=False)
# Settings in the range where PPO learns on this env (development sweep, stage-5 report);
# per-step discounting and no observation normalisation to match the reference exactly.
HOSPITAL = dict(n_envs=8, n_steps=256, n_epochs=10, n_minibatches=8, learning_rate=3e-4,
                gamma=0.95, gae_lambda=0.9, clip=0.2, ent_coef=0.001, vf_coef=0.5,
                max_grad_norm=0.5, vf_clip=None, normalize_obs=False, semi_markov=False)


# ---------------------------------------------------------------- environment factories

def _cartpole():
    return gym.make("CartPole-v1")


_POOL = None


def _hospital():
    from ...digital_twin.warm_start import MIN_TRAINING_POOL, WarmStartPool
    from ...simulation.config import SCENARIOS, load_default_config
    from ...simulation.validation.reduced import GreedyPlacement
    from ..env import HospitalAllocationEnv
    global _POOL
    if _POOL is None:
        _POOL = WarmStartPool.generate(load_default_config(), SCENARIOS["A"],
                                       range(MIN_TRAINING_POOL), GreedyPlacement())
    return HospitalAllocationEnv(start_mode="warm", pool=_POOL)


# ---------------------------------------------------------------- runs

def run_ours(args) -> list[tuple[int, float]]:
    task, seed, steps = args
    hp = CARTPOLE if task == "cartpole" else HOSPITAL
    make = _cartpole if task == "cartpole" else _hospital
    config = PPOConfig(total_steps=steps, seed=seed, **hp)
    # Episode seeds: disjoint per run, inside the TRAIN range for the hospital env.
    seeds = itertools.count(seed * 10_000)
    _, log = train([make] * hp["n_envs"], config, seeds)
    return log.episode_returns


def run_reference(args) -> list[tuple[int, float]]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    task, seed, steps = args
    torch.set_num_threads(1)
    hp = CARTPOLE if task == "cartpole" else HOSPITAL
    returns: list[tuple[int, float]] = []

    class Recorder(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" in info:
                    returns.append((self.num_timesteps, float(info["episode"]["r"])))
            return True

    lr = hp["learning_rate"]
    kwargs = dict(n_steps=hp["n_steps"], batch_size=hp["n_envs"] * hp["n_steps"] // hp["n_minibatches"],
                  n_epochs=hp["n_epochs"], learning_rate=lambda progress: lr * progress,
                  gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip"],
                  ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
                  seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                     activation_fn=torch.nn.Tanh))
    if task == "cartpole":
        env = make_vec_env("CartPole-v1", n_envs=hp["n_envs"], seed=seed)
        model = PPO("MlpPolicy", env, **kwargs)
    else:
        from sb3_contrib import MaskablePPO
        env = make_vec_env(_hospital, n_envs=hp["n_envs"], seed=seed * 10_000)
        model = MaskablePPO("MlpPolicy", env, **kwargs)
    model.learn(total_timesteps=steps, callback=Recorder())
    return returns


# ---------------------------------------------------------------- analysis

def _final(curve: list[tuple[int, float]]) -> float:
    return float(np.mean([r for _, r in curve[-20:]]))


def _binned(curve, steps, bins=10) -> list[float]:
    edges = np.linspace(0, steps, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [r for s, r in curve if lo < s <= hi]
        out.append(float(np.mean(vals)) if vals else math.nan)
    return out


def _ci(x) -> str:
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{x.mean():.2f} [{x.mean() - h:.2f}, {x.mean() + h:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=["cartpole", "hospital"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    steps = args.steps or (200_000 if args.task == "cartpole" else 300_000)

    t0 = time.perf_counter()
    jobs = [(args.task, s, steps) for s in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        ours_f = [pool.submit(run_ours, j) for j in jobs]
        ref_f = [pool.submit(run_reference, j) for j in jobs]
        ours = [f.result() for f in ours_f]
        ref = [f.result() for f in ref_f]

    final_ours, final_ref = [_final(c) for c in ours], [_final(c) for c in ref]
    p = stats.ttest_ind(final_ours, final_ref, equal_var=False).pvalue
    print(f"Task {args.task}: {steps} steps x {args.seeds} seeds ({time.perf_counter() - t0:.0f} s)")
    print(f"  final return (last 20 episodes), mean [95% CI over seeds]:")
    print(f"    in-house : {_ci(final_ours)}   per seed {[round(v, 1) for v in final_ours]}")
    print(f"    reference: {_ci(final_ref)}   per seed {[round(v, 1) for v in final_ref]}")
    print(f"    Welch t-test p = {p:.3g}")
    print("  learning curve (mean return per 10% of training, averaged over seeds):")
    print("    in-house : " + " ".join(f"{v:8.1f}" for v in np.nanmean([_binned(c, steps) for c in ours], axis=0)))
    print("    reference: " + " ".join(f"{v:8.1f}" for v in np.nanmean([_binned(c, steps) for c in ref], axis=0)))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ppo_validation_{args.task}.json").write_text(json.dumps(
        {"steps": steps, "ours": ours, "reference": ref, "p_final": p}), encoding="utf-8")


if __name__ == "__main__":
    main()
