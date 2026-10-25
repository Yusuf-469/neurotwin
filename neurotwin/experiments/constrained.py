"""λ-conditioned PPO with online dual ascent vs fixed-β PPO (RESEARCH.md v2.6 §6.19).

All constants are pre-registered in neurotwin/rl/constrained.py. Training: scenario A, training
seeds, same hyperparameters, agent seeds and episode seeds as the final PPO (K = 5). Evaluation:
test seeds, scenarios A-E, paired with fixed-β PPO (final_A); Holm over the family.

Run: python -m neurotwin.experiments.constrained train --workers 5
     python -m neurotwin.experiments.constrained evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..rl.constrained import BUDGET_PER_100
from .evaluate import ROOT, run_job
from .seeds import TEST
from .statistics import apply_holm, compare
from .train_ppo import RESULTS, train_agent

K = 5
OUT = ROOT / "docs" / "data" / "stage7"
METRICS = (
    "diversion_per_100",
    "mean_boarding_h",
    "p90_boarding_h",
    "off_service_per_100",
    "J_per_day",
)


def train_all(workers: int) -> None:
    """Train K λ-conditioned agents (same seeds as the final PPO)."""
    jobs = [("ppo_lagrangian", k, {}, {"lambda_conditioned": True}) for k in range(K)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(train_agent, jobs):
            print(f"agent {r['agent_seed']} validation J {r['mean_J']:.1f}", flush=True)
    print(f"trained {len(jobs)} agents in {(time.perf_counter() - t0) / 60:.0f} min")


def evaluate_all(workers: int) -> None:
    """λ-PPO vs fixed-β PPO on every test seed and scenario; write rows and comparisons."""
    seeds = list(TEST)
    variants = {"ppo": RESULTS / "final_A", "ppo_lagrangian": RESULTS / "ppo_lagrangian"}
    jobs = [
        (name, a, scen, s, str(d), {})
        for name, d in variants.items()
        for a in range(K)
        for scen in "ABCDE"
        for s in seeds
    ]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=4))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "constrained_episodes.json").write_text(json.dumps(rows), encoding="utf-8")

    def matrix(policy: str, scen: str, metric: str) -> np.ndarray:
        idx = {
            (int(r["agent"]), int(r["seed"])): r[metric]
            for r in rows
            if r["policy"] == policy and r["scenario"] == scen
        }
        return np.array([[idx[(a, s)] for s in seeds] for a in range(K)])

    comps, budget = [], {}
    for scen in "ABCDE":
        for metric in METRICS:
            comps.append(
                compare(
                    scen,
                    metric,
                    "ppo",
                    matrix("ppo_lagrangian", scen, metric),
                    matrix("ppo", scen, metric).mean(0),
                )
            )
        for policy in variants:
            d = matrix(policy, scen, "diversion_per_100")
            budget[f"{scen}:{policy}"] = float(np.mean(d <= BUDGET_PER_100))
        lam = [
            r["lambda_mean_measured"]
            for r in rows
            if r["policy"] == "ppo_lagrangian" and r["scenario"] == scen
        ]
        budget[f"{scen}:lambda_mean"] = float(np.mean(lam))
    apply_holm(comps)
    (OUT / "constrained_comparisons.json").write_text(
        json.dumps({"comparisons": [c.as_dict() for c in comps], "budget": budget}, indent=1),
        encoding="utf-8",
    )
    for c in comps:
        flag = "*" if c.significant_holm else ""
        print(
            f"{c.scenario} {c.metric:20s} lambda-PPO - PPO {c.diff_mean:9.2f} "
            f"[{c.ci_low:9.2f}, {c.ci_high:9.2f}] {flag}"
        )
    print(json.dumps(budget, indent=1))


def main() -> None:
    """Command-line entry point (train | evaluate)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    (train_all if args.command == "train" else evaluate_all)(args.workers)


if __name__ == "__main__":
    main()
