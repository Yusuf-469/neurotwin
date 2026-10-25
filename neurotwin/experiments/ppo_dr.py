"""PPO-DR: domain-randomised training (RESEARCH.md v2 §6.23, RQ8).

Each training episode samples operating conditions uniformly from a range strictly milder than
every test scenario:

    arrivals 0.95-1.04, staff 0.97-1.0, physical capacity 0.97-1.0

Two leakage guards are checked in code before training (``check_training_range``):
1. every test scenario B-E lies outside the range on every perturbed axis;
2. the worst corner of the range has lower analytical offered load (general and ICU) than the
   mildest test scenario.
Everything else (hyperparameters, K = 5 agents, agent seeds, episode seeds) is identical to the
final PPO, so PPO-DR vs PPO isolates the effect of training diversity. Both are evaluated on the
same test seeds in A-E.

Run: python -m neurotwin.experiments.ppo_dr train --workers 5
     python -m neurotwin.experiments.ppo_dr evaluate
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..simulation.config import SCENARIOS, ScenarioConfig, load_default_config
from .evaluate import ROOT, run_job
from .load import offered_load
from .seeds import TEST
from .statistics import apply_holm, compare
from .train_ppo import RESULTS, train_agent

ARRIVALS = (0.95, 1.04)
STAFF = (0.97, 1.0)
CAPACITY = (0.97, 1.0)
K_DR = 5
OUT = ROOT / "docs" / "data" / "stage7"
METRICS = (
    "J_per_day",
    "mean_boarding_h",
    "p90_boarding_h",
    "diversion_per_100",
    "off_service_per_100",
)


def sample_scenario(rng: np.random.Generator) -> ScenarioConfig:
    """One training scenario drawn uniformly from the PPO-DR range."""
    return ScenarioConfig(
        "dr_train",
        arrival_multiplier=float(rng.uniform(*ARRIVALS)),
        staff_multiplier=float(rng.uniform(*STAFF)),
        capacity_multiplier=float(rng.uniform(*CAPACITY)),
    )


def check_training_range() -> dict:
    """Raise if the training range is not strictly milder than every test scenario."""
    for key in "BCDE":
        s = SCENARIOS[key]
        perturbed = [
            (s.arrival_multiplier != 1.0, s.arrival_multiplier <= ARRIVALS[1]),
            (s.staff_multiplier != 1.0, s.staff_multiplier >= STAFF[0]),
            (s.capacity_multiplier != 1.0, s.capacity_multiplier >= CAPACITY[0]),
        ]
        if any(is_perturbed and inside for is_perturbed, inside in perturbed):
            raise ValueError(f"test scenario {key} overlaps the PPO-DR training range")
    cfg = load_default_config()
    corners = [
        ScenarioConfig("corner", a, s, c)
        for a, s, c in itertools.product(ARRIVALS, STAFF, CAPACITY)
    ]
    worst = {k: max(offered_load(cfg, c)[k] for c in corners) for k in ("rho_general", "rho_ICU")}
    mildest = {k: min(offered_load(cfg, SCENARIOS[s])[k] for s in "BCDE") for k in worst}
    if any(worst[k] >= mildest[k] for k in worst):
        raise ValueError(f"PPO-DR worst corner {worst} not below mildest test load {mildest}")
    return {"worst_corner": worst, "mildest_test": mildest}


def train_all(workers: int) -> None:
    """Train K PPO-DR agents (same agent seeds and episode seeds as the final PPO)."""
    print("leakage guards:", check_training_range(), flush=True)
    jobs = [("ppo_dr", k, {}, {"domain_randomization": True}) for k in range(K_DR)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(train_agent, jobs):
            print(f"agent {r['agent_seed']} validation J {r['mean_J']:.1f}", flush=True)
    print(f"trained {len(jobs)} agents in {(time.perf_counter() - t0) / 60:.0f} min")


def evaluate_all(workers: int) -> None:
    """PPO-DR vs PPO on the test seeds in every scenario (paired, Holm over the family)."""
    seeds = list(TEST)
    variants = {"ppo": RESULTS / "final_A", "ppo_dr": RESULTS / "ppo_dr"}
    jobs, names = [], []
    for name, d in variants.items():
        for a in range(K_DR):
            for scen in "ABCDE":
                for s in seeds:
                    jobs.append(("ppo", a, scen, s, str(d), {}))
                    names.append(name)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=4))
    for row, name in zip(rows, names):
        row["variant"] = name
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppo_dr_episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    comps = []
    for scen in "ABCDE":
        for metric in METRICS:

            def matrix(variant):
                return np.array(
                    [
                        [
                            next(
                                r[metric]
                                for r in rows
                                if r["variant"] == variant
                                and r["scenario"] == scen
                                and int(r["agent"]) == a
                                and int(r["seed"]) == s
                            )
                            for s in seeds
                        ]
                        for a in range(K_DR)
                    ]
                )

            # PPO-DR minus PPO; the "baseline" is PPO averaged over its agents per seed.
            comps.append(compare(scen, metric, "ppo", matrix("ppo_dr"), matrix("ppo").mean(0)))
    apply_holm(comps)
    (OUT / "ppo_dr_comparisons.json").write_text(
        json.dumps([c.as_dict() for c in comps], indent=1), encoding="utf-8"
    )
    for c in comps:
        flag = "*" if c.significant_holm else ""
        print(
            f"{c.scenario} {c.metric:20s} PPO-DR - PPO {c.diff_mean:9.2f} "
            f"[{c.ci_low:9.2f}, {c.ci_high:9.2f}] {flag}"
        )


def main() -> None:
    """Command-line entry point (check | train | evaluate)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check", "train", "evaluate"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.command == "check":
        print(check_training_range())
    elif args.command == "train":
        train_all(args.workers)
    else:
        evaluate_all(args.workers)


if __name__ == "__main__":
    main()
