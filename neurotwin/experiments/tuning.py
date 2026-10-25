"""Baseline parameter tuning (RESEARCH.md v2 §6.7 "Tuning budget fairness", §6.20).

Protocol (fixed before any tuning run)
-------------------------------------
* Scenario A only, VALIDATION seeds only (asserted), full evaluation protocol (empty start,
  35-day warm-up, 14 measured days). This is the same data PPO's hyperparameters are tuned on.
* Objective: the shared episode cost J per measured day (experiments/objective.py), i.e. the
  same objective the RL reward optimises.
* Every candidate runs on the same seeds, so candidates are compared under CRN.
* Selection rule: the candidate with the lowest mean J. The paired difference to the runner-up
  is reported so that a "winner by noise" is visible.
* Trunk reservation is tuned in two stages: its off-service waiting rule is fixed to the tuned
  Priority value, then the reservation levels r_MED, r_SURG, r_ICU are grid-searched over
  0..5 (RESEARCH.md §6.20). This is documented as a deliberate simplification of a joint search.

Run: python -m neurotwin.experiments.tuning [--policies fcfs priority ...] [--workers 16]
Writes config/baselines_tuned.json (committed, with provenance).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..policies import CONFIG_DIR, load_baseline_params, make_policy
from ..simulation.config import SCENARIOS, load_default_config
from ..simulation.hospital import HospitalSimulation, run_episode
from .objective import CostWeights, measured_cost
from .seeds import VALIDATION, require_split

TUNED_PATH = CONFIG_DIR / "baselines_tuned.json"
WAIT_GRID = [2.0, 4.0, 8.0, 12.0]


def candidate_grid(name: str, tuned: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Parameter grid for one baseline (trunk reservation uses Priority's W_os)."""
    if name in ("fcfs", "priority"):
        return [{"off_service_wait_hours": w} for w in WAIT_GRID]
    if name == "load_balancing":
        return [{"imbalance_threshold": t} for t in (0.05, 0.1, 0.2, 0.3)]
    if name == "trunk_reservation":
        wait = tuned["priority"]["off_service_wait_hours"]
        return [
            {"reserve": {"MED": m, "SURG": s, "ICU": i}, "off_service_wait_hours": wait}
            for m, s, i in itertools.product(range(6), repeat=3)
        ]
    if name == "mip":
        # The off-service cost is tuned (not fixed to the reward's kappa): the myopic model
        # cannot see that off-service stays are longer and consume future capacity, so a tuned
        # surrogate cost gives the MIP the same tuning budget as the heuristics. A first run
        # with it fixed at kappa = 2 tuned only the breach cost, which proved irrelevant
        # (docs/stage4_baselines_report.md).
        base = {k: v for k, v in tuned["mip"].items() if not k.startswith("_")}
        return [
            {**base, "breach_cost": b, "off_service_cost": k, "review_horizon_hours": h}
            for b in (0.0, 10.0)
            for k in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
            for h in (1.0, 4.0)
        ]
    raise ValueError(name)


def evaluate_one(args: tuple[str, dict[str, Any], int]) -> float:
    """Worker: cost J per measured day for one (policy, params, seed). Top-level for pickling."""
    name, params, seed = args
    sim = HospitalSimulation(load_default_config(), SCENARIOS["A"], seed=seed)
    run_episode(sim, make_policy(name, params))
    return measured_cost(sim, CostWeights.load())


def tune(name: str, candidates: list[dict[str, Any]], seeds, workers: int) -> dict[str, Any]:
    """Evaluate every candidate on the same validation seeds; pick the lowest mean J."""
    seeds = list(seeds)
    require_split(seeds, "validation")
    tasks = [(name, p, s) for p in candidates for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        costs = list(pool.map(evaluate_one, tasks, chunksize=4))
    matrix = np.asarray(costs).reshape(len(candidates), len(seeds))
    means = matrix.mean(axis=1)
    order = np.argsort(means)
    best, second = int(order[0]), int(order[1]) if len(order) > 1 else None
    summary: dict[str, Any] = {
        "params": candidates[best],
        "objective_mean": float(means[best]),
        "objective_ci95": _ci(matrix[best]),
        "n_candidates": len(candidates),
    }
    if second is not None:
        diff = matrix[second] - matrix[best]  # paired by seed (CRN)
        summary["runner_up"] = {
            "params": candidates[second],
            "objective_mean": float(means[second]),
            "paired_diff_mean": float(diff.mean()),
            "paired_diff_ci95": _ci(diff),
        }
    summary["all"] = [{"params": candidates[i], "objective_mean": float(means[i])} for i in order]
    return summary


def _ci(x: np.ndarray) -> list[float]:
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean() - half), float(x.mean() + half)]


def main() -> None:
    """Command-line entry point: tune baselines and write config/baselines_tuned.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["fcfs", "priority", "load_balancing", "trunk_reservation", "mip"],
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tuned_file = json.loads(TUNED_PATH.read_text(encoding="utf-8")) if TUNED_PATH.exists() else {}
    for name in args.policies:
        current = load_baseline_params(tuned=True)  # includes results of earlier policies
        candidates = candidate_grid(name, current)
        t0 = time.perf_counter()
        result = tune(name, candidates, VALIDATION, args.workers)
        result["provenance"] = {
            "scenario": "A",
            "seeds": [VALIDATION.start, VALIDATION.stop - 1],
            "objective": "J per measured day (config/reward_default.json)",
            "date": date.today().isoformat(),
            "runtime_s": round(time.perf_counter() - t0, 1),
        }
        tuned_file[name] = {k: v for k, v in result.items() if k != "all"}
        tuned_file["_comment"] = (
            "Tuned on scenario A validation seeds only "
            "(neurotwin.experiments.tuning). Do not edit by hand."
        )
        TUNED_PATH.write_text(json.dumps(tuned_file, indent=2), encoding="utf-8")
        ru = result.get("runner_up")
        print(
            f"{name}: best {result['params']} J={result['objective_mean']:.2f} "
            f"CI={[round(v, 2) for v in result['objective_ci95']]}"
            + (
                f" | runner-up {ru['params']} paired diff {ru['paired_diff_mean']:.2f} "
                f"CI={[round(v, 2) for v in ru['paired_diff_ci95']]}"
                if ru
                else ""
            )
            + f" ({result['provenance']['runtime_s']} s, {len(candidates)} candidates)"
        )
        for row in result["all"][:6]:
            print(f"    {row['objective_mean']:9.2f}  {row['params']}")


if __name__ == "__main__":
    main()
