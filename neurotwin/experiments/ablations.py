"""Ablation experiments (RESEARCH.md v2 §6.16).

Each ablation retrains PPO with one change, using the final configuration otherwise, and is
compared with the full model (final_A) on the same test seeds (CRN), scenarios A (training)
and E (combined stress).

Deviation (compute budget, disclosed): K = 3 agents per ablation instead of the
pre-registered 5. The full model keeps its 5 agents. Agent seeds 0-2 are shared with the full
model, so ablation and full agents with the same seed differ only by the ablated component.

Feature indices are 0-based positions in rl/features.FEATURE_NAMES; the spec's 1-based numbers
are given in comments.

Run: python -m neurotwin.experiments.ablations train   (about 1 hour on 18 cores)
     python -m neurotwin.experiments.ablations evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .evaluate import ROOT, run_job
from .seeds import TEST
from .statistics import apply_holm, compare
from .train_ppo import RESULTS, train_agent

K_ABLATION = 3
ABLATIONS: dict[str, dict] = {
    "B_no_demand_history": {"drop_features": [13, 14, 15]},       # spec features 14-16
    "C_no_temporal": {"drop_features": [9, 10, 11, 12]},          # 10-13
    "D_no_staffing": {"drop_features": [7, 8]},                   # 8-9
    "E_no_priority_aggregates": {"drop_features": [16, 17, 18]},  # 17-19
    "F1_no_diversion_penalty": {"beta": 0.0},
    "F2_no_off_service_penalty": {"kappa": 0.0},
    "F3_uniform_priority_weights": {"uniform_omega": True},
    "G_no_icu_flow_state": {"drop_features": [5, 6]},             # 6-7
}
OUT = ROOT / "docs" / "data" / "stage7"
METRICS = ("J_per_day", "mean_boarding_h", "p90_boarding_h", "diversion_per_100",
           "off_service_per_100")


def train_all(workers: int) -> None:
    jobs = [(f"ablation_{name}", k, {}, opts) for name, opts in ABLATIONS.items()
            for k in range(K_ABLATION)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(train_agent, jobs):
            print(f"agent {r['agent_seed']} validation J {r['mean_J']:.1f}", flush=True)
    print(f"trained {len(jobs)} agents in {(time.perf_counter() - t0) / 60:.0f} min")


def evaluate_all(workers: int) -> None:
    seeds = list(TEST)
    variants = {"full": (RESULTS / "final_A", 5)}
    variants.update({n: (RESULTS / f"ablation_{n}", K_ABLATION) for n in ABLATIONS})
    jobs = [("ppo", a, scen, s, str(d), {}) for name, (d, k) in variants.items()
            for a in range(k) for scen in ("A", "E") for s in seeds]
    names = [name for name, (d, k) in variants.items() for a in range(k) for _ in ("A", "E")
             for _ in seeds]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=4))
    for row, name in zip(rows, names):
        row["variant"] = name
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation_episodes.json").write_text(json.dumps(rows), encoding="utf-8")

    comps = []
    for scen in ("A", "E"):
        for metric in METRICS:
            def matrix(variant, k):
                return np.array([[next(r[metric] for r in rows if r["variant"] == variant
                                       and r["scenario"] == scen and int(r["agent"]) == a
                                       and int(r["seed"]) == s) for s in seeds] for a in range(k)])
            full = matrix("full", 5).mean(axis=0)            # full model: agent-averaged per seed
            for name in ABLATIONS:
                c = compare(scen, metric, f"full_vs_{name}", matrix(name, K_ABLATION), full)
                comps.append(c)
    apply_holm(comps)                                      # family: 8 ablations x 2 x 5 metrics
    (OUT / "ablation_comparisons.json").write_text(
        json.dumps([c.as_dict() for c in comps], indent=1), encoding="utf-8")
    for c in comps:
        if c.metric == "J_per_day":
            print(f"{c.scenario} {c.baseline:40s} ablated-full ΔJ {c.diff_mean:8.1f} "
                  f"[{c.ci_low:8.1f}, {c.ci_high:8.1f}] {'*' if c.significant_holm else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    (train_all if args.command == "train" else evaluate_all)(args.workers)


if __name__ == "__main__":
    main()
