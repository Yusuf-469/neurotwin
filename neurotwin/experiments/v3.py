"""Crisis-trained agent v3 (RESEARCH.md v2.10 §6.27).

Steps (run in order):
  finetune  continue PPO from each BC+PPO agent with domain randomisation covering the crises
  validate  v2 vs v3 vs Load-Balancing on VALIDATION seeds in all scenarios; select v2 or v3
  test      one-time test of the selected agent and all baselines on FRESH_TEST_2 (A-E)

Disclosed: the training range covers the stress scenarios, so B-E are in-distribution for v3.

Run: python -m neurotwin.experiments.v3 finetune|validate|test --workers 5
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..simulation.config import ScenarioConfig
from .evaluate import BASELINES, ROOT, analyse, run_job, summary_table, write_rows
from .final_eval import j_comparisons
from .robustness import bootstrap_degradation, load_matrices
from .seeds import FRESH_TEST_2, VALIDATION, require_split
from .statistics import apply_holm, compare
from .train_ppo import RESULTS, train_agent

K = 5
V2_DIR = RESULTS / "bc_ppo"
V3_DIR = RESULTS / "v3"
OUT = ROOT / "docs" / "data" / "v3"
FINETUNE = {"learning_rate": 1e-4, "freeze_obs_norm": True, "critic_warmup_updates": 30}
RANGE = {"arrival": (0.95, 1.15), "staff": (0.85, 1.00), "capacity": (0.85, 1.00)}
DIRS = {"v2": V2_DIR, "v3": V3_DIR}


def sample_wide(rng: np.random.Generator) -> ScenarioConfig:
    """Per-episode training conditions covering every stress scenario (§6.27)."""
    return ScenarioConfig(
        "dr_wide",
        arrival_multiplier=float(rng.uniform(*RANGE["arrival"])),
        staff_multiplier=float(rng.uniform(*RANGE["staff"])),
        capacity_multiplier=float(rng.uniform(*RANGE["capacity"])),
    )


def finetune(workers: int) -> None:
    jobs = [
        ("v3", k, FINETUNE, {"init_from": str(V2_DIR / f"agent_{k}.pt"), "dr_wide": True})
        for k in range(K)
    ]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(train_agent, jobs):
            print(
                f"agent {r['agent_seed']} validation J (scenario A) {r['mean_J']:.1f}", flush=True
            )


def validate(workers: int) -> None:
    seeds = list(VALIDATION)
    require_split(seeds, "validation")
    jobs = [
        ("ppo_ensemble", 0, sc, s, str(DIRS[v]), {}) for v in DIRS for sc in "ABCDE" for s in seeds
    ]
    names = [v for v in DIRS for _ in "ABCDE" for _ in seeds]
    jobs += [("load_balancing", -1, sc, s, "", {}) for sc in "ABCDE" for s in seeds]
    names += ["load_balancing"] * (5 * len(seeds))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=2))
    mean = {}
    for name in ("v2", "v3", "load_balancing"):
        for sc in "ABCDE":
            mean[(name, sc)] = float(
                np.mean(
                    [
                        r["J_per_day"]
                        for r, n in zip(rows, names)
                        if n == name and r["scenario"] == sc
                    ]
                )
            )
    score = {
        v: float(
            np.mean(
                [
                    (mean[(v, sc)] - mean[("load_balancing", sc)]) / mean[("load_balancing", sc)]
                    for sc in "ABCDE"
                ]
            )
        )
        for v in DIRS
    }
    selected = min(score, key=score.get)
    out = {
        "mean_J": {f"{n}:{sc}": v for (n, sc), v in mean.items()},
        "score_vs_lb": score,
        "selected": selected,
        "rule": "lowest mean relative J vs Load-Balancing over A-E, validation seeds",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for sc in "ABCDE":
        lb = mean[("load_balancing", sc)]
        print(
            sc,
            " ".join(
                f"{v}: {mean[(v, sc)]:7.1f} ({100 * (mean[(v, sc)] - lb) / lb:+5.1f}%)"
                for v in DIRS
            ),
            f"LB: {lb:7.1f}",
        )
    print("score vs LB:", {k: round(100 * v, 2) for k, v in score.items()}, "selected:", selected)


def test(workers: int) -> None:
    sel = json.loads((OUT / "validation.json").read_text(encoding="utf-8"))["selected"]
    if (OUT / "test.json").exists():
        raise SystemExit("FRESH_TEST_2 has already been used; refusing to run it again")
    seeds = list(FRESH_TEST_2)
    require_split(seeds, "fresh_test_2")
    jobs = [("ppo_ensemble", 0, sc, s, str(DIRS[sel]), {}) for sc in "ABCDE" for s in seeds]
    jobs += [(b, -1, sc, s, "", {}) for b in BASELINES for sc in "ABCDE" for s in seeds]
    jobs.sort(key=lambda j: j[0] != "mip")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=2))
    rows = [
        {**r, "policy": "ppo", "agent": 0} if r["policy"] == "ppo_ensemble" else r for r in rows
    ]
    write_rows(rows, OUT / "episodes.csv")
    comps = analyse(rows)
    (OUT / "comparisons.json").write_text(
        json.dumps([c.as_dict() for c in comps], indent=1), encoding="utf-8"
    )
    (OUT / "J_comparisons.json").write_text(
        json.dumps(j_comparisons(rows), indent=1), encoding="utf-8"
    )
    (OUT / "summary.json").write_text(json.dumps(summary_table(rows), indent=1), encoding="utf-8")
    m, _ = load_matrices(rows)
    (OUT / "robustness.json").write_text(
        json.dumps(bootstrap_degradation(m), indent=1), encoding="utf-8"
    )

    def vec(policy, sc):
        return np.array(
            [
                next(
                    r["J_per_day"]
                    for r in rows
                    if r["policy"] == policy and r["scenario"] == sc and int(r["seed"]) == s
                )
                for s in seeds
            ]
        )

    primary = [
        compare(
            sc, "J_per_day", "load_balancing", vec("ppo", sc)[None, :], vec("load_balancing", sc)
        )
        for sc in "ABCDE"
    ]
    apply_holm(primary)
    collapses = {
        p: sum(r["diversion_per_100"] > 50 for r in rows if r["policy"] == p)
        for p in ("ppo",) + BASELINES
    }
    (OUT / "test.json").write_text(
        json.dumps(
            {
                "selected": sel,
                "primary_vs_load_balancing": [c.as_dict() for c in primary],
                "collapses": collapses,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"{len(rows)} episodes in {time.perf_counter() - t0:.0f} s; selected {sel}; collapses {collapses}"
    )
    for c in primary:
        print(
            f"{c.scenario} {sel} - LB {c.diff_mean:8.1f} ({100 * c.diff_mean / c.baseline_mean:+.1f}%) "
            f"[{c.ci_low:8.1f}, {c.ci_high:8.1f}] p_holm {c.p_holm:.4f} {'*' if c.significant_holm else ''}"
        )
    for c in j_comparisons(rows):
        print(f"  {c['scenario']} vs {c['baseline']:18s} {c['rel_diff_pct']:+6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["finetune", "validate", "test"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    {"finetune": finetune, "validate": validate, "test": test}[args.command](args.workers)


if __name__ == "__main__":
    main()
