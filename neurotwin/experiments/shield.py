"""Safety shield against the collapse mode (RESEARCH.md v2.7 §6.25, pre-registered).

The five final agents run with the shield (no HOLD while the ED is full and a placement is
feasible) on the 50 test seeds in scenarios A-E. Unshielded PPO and Load-Balancing are read from
the Stage 6 episodes: same seeds and a deterministic simulator, so re-running them would
reproduce identical rows.

Run: python -m neurotwin.experiments.shield --workers 10
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .evaluate import DEFAULT_AGENTS, ROOT, read_rows, run_job
from .seeds import TEST
from .statistics import apply_holm, compare

K = 5
COLLAPSE_DIVERSION = 50.0  # diversions per 100 arrivals; the Stage 6 collapse definition
STAGE6 = ROOT / "docs" / "data" / "stage6" / "episodes.csv"
OUT = ROOT / "docs" / "data" / "stage7"


def main() -> None:
    """Run the shielded agents on the test seeds and compare with Stage 6."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    seeds = list(TEST)
    jobs = [
        ("ppo_shielded", a, sc, s, str(DEFAULT_AGENTS), {})
        for a in range(K)
        for sc in "ABCDE"
        for s in seeds
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=4))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shield_episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    stage6 = [r for r in read_rows(STAGE6) if r["policy"] in ("ppo", "load_balancing")]

    def matrix(source, policy, scen, metric, agents):
        idx = {
            (int(r["agent"]), int(r["seed"])): r[metric]
            for r in source
            if r["policy"] == policy and r["scenario"] == scen
        }
        return np.array([[idx[(a, s)] for s in seeds] for a in agents])

    collapses = {
        "ppo_shielded": sum(r["diversion_per_100"] > COLLAPSE_DIVERSION for r in rows),
        "ppo": sum(
            r["diversion_per_100"] > COLLAPSE_DIVERSION for r in stage6 if r["policy"] == "ppo"
        ),
        "load_balancing": sum(
            r["diversion_per_100"] > COLLAPSE_DIVERSION
            for r in stage6
            if r["policy"] == "load_balancing"
        ),
    }
    comps = []
    agent0 = {}
    for sc in "ABCDE":
        shielded = matrix(rows, "ppo_shielded", sc, "J_per_day", range(K))
        comps.append(
            compare(
                sc,
                "J_per_day",
                "ppo",
                shielded,
                matrix(stage6, "ppo", sc, "J_per_day", range(K)).mean(0),
            )
        )
        comps.append(
            compare(
                sc,
                "J_per_day",
                "load_balancing",
                shielded,
                matrix(stage6, "load_balancing", sc, "J_per_day", [-1])[0],
            )
        )
        a0s = matrix(rows, "ppo_shielded", sc, "J_per_day", [0])[0]
        a0 = matrix(stage6, "ppo", sc, "J_per_day", [0])[0]
        agent0[sc] = {
            "shielded_mean_J": float(a0s.mean()),
            "unshielded_mean_J": float(a0.mean()),
            "shielded_max_J": float(a0s.max()),
            "unshielded_max_J": float(a0.max()),
        }
    apply_holm(comps)
    activations = {
        sc: float(np.mean([r["shield_activations"] for r in rows if r["scenario"] == sc]))
        for sc in "ABCDE"
    }
    out = {
        "collapses": collapses,
        "n_episodes": len(rows),
        "comparisons": [c.as_dict() for c in comps],
        "agent0": agent0,
        "mean_shield_activations_per_episode": activations,
    }
    (OUT / "shield_comparisons.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.0f} s")
    print("collapses:", collapses)
    print("shield activations per episode:", {k: round(v, 1) for k, v in activations.items()})
    for c in comps:
        print(
            f"{c.scenario} shielded - {c.baseline:15s} {c.diff_mean:8.1f} [{c.ci_low:8.1f}, {c.ci_high:8.1f}] "
            f"{'*' if c.significant_holm else ''}"
        )
    print(json.dumps(agent0, indent=1))


if __name__ == "__main__":
    main()
