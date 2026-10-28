"""Full comparison of the final agent on the fresh seed block (RESEARCH.md v2.9, §6.26 addendum).

The final agent (ENS(BC+PPO), frozen) and Load-Balancing were already run on FRESH_TEST in
§6.26 (docs/data/stage7/bc_ppo_test_episodes.json). This adds the remaining baselines on the same
seeds and repeats the Stage 6 analysis unchanged: 100 primary comparisons with Holm, secondary J
comparisons, summary table and robustness index. In the output the final agent carries the
policy name "ppo" so the Stage 6 analysis code applies as is; the dashboard labels it v2.

Run: python -m neurotwin.experiments.final_eval --workers 10
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .evaluate import BASELINES, ROOT, analyse, run_job, summary_table, write_rows
from .robustness import bootstrap_degradation, load_matrices
from .seeds import FRESH_TEST, require_split
from .statistics import compare

OUT = ROOT / "docs" / "data" / "final"
BC_ROWS = ROOT / "docs" / "data" / "stage7" / "bc_ppo_test_episodes.json"
NEW_BASELINES = tuple(b for b in BASELINES if b != "load_balancing")


def j_comparisons(rows: list[dict]) -> list[dict]:
    """Secondary J comparisons (descriptive 95% CI), same fields as Stage 6."""
    seeds = sorted({int(r["seed"]) for r in rows})
    out = []
    for sc in "ABCDE":
        sub = [r for r in rows if r["scenario"] == sc]
        ppo = np.array(
            [
                [
                    next(
                        r["J_per_day"] for r in sub if r["policy"] == "ppo" and int(r["seed"]) == s
                    )
                    for s in seeds
                ]
            ]
        )
        for b in BASELINES:
            base = np.array(
                [
                    next(r["J_per_day"] for r in sub if r["policy"] == b and int(r["seed"]) == s)
                    for s in seeds
                ]
            )
            c = compare(sc, "J_per_day", b, ppo, base)
            out.append(
                {
                    "scenario": sc,
                    "baseline": b,
                    "ppo_mean": c.ppo_mean,
                    "baseline_mean": c.baseline_mean,
                    "diff": c.diff_mean,
                    "ci_low": c.ci_low,
                    "ci_high": c.ci_high,
                    "rel_diff_pct": 100 * c.diff_mean / c.baseline_mean,
                }
            )
    return out


def main() -> None:
    """Run the remaining baselines on FRESH_TEST and write the Stage 6-format analysis."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    seeds = list(FRESH_TEST)
    require_split(seeds, "fresh_test")
    jobs = [(b, -1, sc, s, "", {}) for b in NEW_BASELINES for sc in "ABCDE" for s in seeds]
    jobs.sort(key=lambda j: j[0] != "mip")  # longest jobs first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        new = list(ex.map(run_job, jobs, chunksize=2))
    previous = json.loads(BC_ROWS.read_text(encoding="utf-8"))
    rows = []
    for r in previous:
        if r["variant"] == "ens_bc_ppo":
            rows.append(
                {**{k: v for k, v in r.items() if k != "variant"}, "policy": "ppo", "agent": 0}
            )
        elif r["variant"] == "load_balancing":
            rows.append({k: v for k, v in r.items() if k != "variant"})
    rows += new
    keys = list(rows[0])
    rows = [{k: r.get(k) for k in keys} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
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
    print(
        f"{len(new)} new episodes in {time.perf_counter() - t0:.0f} s; "
        f"{sum(c.significant_holm for c in comps)}/{len(comps)} comparisons significant after Holm"
    )
    for c in j_comparisons(rows):
        print(
            f"{c['scenario']} v2 - {c['baseline']:18s} {c['rel_diff_pct']:+6.1f}%  [{c['ci_low']:8.1f}, {c['ci_high']:8.1f}]"
        )


if __name__ == "__main__":
    main()
