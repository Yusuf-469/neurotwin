"""Measured variance reduction from Common Random Numbers (RESEARCH.md v2 §6.24).

Scenario A. For each baseline b, the difference PPO - b is estimated two ways:
* paired (CRN): both policies on the same 50 test seeds (Stage 6 data);
* independent: PPO on the test seeds, b on its own disjoint block (CRN_INDEPENDENT seeds,
  used only here).
Reported: 95% t-CI widths, their ratio, and the variance reduction
1 - Var(paired d) / (Var(PPO) + Var(b)), which equals the correlation CRN induces between the
two policies' per-seed outcomes (for equal variances). PPO values are agent-averaged per seed.

Run: python -m neurotwin.experiments.crn_check
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from .evaluate import BASELINES, DEFAULT_AGENTS, ROOT, build_jobs, read_rows, run_job
from .seeds import CRN_INDEPENDENT, require_split

OUT = ROOT / "docs" / "data" / "stage7"
METRICS = ("J_per_day", "mean_boarding_h")


def _half_width(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def main() -> None:
    """Run the independent-seed block and report CRN variance reduction."""
    seeds = list(CRN_INDEPENDENT)
    require_split(seeds, "crn_independent")
    jobs = build_jobs(seeds, ["A"], 0, DEFAULT_AGENTS, policies=BASELINES)
    with ProcessPoolExecutor(4) as ex:
        independent = list(ex.map(run_job, jobs))
    test = [
        r
        for r in read_rows(ROOT / "docs" / "data" / "stage6" / "episodes.csv")
        if r["scenario"] == "A"
    ]
    test_seeds = sorted({int(r["seed"]) for r in test})
    out = []
    for metric in METRICS:
        ppo = np.array(
            [
                np.mean([r[metric] for r in test if r["policy"] == "ppo" and int(r["seed"]) == s])
                for s in test_seeds
            ]
        )
        for b in BASELINES:
            paired_b = np.array(
                [
                    next(r[metric] for r in test if r["policy"] == b and int(r["seed"]) == s)
                    for s in test_seeds
                ]
            )
            indep_b = np.array(
                [
                    r[metric]
                    for r in sorted(independent, key=lambda r: r["seed"])
                    if r["policy"] == b
                ]
            )
            d = ppo - paired_b
            hw_paired = _half_width(d)
            hw_indep = float(
                stats.t.ppf(0.975, 2 * len(ppo) - 2)
                * math.sqrt(ppo.var(ddof=1) / len(ppo) + indep_b.var(ddof=1) / len(indep_b))
            )
            reduction = 1 - d.var(ddof=1) / (ppo.var(ddof=1) + paired_b.var(ddof=1))
            entry = {
                "metric": metric,
                "baseline": b,
                "paired_diff": float(d.mean()),
                "paired_ci_half_width": hw_paired,
                "independent_diff": float(ppo.mean() - indep_b.mean()),
                "independent_ci_half_width": hw_indep,
                "width_ratio_independent_over_paired": hw_indep / hw_paired,
                "variance_reduction": float(reduction),
                "per_seed_correlation": float(np.corrcoef(ppo, paired_b)[0, 1]),
            }
            out.append(entry)
            print(
                f"{metric:16s} vs {b:17s} paired {entry['paired_diff']:7.2f} ±{hw_paired:6.2f} | "
                f"independent {entry['independent_diff']:7.2f} ±{hw_indep:6.2f} | "
                f"CI {entry['width_ratio_independent_over_paired']:4.1f}x narrower with CRN, "
                f"variance reduction {reduction:5.1%}, corr {entry['per_seed_correlation']:.3f}"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crn_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
