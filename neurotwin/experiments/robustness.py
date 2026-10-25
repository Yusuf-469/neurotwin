"""Robustness and degradation under stress (RESEARCH.md v2 §6.15), from the Stage 6 test data.

Degradation of metric m for policy p in stress scenario S, relative to the training scenario A:
    Deg_m(p, S) = s_m * (mean_S - mean_A) / mean_A,  s_m = +1 (all metrics here are
lower-is-better), so positive = worse. It is computed on means (ratio of means) and bootstrapped
by resampling the 50 seeds jointly across scenarios (same seed index in A and S) and, for PPO,
the 5 agents.

Deviation from §6.15, disclosed: **diversion is reported as an absolute change** (per 100
arrivals), because its scenario-A value is about 0.1-0.6 per 100, which makes a relative change
meaningless (division by nearly zero). The Robustness Index RI(p) = mean relative degradation
over S in B-E and m in {J, mean boarding, p90 boarding, off-service}.

Run: python -m neurotwin.experiments.robustness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluate import BASELINES, ROOT, read_rows
from .statistics import BOOTSTRAP_SEED, N_BOOT

DATA = ROOT / "docs" / "data" / "stage6" / "episodes.csv"
OUT = ROOT / "docs" / "data" / "stage7"
POLICIES = ("ppo",) + BASELINES
RELATIVE = ("J_per_day", "mean_boarding_h", "p90_boarding_h", "off_service_per_100")
STRESS = ("B", "C", "D", "E")


def load_matrices(rows) -> tuple[dict, list[int]]:
    """M[policy][scenario][metric] -> array (K agents, n seeds)."""
    seeds = sorted({int(r["seed"]) for r in rows})
    index = {(r["policy"], int(r["agent"]), r["scenario"], int(r["seed"])): r for r in rows}
    m = {}
    for p in POLICIES:
        agents = sorted({int(r["agent"]) for r in rows if r["policy"] == p})
        m[p] = {s: {metric: np.array([[index[(p, a, s, sd)][metric] for sd in seeds]
                                      for a in agents])
                    for metric in RELATIVE + ("diversion_per_100",)}
                for s in ("A",) + STRESS}
    return m, seeds


def bootstrap_degradation(m: dict, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = next(iter(m["ppo"]["A"].values())).shape[1]
    seed_idx = rng.integers(0, n, size=(n_boot, n))
    agent_idx = {p: rng.integers(0, m[p]["A"]["J_per_day"].shape[0], size=(n_boot, m[p]["A"]["J_per_day"].shape[0]))
                 for p in POLICIES}
    results: dict = {p: {} for p in POLICIES}
    ri_boot = {p: np.zeros(n_boot) for p in POLICIES}
    for p in POLICIES:
        for s in STRESS:
            for metric in RELATIVE + ("diversion_per_100",):
                a_mat, s_mat = m[p]["A"][metric], m[p][s][metric]
                obs_a, obs_s = a_mat.mean(), s_mat.mean()
                relative = metric != "diversion_per_100"
                obs = (obs_s - obs_a) / obs_a * 100 if relative else obs_s - obs_a
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    ag, sd = agent_idx[p][b], seed_idx[b]
                    va = a_mat[ag][:, sd].mean()
                    vs = s_mat[ag][:, sd].mean()
                    boot[b] = (vs - va) / va * 100 if relative else vs - va
                if relative:
                    ri_boot[p] += boot / (len(STRESS) * len(RELATIVE))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                results[p][f"{s}:{metric}"] = {"estimate": float(obs), "ci": [float(lo), float(hi)],
                                               "unit": "% change" if relative else "abs change per 100"}
    ri = {}
    for p in POLICIES:
        obs = np.mean([results[p][f"{s}:{metric}"]["estimate"] for s in STRESS for metric in RELATIVE])
        lo, hi = np.percentile(ri_boot[p], [2.5, 97.5])
        ri[p] = {"estimate": float(obs), "ci": [float(lo), float(hi)]}
    # PPO minus each baseline: same bootstrap replicates (seeds shared), so differences are paired.
    ri_diff = {}
    for b in BASELINES:
        d = ri_boot["ppo"] - ri_boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        ri_diff[b] = {"estimate": ri["ppo"]["estimate"] - ri[b]["estimate"], "ci": [float(lo), float(hi)]}
    return {"degradation": results, "robustness_index": ri, "ppo_minus_baseline_RI": ri_diff}


def main() -> None:
    m, _ = load_matrices(read_rows(DATA))
    res = bootstrap_degradation(m)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robustness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("Robustness Index (mean % degradation over B-E and J/boarding/p90/off-service; lower = more robust)")
    for p, v in sorted(res["robustness_index"].items(), key=lambda kv: kv[1]["estimate"]):
        print(f"  {p:18s} {v['estimate']:7.1f}%  [{v['ci'][0]:7.1f}, {v['ci'][1]:7.1f}]")
    print("PPO RI minus baseline RI (negative = PPO more robust)")
    for b, v in res["ppo_minus_baseline_RI"].items():
        print(f"  vs {b:18s} {v['estimate']:7.1f}  [{v['ci'][0]:7.1f}, {v['ci'][1]:7.1f}]")
    print("J degradation % by scenario")
    for p in POLICIES:
        cells = " ".join(f"{s}:{res['degradation'][p][f'{s}:J_per_day']['estimate']:6.0f}" for s in STRESS)
        print(f"  {p:18s} {cells}")
    print("Diversion absolute change per 100 by scenario")
    for p in POLICIES:
        cells = " ".join(f"{s}:{res['degradation'][p][f'{s}:diversion_per_100']['estimate']:6.2f}" for s in STRESS)
        print(f"  {p:18s} {cells}")


if __name__ == "__main__":
    main()
