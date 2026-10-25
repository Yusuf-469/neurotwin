"""Primary experiment matrix (RESEARCH.md v2 §6.22) and its statistical analysis (§6.14).

Every policy runs the full protocol (empty start, 35-day warm-up, 14 measured days) on the same
seeds in every scenario (CRN). PPO contributes K agents. Rows are written to a CSV with one row
per (policy, agent, scenario, seed); the analysis then compares PPO with each baseline.

``--split test`` is the final evaluation and is run once. ``--split validation`` uses the
validation seeds (development only).

Run: python -m neurotwin.experiments.evaluate --split test --out docs/data/stage6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..policies import load_baseline_params, make_policy
from ..policies.ppo_policy import PPOPolicy
from ..simulation.config import SCENARIOS, load_default_config
from ..simulation.hospital import HospitalSimulation, run_episode
from .metrics import ALL_METRICS, PRIMARY_METRICS, episode_metrics
from .objective import CostWeights
from .seeds import TEST, VALIDATION, require_split
from .statistics import apply_holm, compare

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENTS = ROOT / "results" / "ppo" / "final_A"
BASELINES = ("fcfs", "priority", "load_balancing", "trunk_reservation", "mip")


def run_job(job: tuple[str, int, str, int, str, dict]) -> dict[str, Any]:
    """Worker: one episode. job = (policy, agent, scenario, seed, agent_dir, config overrides)."""
    policy_name, agent, scenario, seed, agent_dir, overrides = job
    import torch
    torch.set_num_threads(1)
    config = load_default_config().with_overrides(**overrides) if overrides else load_default_config()
    if policy_name == "ppo":
        policy = PPOPolicy.load(Path(agent_dir) / f"agent_{agent}.pt")
    else:
        policy = make_policy(policy_name, load_baseline_params()[policy_name])
    sim = HospitalSimulation(config, SCENARIOS[scenario], seed=seed)
    t0 = time.perf_counter()
    run_episode(sim, policy)
    row = {"policy": policy_name, "agent": agent, "scenario": scenario, "seed": seed,
           "runtime_s": time.perf_counter() - t0}
    row.update(episode_metrics(sim, CostWeights.load()))
    return row


def build_jobs(seeds, scenarios, agents: int, agent_dir: Path, overrides: dict | None = None,
               policies=("ppo",) + BASELINES) -> list[tuple]:
    jobs = []
    for scenario in scenarios:
        for seed in seeds:
            for p in policies:
                for a in (range(agents) if p == "ppo" else [-1]):
                    jobs.append((p, a, scenario, seed, str(agent_dir), overrides or {}))
    return jobs


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["scenario"], r["policy"], r["agent"], r["seed"])))


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("policy", "scenario"):
                r[k] = float(v)
    return rows


def analyse(rows: list[dict], metrics=PRIMARY_METRICS, baselines=BASELINES) -> list:
    """PPO vs each baseline, per scenario and metric; Holm over the whole family."""
    comparisons = []
    scenarios = sorted({r["scenario"] for r in rows})
    for scenario in scenarios:
        sub = [r for r in rows if r["scenario"] == scenario]
        seeds = sorted({int(r["seed"]) for r in sub})
        agents = sorted({int(r["agent"]) for r in sub if r["policy"] == "ppo"})
        for metric in metrics:
            ppo = np.array([[next(r[metric] for r in sub if r["policy"] == "ppo"
                                  and int(r["agent"]) == a and int(r["seed"]) == s)
                             for s in seeds] for a in agents])
            for b in baselines:
                base = np.array([next(r[metric] for r in sub if r["policy"] == b
                                      and int(r["seed"]) == s) for s in seeds])
                culprit = "ppo" if np.isnan(ppo).any() else b if np.isnan(base).any() else None
                if culprit is not None:
                    raise ValueError(f"NaN in {metric}/{scenario}/{culprit}; primary metrics "
                                     "must be defined for every episode")
                comparisons.append(compare(scenario, metric, b, ppo, base))
    apply_holm(comparisons)
    return comparisons


def summary_table(rows: list[dict], metrics=ALL_METRICS) -> list[dict]:
    """Mean and 95% t-CI over seeds per (scenario, policy); PPO averaged over agents first."""
    from scipy import stats
    out = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for policy in ("ppo",) + BASELINES:
            sub = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not sub:
                continue
            seeds = sorted({int(r["seed"]) for r in sub})
            entry = {"scenario": scenario, "policy": policy}
            for m in metrics:
                per_seed = np.array([np.nanmean([r[m] for r in sub if int(r["seed"]) == s])
                                     for s in seeds])
                per_seed = per_seed[~np.isnan(per_seed)]
                if len(per_seed) < 2:
                    entry[m] = (math.nan, math.nan, math.nan)
                    continue
                half = stats.t.ppf(0.975, len(per_seed) - 1) * per_seed.std(ddof=1) / math.sqrt(len(per_seed))
                entry[m] = (float(per_seed.mean()), float(per_seed.mean() - half),
                            float(per_seed.mean() + half))
            out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["test", "validation"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENTS)
    parser.add_argument("--scenarios", nargs="+", default=list("ABCDE"))
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()

    seeds = list(TEST if args.split == "test" else VALIDATION)
    require_split(seeds, args.split)
    jobs = build_jobs(seeds, args.scenarios, args.agents, args.agent_dir)
    # Longest jobs (MIP, stressed scenarios) first for better load balancing.
    jobs.sort(key=lambda j: (j[0] != "mip", j[2] == "A"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_job, jobs, chunksize=2))
    write_rows(rows, args.out / "episodes.csv")
    comparisons = analyse(rows)
    (args.out / "comparisons.json").write_text(
        json.dumps([c.as_dict() for c in comparisons], indent=1), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary_table(rows), indent=1), encoding="utf-8")
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.0f} s; "
          f"{sum(c.significant_holm for c in comparisons)}/{len(comparisons)} comparisons "
          f"significant after Holm")


if __name__ == "__main__":
    main()
