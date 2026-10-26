"""Imitation-initialised PPO and ensembles vs Load-Balancing (RESEARCH.md v2.8 §6.26).

Steps (run in order):
  clone     behaviour-clone Load-Balancing into K actor networks (TRAIN seeds)
  finetune  PPO fine-tuning from each clone (TRAIN seeds)
  validate  score every candidate on VALIDATION seeds (scenario A) and select one
  test      one-time final test of the selected candidate on FRESH_TEST seeds (A-E)

Run: python -m neurotwin.experiments.bc_ppo clone|finetune|validate|test --workers 5
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..policies import load_baseline_params, make_policy
from ..policies.ensemble_ppo import EnsemblePPOPolicy
from ..policies.ppo_policy import PPOPolicy
from ..simulation.config import load_default_config
from .evaluate import DEFAULT_AGENTS, ROOT, run_job
from .seeds import FRESH_TEST, VALIDATION, require_split
from .statistics import apply_holm, compare
from .train_ppo import RESULTS, episode_seeds, evaluate, train_agent

K = 5
BC_EPISODES = 30
BC_DIR = RESULTS / "bc"
FT_DIR = RESULTS / "bc_ppo"
OUT = ROOT / "docs" / "data" / "stage7"
FINETUNE = {"learning_rate": 1e-4, "freeze_obs_norm": True, "critic_warmup_updates": 30}
CANDIDATES = ("bc_ppo", "ens_ppo", "ens_bc_ppo")  # selection set (§6.26)


def _clone_job(k: int) -> dict:
    import torch

    from ..rl.bc import clone, collect

    torch.set_num_threads(1)
    seeds = episode_seeds(k)
    data_seeds = [next(seeds) for _ in range(BC_EPISODES)]
    teacher = make_policy("load_balancing", load_baseline_params()["load_balancing"])
    demos = collect(teacher, load_default_config(), data_seeds)
    agent, stats = clone(demos, seed=k)
    BC_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(BC_DIR / f"agent_{k}.pt")
    return {"agent": k, **stats}


def _val_job(args: tuple[str, int]) -> tuple[str, int, list[float]]:
    import torch

    torch.set_num_threads(1)
    name, k = args
    if name == "load_balancing":
        policy = make_policy(name, load_baseline_params()[name])
    elif name == "ppo":
        policy = PPOPolicy.load(DEFAULT_AGENTS / f"agent_{k}.pt")
    elif name in ("bc", "bc_ppo"):
        policy = PPOPolicy.load((BC_DIR if name == "bc" else FT_DIR) / f"agent_{k}.pt")
    elif name == "ens_ppo":
        policy = EnsemblePPOPolicy.load(sorted(DEFAULT_AGENTS.glob("agent_*.pt")))
    else:  # ens_bc_ppo
        policy = EnsemblePPOPolicy.load(sorted(FT_DIR.glob("agent_*.pt")))
    return name, k, evaluate(policy)


def validate(workers: int) -> dict:
    """Validation J (scenario A) of every candidate; selects the lowest-J candidate in 2-4."""
    require_split(list(VALIDATION), "validation")
    jobs = [("load_balancing", -1), ("ens_ppo", -1), ("ens_bc_ppo", -1)]
    jobs += [(n, k) for n in ("ppo", "bc", "bc_ppo") for k in range(K)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_val_job, jobs))
    scores: dict[str, dict] = {}
    for name, k, costs in res:
        scores.setdefault(name, {"per_agent": {}})["per_agent"][k] = float(np.mean(costs))
    for v in scores.values():
        v["mean_J"] = float(np.mean(list(v["per_agent"].values())))
    selected = min(CANDIDATES, key=lambda n: scores[n]["mean_J"])
    out = {
        "validation_J": scores,
        "selected": selected,
        "rule": "lowest mean validation J among " + ", ".join(CANDIDATES),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bc_ppo_validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for n, v in sorted(scores.items(), key=lambda kv: kv[1]["mean_J"]):
        print(
            f"{n:15s} validation J {v['mean_J']:7.1f}   per agent {[round(x, 1) for x in v['per_agent'].values()]}"
        )
    print("selected:", selected)
    return out


def test(workers: int) -> None:
    """One-time final test of the selected candidate on FRESH_TEST, A-E (§6.26)."""
    sel = json.loads((OUT / "bc_ppo_validation.json").read_text(encoding="utf-8"))["selected"]
    out_path = OUT / "bc_ppo_test.json"
    if out_path.exists():
        raise SystemExit("the fresh test block has already been used; refusing to run it again")
    seeds = list(FRESH_TEST)
    require_split(seeds, "fresh_test")
    cand_jobs = (
        [("ppo", k, sc, s, str(FT_DIR), {}) for k in range(K) for sc in "ABCDE" for s in seeds]
        if sel == "bc_ppo"
        else [
            ("ppo_ensemble", 0, sc, s, str(DEFAULT_AGENTS if sel == "ens_ppo" else FT_DIR), {})
            for sc in "ABCDE"
            for s in seeds
        ]
    )
    ref_jobs = [("load_balancing", -1, sc, s, "", {}) for sc in "ABCDE" for s in seeds]
    ref_jobs += [
        ("ppo", k, sc, s, str(DEFAULT_AGENTS), {})
        for k in range(K)
        for sc in "ABCDE"
        for s in seeds
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        cand = list(ex.map(run_job, cand_jobs, chunksize=4))
        refs = list(ex.map(run_job, ref_jobs, chunksize=4))
    for r in cand:
        r["variant"] = sel
    for r in refs:
        r["variant"] = r["policy"]
    rows = cand + refs

    def matrix(variant, scen, metric="J_per_day"):
        sub = [r for r in rows if r["variant"] == variant and r["scenario"] == scen]
        agents = sorted({int(r["agent"]) for r in sub})
        idx = {(int(r["agent"]), int(r["seed"])): r[metric] for r in sub}
        return np.array([[idx[(a, s)] for s in seeds] for a in agents])

    primary, secondary = [], []
    for sc in "ABCDE":
        c = matrix(sel, sc)
        primary.append(
            compare(sc, "J_per_day", "load_balancing", c, matrix("load_balancing", sc)[0])
        )
        secondary.append(compare(sc, "J_per_day", "ppo", c, matrix("ppo", sc).mean(0)))
    apply_holm(primary)
    apply_holm(secondary)
    collapses = {
        v: sum(r["diversion_per_100"] > 50 for r in rows if r["variant"] == v)
        for v in (sel, "load_balancing", "ppo")
    }
    result = {
        "selected": sel,
        "seeds": [seeds[0], seeds[-1]],
        "primary_vs_load_balancing": [c.as_dict() for c in primary],
        "secondary_vs_ppo": [c.as_dict() for c in secondary],
        "collapses": collapses,
        "n_episodes": {
            v: sum(r["variant"] == v for r in rows) for v in (sel, "load_balancing", "ppo")
        },
    }
    (OUT / "bc_ppo_test_episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.0f} s; collapses {collapses}")
    for c in primary:
        print(
            f"{c.scenario} {sel} - load_balancing {c.diff_mean:8.1f} [{c.ci_low:8.1f}, {c.ci_high:8.1f}] "
            f"({100 * c.diff_mean / c.baseline_mean:+.1f}%) {'*' if c.significant_holm else ''}"
        )
    for c in secondary:
        print(
            f"{c.scenario} {sel} - ppo {c.diff_mean:8.1f} [{c.ci_low:8.1f}, {c.ci_high:8.1f}] {'*' if c.significant_holm else ''}"
        )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["clone", "finetune", "validate", "test"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.command == "clone":
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(_clone_job, range(K)):
                print(r, flush=True)
    elif args.command == "finetune":
        jobs = [
            ("bc_ppo", k, FINETUNE, {"init_from": str(BC_DIR / f"agent_{k}.pt")}) for k in range(K)
        ]
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(train_agent, jobs):
                print(f"agent {r['agent_seed']} validation J {r['mean_J']:.1f}", flush=True)
    elif args.command == "validate":
        validate(args.workers)
    else:
        test(args.workers)


if __name__ == "__main__":
    main()
