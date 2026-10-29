"""Map of what the latest agent learned (descriptive; dashboard Results page).

Runs the crisis-trained agent (v3, five-network vote) in all five situations and records every
decision where it had a real choice (at least two valid moves). For each such decision the map
uses the network's own internal representation: the 64 activations of the last hidden layer of
agent 0's policy network, i.e. how the network encodes the situation just before scoring the
moves. These are projected onto their two leading principal components (PCA via SVD, plain
NumPy), so nearby dots are situations the network treats as similar.

Each axis is described by the input features most correlated with it, so the picture can be
read without knowing the network. Seeds come from the TRAIN range (90,000-90,003): this is a
description of behaviour, not an evaluation, and it must not touch any validation or test block.
Nothing here is a performance claim.

Usage: python -m neurotwin.experiments.policy_map
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..policies import load_baseline_params, make_policy
from ..policies.ensemble_ppo import EnsemblePPOPolicy
from ..policies.insight_recorder import InsightRecorder, compare
from ..rl.actions import Action
from ..rl.features import FeatureScales
from ..rl.introspect import input_groups
from ..simulation.config import SCENARIOS, load_default_config
from ..simulation.hospital import HospitalSimulation, run_until
from .seeds import require_split

ROOT = Path(__file__).resolve().parents[2]
AGENTS = sorted((ROOT / "results" / "ppo" / "v3").glob("agent_*.pt"))
OUT = ROOT / "docs" / "data" / "v3" / "policy_map.json"
SEEDS = [90_000, 90_001, 90_002, 90_003]
MAX_POINTS = 3000
SUBSAMPLE_SEED = 7


def hidden(agent, obs: np.ndarray) -> np.ndarray:
    """Last hidden layer of the actor (the representation the output layer scores)."""
    with torch.no_grad():
        x = torch.as_tensor(agent.obs_rms.normalize(obs))
        return agent.model.actor[:-1](x).numpy()


def collect() -> list[dict]:
    require_split(SEEDS, "train")
    config = load_default_config()
    rows = []
    for sc in "ABCDE":
        for seed in SEEDS:
            rec = InsightRecorder(
                EnsemblePPOPolicy.load(AGENTS),
                make_policy("load_balancing", load_baseline_params()["load_balancing"]),
                keep=10**9,
            )
            sim = HospitalSimulation(config, SCENARIOS[sc], seed=seed)
            run_until(sim, rec)
            for snap in rec.snapshots.values():
                if snap["mask"].sum() < 2:  # only HOLD possible: no choice was made
                    continue
                rows.append({"scenario": sc, "seed": seed, **snap})
    return rows


def build() -> dict:
    rows = collect()
    n_total = len(rows)
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    if len(rows) > MAX_POINTS:
        keep = np.sort(rng.choice(len(rows), MAX_POINTS, replace=False))
        rows = [rows[i] for i in keep]
    obs = np.stack([r["obs"] for r in rows])
    agent0 = EnsemblePPOPolicy.load(AGENTS[:1]).agents[0]
    agents = EnsemblePPOPolicy.load(AGENTS).agents
    h = np.stack([hidden(agent0, o) for o in obs])
    hc = h - h.mean(0)
    _, sv, vt = np.linalg.svd(hc, full_matrices=False)
    xy = hc @ vt[:2].T
    explained = (sv**2 / (sv**2).sum())[:2]

    scales = FeatureScales.from_config(load_default_config())
    groups = input_groups(scales)
    axes = []
    for k in range(2):
        corr = []
        for g in groups:
            # a grouped input (time of day) is summarised by its strongest component
            cs = [abs(np.corrcoef(obs[:, i], xy[:, k])[0, 1]) if obs[:, i].std() > 0 else 0.0
                  for i in g.idx]
            signed = [np.corrcoef(obs[:, i], xy[:, k])[0, 1] if obs[:, i].std() > 0 else 0.0
                      for i in g.idx]
            j = int(np.argmax(cs))
            corr.append((g.label, float(signed[j])))
        corr.sort(key=lambda t: -abs(t[1]))
        # PCA signs are arbitrary: orient each axis so its strongest input increases along it
        # (keeps the dashboard's "more crowded ->" label true if the map is regenerated)
        if corr[0][1] < 0:
            xy[:, k] *= -1
            corr = [(lab, -c) for lab, c in corr]
        axes.append({
            "explained_variance": round(float(explained[k]), 4),
            "top_inputs": [{"label": lab, "correlation": round(c, 3)} for lab, c in corr[:4]],
        })

    points = []
    for r, (x, y) in zip(rows, xy):
        probs = np.mean([a.action_probabilities(r["obs"], r["mask"]) for a in agents], axis=0)
        chosen = int(np.argmax(np.where(r["mask"], probs, -1.0)))
        o = r["obs"]
        points.append({
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "action": Action(chosen).name,
            "confidence": round(float(probs[chosen]), 3),
            "scenario": r["scenario"],
            "agreement": compare(r["agent"], r["rule"]),
            # a few readable facts for the hover card (same inverse scaling as the insight panel)
            "occ": [round(float(v), 2) for v in o[:3]],
            "waiting": int(round(float(o[3] + o[4]) * scales.q_max)),
            "day": round(float(r["time"]) / 24, 2),
        })
    return {
        "description": "Each dot is one decision where the agent had a real choice. Position: the "
        "agent's internal representation (last hidden layer, network 0), projected by PCA.",
        "agent": "v3 (five-network vote; positions from network 0)",
        "seeds": SEEDS,
        "seed_split": "train",
        "scenarios": list("ABCDE"),
        "n_decisions_with_choice": n_total,
        "max_points": MAX_POINTS,
        "axes": axes,
        "points": points,
    }


def main() -> None:
    out = build()
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({len(out['points'])} points, axes var {[a['explained_variance'] for a in out['axes']]})")
    for k, a in enumerate(out["axes"]):
        print(f"axis {k + 1}:", a["top_inputs"])


if __name__ == "__main__":
    main()
