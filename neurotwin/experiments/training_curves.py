"""Training curves of the three agent generations, for the dashboard's training replay.

Condenses the per-update training logs (results/ppo/<run>/log_<k>.json, five agents per
generation) into about 150 points per curve: for each step bin, the median and the range over
the five agents of

* reward per training episode (mean of the episodes finished in the bin, then a trailing mean
  over 10 bins, because training days differ a lot in difficulty). These are
  training episodes (sampled actions, random warm starting states; v3 also randomised
  conditions), so they are not test performance and are not comparable across generations;
* policy entropy (how undecided the policy is: high = close to random, low = decisive);
* critic explained variance (how well the value network predicts returns; 1 = perfect).

Only existing logs are read; nothing is retrained. Network snapshots from mid-training were not
kept, so the replay shows how learning progressed, not the agent's behaviour at each stage.

Usage: python -m neurotwin.experiments.training_curves
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "v1": ("final_A", "learned from scratch, normal days only"),
    "v2": ("bc_ppo", "started as a copy of the balance-the-wards rule, then reinforcement learning"),
    "v3": ("v3", "continued from v2 on randomly varied days, including crises"),
}
OUT = ROOT / "docs" / "data" / "training_curves.json"
N_BINS = 150
SMOOTH_BINS = 10


def _agent_curves(log: dict, edges: np.ndarray) -> dict[str, np.ndarray]:
    ups = log["updates"]
    steps = np.array([u["step"] for u in ups], float)
    out = {}
    for key in ("entropy", "explained_variance"):
        vals = np.array([u[key] for u in ups], float)
        out[key] = np.array([
            np.nanmean(vals[(steps > lo) & (steps <= hi)]) if ((steps > lo) & (steps <= hi)).any() else np.nan
            for lo, hi in zip(edges[:-1], edges[1:])
        ])
    ep = np.array(log["episode_returns"], float)
    out["episode_reward"] = np.array([
        np.mean(ep[(ep[:, 0] > lo) & (ep[:, 0] <= hi), 1]) if ((ep[:, 0] > lo) & (ep[:, 0] <= hi)).any() else np.nan
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    return out


def _fill(x: np.ndarray) -> np.ndarray:
    """Carry the last value forward over empty bins (few episodes end in some bins)."""
    x = x.copy()
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    return x


def _trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1): i + 1]
        seg = seg[~np.isnan(seg)]
        out[i] = seg.mean() if len(seg) else np.nan
    return out


def build() -> dict:
    out = {"bins": N_BINS, "note": __doc__.split("\n\n")[1].replace("\n", " "), "versions": {}}
    for version, (run, story) in RUNS.items():
        logs = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((ROOT / "results" / "ppo" / run).glob("log_*.json"))]
        total = max(u["step"] for lg in logs for u in lg["updates"])
        edges = np.linspace(0, total, N_BINS + 1)
        per_agent = [_agent_curves(lg, edges) for lg in logs]
        curves = {}
        for key in ("episode_reward", "entropy", "explained_variance"):
            m = np.stack([_fill(a[key]) for a in per_agent])
            if key == "episode_reward":
                # training days differ a lot in difficulty, so reward is smoothed with a trailing
                # mean over SMOOTH_BINS bins (about 200,000 steps) before taking median/range
                m = np.stack([_trailing_mean(row, SMOOTH_BINS) for row in m])
            curves[key] = {
                "median": [None if np.isnan(v) else round(float(v), 4) for v in np.nanmedian(m, 0)],
                "low": [None if np.isnan(v) else round(float(v), 4) for v in np.nanmin(m, 0)],
                "high": [None if np.isnan(v) else round(float(v), 4) for v in np.nanmax(m, 0)],
            }
        out["versions"][version] = {
            "run": run,
            "story": story,
            "n_agents": len(logs),
            "total_steps": int(total),
            "step": [int(v) for v in edges[1:]],
            "curves": curves,
        }
    return out


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for v, d in data["versions"].items():
        c = d["curves"]
        print(v, d["n_agents"], "agents,", d["total_steps"], "steps; entropy",
              c["entropy"]["median"][0], "->", c["entropy"]["median"][-1], "; reward",
              c["episode_reward"]["median"][1], "->", c["episode_reward"]["median"][-1])


if __name__ == "__main__":
    main()
