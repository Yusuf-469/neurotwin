"""Policy-structure analysis (RESEARCH.md v2 §6.20, RQ4): does PPO reserve beds?

Reservation states: a primary-ward placement is feasible for some waiting general admission
(action 1 is valid), and every waiting admission request has priority 3-5 (no urgent patient
is waiting). In such a state HOLD means keeping a free bed empty for a possible urgent arrival.
For each reservation state we record b = free staffed beds in the ward that action 1 would use,
the policy's P(HOLD) and whether its deterministic choice is HOLD.

Threshold (trunk-reservation) structure means P(HOLD) is high when b is small and falls as b
grows. Monotonicity is tested by isotonic regression (pool-adjacent-violators, decreasing) and
Spearman's rank correlation, separately for scenario A (training) and E (combined stress).
The tuned trunk-reservation baseline has r = 0 (never holds), so any systematic holding is
learned behaviour not present in the best threshold baseline.

Run: python -m neurotwin.experiments.structure
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy import stats

from ..policies.ppo_policy import PPOPolicy
from ..rl.actions import Action, action_mask, action_to_placement
from ..rl.features import FeatureScales, features
from ..rl.ppo.masking import masked_distribution
from ..simulation.config import SCENARIOS, load_default_config
from ..simulation.entities import RequestType
from ..simulation.hospital import HospitalSimulation, run_episode
from .evaluate import DEFAULT_AGENTS, ROOT
from .seeds import TEST

OUT = ROOT / "docs" / "data" / "stage7"
SEEDS = list(TEST)[:20]
MAX_BIN = 8


def _record(args: tuple[int, str, int]) -> list[tuple[int, float, int]]:
    agent_idx, scenario, seed = args
    torch.set_num_threads(1)
    policy = PPOPolicy.load(DEFAULT_AGENTS / f"agent_{agent_idx}.pt")
    agent = policy.agent
    records: list[tuple[int, float, int]] = []

    class Recorder:
        scales = None

        def decide(self, sim):
            if self.scales is None:
                self.scales = FeatureScales.from_config(sim.config)
            mask = action_mask(sim)
            obs = features(sim, self.scales)
            with torch.no_grad():
                logits, _ = agent.model(torch.as_tensor(agent.obs_rms.normalize(obs)).unsqueeze(0))
                probs = masked_distribution(logits, torch.as_tensor(mask).unsqueeze(0)).probs[0]
            action = int(probs.argmax())
            if sim.now >= sim.config.warmup_hours and mask[Action.ADMIT_PRIMARY_PRIORITY]:
                admissions = [r for r in sim.waiting_requests()
                              if r.rtype in (RequestType.ADMIT_GENERAL, RequestType.ADMIT_ICU)]
                if admissions and all(r.priority >= 3 for r in admissions):
                    _, unit = action_to_placement(sim, Action.ADMIT_PRIMARY_PRIORITY)
                    records.append((sim.free_staffed(unit), float(probs[Action.HOLD]),
                                    int(action == Action.HOLD)))
            return None if action == Action.HOLD else action_to_placement(sim, action)

    run_episode(HospitalSimulation(load_default_config(), SCENARIOS[scenario], seed=seed), Recorder())
    return records


def isotonic_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit of a non-increasing sequence."""
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:                    # violates non-increasing
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, _, n in blocks])


def main() -> None:
    results = {}
    for scenario in ("A", "E"):
        jobs = [(a, scenario, s) for a in range(5) for s in SEEDS]
        with ProcessPoolExecutor(4) as ex:
            recs = [r for chunk in ex.map(_record, jobs) for r in chunk]
        free = np.array([r[0] for r in recs])
        p_hold = np.array([r[1] for r in recs])
        chose_hold = np.array([r[2] for r in recs])
        bins = defaultdict(list)
        for f, p, c in recs:
            bins[min(f, MAX_BIN)].append((p, c))
        keys = sorted(bins)
        mean_p = np.array([np.mean([p for p, _ in bins[k]]) for k in keys])
        counts = np.array([len(bins[k]) for k in keys])
        iso = isotonic_decreasing(mean_p, counts)
        sse_iso = float(np.sum(counts * (mean_p - iso) ** 2))
        sse_flat = float(np.sum(counts * (mean_p - np.average(mean_p, weights=counts)) ** 2))
        rho, p_rho = stats.spearmanr(free, p_hold)
        results[scenario] = {
            "n_states": len(recs),
            "bins": [{"free_beds": int(k) if k < MAX_BIN else f">={MAX_BIN}", "n": int(n),
                      "mean_p_hold": float(m),
                      "share_argmax_hold": float(np.mean([c for _, c in bins[k]]))}
                     for k, n, m in zip(keys, counts, mean_p)],
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "isotonic_r2_vs_flat": 1 - sse_iso / sse_flat if sse_flat > 0 else float("nan"),
            "overall_share_argmax_hold": float(chose_hold.mean()),
        }
        print(f"Scenario {scenario}: {len(recs)} reservation states; overall argmax-HOLD share "
              f"{chose_hold.mean():.3f}; Spearman rho(free beds, P(HOLD)) = {rho:.3f} (p={p_rho:.2g}); "
              f"isotonic (decreasing) R^2 = {results[scenario]['isotonic_r2_vs_flat']:.3f}")
        for b in results[scenario]["bins"]:
            print(f"   free beds {b['free_beds']!s:>4}: n={b['n']:6d}  mean P(HOLD)={b['mean_p_hold']:.3f}  "
                  f"argmax HOLD={b['share_argmax_hold']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "structure.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
