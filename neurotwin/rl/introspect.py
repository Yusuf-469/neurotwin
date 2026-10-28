"""Looking inside a trained agent's decision (dashboard "Inside the agent" panel).

For one recorded decision this computes, from the real networks:

* each agent's masked action probabilities, their mean (the ensemble's vote) and how many
  agents' own top choice agrees with the ensemble;
* integrated-gradients attributions (Sundararajan et al., 2017) of the agents' preference for
  the chosen action over the runner-up, to the 21 raw inputs.

Target: the mean over agents of (logit of chosen action - logit of runner-up). Logit margins are
used rather than probabilities because the agents are often near-certain (p > 0.99), where the
softmax is flat and probability gradients vanish; the margin also answers the natural question
"why this move rather than the next-best one?".

Baseline: the training-average observation (the mean of the agents' frozen observation
normalisers). Attribution i is (x_i - baseline_i) times the path-averaged gradient; they sum to
f(x) - f(baseline) up to the Riemann-sum error, which is returned so the dashboard can show it.
Attributions are a local account of which inputs pushed the preference up or down relative to
an average situation; they are not a causal proof.

The critic's value estimate is deliberately not reported: rewards are never positive, yet the
critics predict positive values in 14-24% of states (checked on a 28-day run of each agent).
PPO only needs the critic to rank states for its advantage estimates, so its absolute level is
not a calibrated forecast and would mislead if shown as one.

Everything here is read-only: nothing touches the simulation or the agents' weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .actions import Action
from .features import FEATURE_NAMES, WAIT_NORMALISER_HOURS, FeatureScales
from .ppo.agent import PPOAgent
from .ppo.normalize import IdentityNormalizer

IG_STEPS = 48
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _norm_torch(agent: PPOAgent, x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of the agent's frozen observation normaliser."""
    rms = agent.obs_rms
    if isinstance(rms, IdentityNormalizer):
        return x
    mean = torch.as_tensor(np.asarray(rms.mean), dtype=torch.float32)
    std = torch.sqrt(torch.as_tensor(np.asarray(rms.var), dtype=torch.float32) + 1e-8)
    return torch.clamp((x - mean) / std, -rms.clip, rms.clip)


def baseline_observation(agents: Sequence[PPOAgent]) -> np.ndarray:
    """Training-average raw observation: mean of the agents' normaliser means."""
    return np.mean([np.asarray(a.obs_rms.mean, dtype=np.float64) for a in agents], axis=0)


def _margin(agents: Sequence[PPOAgent], x: torch.Tensor, chosen: int, rival: int) -> torch.Tensor:
    """Mean over agents of logit(chosen) - logit(rival); both actions are valid, so unmasked."""
    return torch.stack(
        [(lambda lg: lg[:, chosen] - lg[:, rival])(a.model(_norm_torch(a, x))[0]) for a in agents]
    ).mean(0)


def integrated_gradients(
    agents: Sequence[PPOAgent],
    obs: np.ndarray,
    chosen: int,
    rival: int,
    baseline: np.ndarray | None = None,
    steps: int = IG_STEPS,
) -> tuple[np.ndarray, float, float]:
    """Attributions of the agents' preference for ``chosen`` over ``rival`` to each raw input.

    Returns (attributions, f(obs), f(baseline)). The midpoint Riemann sum over ``steps`` points
    approximates the path integral; completeness error = |sum(attr) - (f(x) - f(b))|.
    """
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    b = torch.as_tensor(
        np.asarray(baseline_observation(agents) if baseline is None else baseline, dtype=np.float32)
    )
    alphas = ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps).unsqueeze(1)
    path = (b + alphas * (x - b)).requires_grad_(True)
    (grads,) = torch.autograd.grad(_margin(agents, path, chosen, rival).sum(), path)
    attr = ((x - b) * grads.mean(0)).detach().numpy().astype(np.float64)
    with torch.no_grad():
        fe = _margin(agents, torch.stack([x, b]), chosen, rival)
    return attr, float(fe[0]), float(fe[1])


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def _count(scale: float) -> Callable[[float], str]:
    return lambda v: f"{max(0, round(v * scale))}"


@dataclass(frozen=True)
class InputGroup:
    """One human-readable input; time of day / day of week combine their sin and cos parts."""

    key: str
    label: str
    idx: tuple[int, ...]
    show: Callable[[np.ndarray], str] = field(compare=False)


def input_groups(scales: FeatureScales) -> list[InputGroup]:
    """Readable labels and value formatters for the 21 features (inverse of rl/features.py)."""
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}

    def one(name: str, label: str, fmt: Callable[[float], str]) -> InputGroup:
        return InputGroup(name, label, (i[name],), lambda o, k=i[name]: fmt(float(o[k])))

    def hour(o: np.ndarray) -> str:
        h = (math.atan2(o[i["hour_sin"]], o[i["hour_cos"]]) / (2 * math.pi) * 24) % 24
        return f"{int(h):02d}:{int((h % 1) * 60):02d}"

    def day(o: np.ndarray) -> str:
        d = (math.atan2(o[i["day_sin"]], o[i["day_cos"]]) / (2 * math.pi) * 7) % 7
        return DAYS[int(d) % 7]

    return [
        one("occ_MED", "Medical ward occupancy", _pct),
        one("occ_SURG", "Surgical ward occupancy", _pct),
        one("occ_ICU", "ICU occupancy", _pct),
        one("req_admit_general", "Patients waiting for a ward bed", _count(scales.q_max)),
        one("req_admit_icu", "Patients waiting for ICU", _count(scales.q_max)),
        one("req_step_down", "ICU patients ready to move to a ward", _count(scales.icu_beds)),
        one("req_escalation", "Ward patients who need ICU", _count(scales.icu_beds)),
        one("nurse_availability", "Nurses on shift (vs normal)", _pct),
        one("doctor_availability", "Doctors on shift (vs normal)", _pct),
        InputGroup("hour", "Time of day", (i["hour_sin"], i["hour_cos"]), hour),
        InputGroup("day", "Day of week", (i["day_sin"], i["day_cos"]), day),
        one("arrivals_1h", "Arrivals in the last hour", _count(scales.peak_rate)),
        one("arrivals_4h", "Arrivals in the last 4 hours", _count(4 * scales.peak_rate)),
        one("discharges_4h", "Discharges in the last 4 hours", _count(4 * scales.peak_rate)),
        one("waiting_high_priority", "Urgent patients waiting", _count(scales.q_max)),
        one("waiting_low_priority", "Less urgent patients waiting", _count(scales.q_max)),
        one("mean_elapsed_wait", "Average wait so far",
            lambda v: f"{v * WAIT_NORMALISER_HOURS:.1f} h"),
        one("diversions_24h", "Ambulances turned away (24 h)", _count(scales.q_max)),
        one("off_service_in_beds", "Patients in the wrong ward", _count(scales.general_beds)),
    ]


def explain_decision(
    agents: Sequence[PPOAgent],
    obs: np.ndarray,
    mask: np.ndarray,
    scales: FeatureScales,
    top: int = 8,
) -> dict[str, Any]:
    """Everything the "Inside the agent" panel shows for one decision (JSON-ready)."""
    obs = np.asarray(obs, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    per_agent = [a.action_probabilities(obs, mask) for a in agents]
    probs = np.mean(per_agent, axis=0)
    chosen = int(np.argmax(np.where(mask, probs, -1.0)))
    votes = sum(int(np.argmax(np.where(mask, p, -1.0))) == chosen for p in per_agent)
    others = [k for k in np.argsort(-probs) if mask[k] and k != chosen]

    attribution: dict[str, Any] | None = None
    if others:  # with a single valid move there is no choice to explain
        rival = int(others[0])
        attr, f_x, f_b = integrated_gradients(agents, obs, chosen, rival)
        rows = [
            {"key": g.key, "label": g.label, "value": g.show(obs),
             "contribution": round(float(sum(attr[k] for k in g.idx)), 4)}
            for g in input_groups(scales)
        ]
        rows.sort(key=lambda r: -abs(r["contribution"]))
        attribution = {
            "method": "integrated gradients",
            "target": "logit margin, chosen minus runner-up (mean over agents)",
            "runner_up": Action(rival).name,
            "baseline": "training-average situation",
            "steps": IG_STEPS,
            "margin": round(f_x, 3),
            "baseline_margin": round(f_b, 3),
            "completeness_error": round(abs(float(attr.sum()) - (f_x - f_b)), 4),
            "inputs": rows[:top],
            "other_inputs_total": round(float(sum(r["contribution"] for r in rows[top:])), 4),
        }
    return {
        "chosen_action": Action(chosen).name,
        "actions": [
            {"action": Action(k).name, "probability": round(float(probs[k]), 4),
             "per_agent": [round(float(p[k]), 4) for p in per_agent]}
            for k in range(len(probs)) if mask[k]
        ],
        "n_agents": len(agents),
        "agents_agreeing": int(votes),
        "attribution": attribution,
    }
