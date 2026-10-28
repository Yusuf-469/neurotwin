"""Decision-time snapshots for the dashboard's "Inside the agent" panel.

Wraps an RL policy inside a live session. At each decision it keeps the agent's exact inputs
(observation and action mask) and asks a reference rule what *it* would have done in the same
state. The rule is only consulted, never obeyed: the wrapped policy's decision is returned
unchanged, and rules are stateless, so asking them has no side effects (tested). The expensive
part (integrated gradients, see rl/introspect.py) runs only when a visitor opens a decision.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation

#: Snapshots kept per session (oldest dropped first); about 200 bytes each.
MAX_SNAPSHOTS = 5000


def _placement(sim: HospitalSimulation, decision: tuple[int, Unit] | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    rid, unit = decision
    r = next(r for r in sim.waiting_requests() if r.rid == rid)
    return {
        "rid": rid,
        "unit": unit.value,
        "priority": r.priority,
        "request_type": r.rtype.value,
        "primary_unit": r.primary_unit.value,
        "waited_hours": round(sim.now - r.created_at, 2),
    }


def compare(agent: dict[str, Any] | None, rule: dict[str, Any] | None) -> str:
    """How the agent's placement relates to the rule's in the same state."""
    if agent is None and rule is None:
        return "both_wait"
    if agent is None:
        return "agent_waits"
    if rule is None:
        return "rule_waits"
    if agent["rid"] == rule["rid"]:
        return "same" if agent["unit"] == rule["unit"] else "same_patient_other_ward"
    return "different_patient"


class InsightRecorder:
    """Policy wrapper that records decision-time inputs plus a reference rule's choice."""

    def __init__(self, policy: Any, rule: Any, keep: int = MAX_SNAPSHOTS) -> None:
        self.policy = policy
        self.rule = rule
        self.keep = keep
        self.snapshots: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
        self._count = 0

    @property
    def name(self) -> str:
        return getattr(self.policy, "name", type(self.policy).__name__)

    @property
    def last_details(self) -> dict[str, Any] | None:
        return getattr(self.policy, "last_details", None)

    @property
    def agents(self) -> list[Any]:
        return list(getattr(self.policy, "agents", None) or [self.policy.agent])

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        rule_choice = _placement(sim, self.rule.decide(sim))
        decision = self.policy.decide(sim)
        agent_choice = _placement(sim, decision)
        self.snapshots[self._count] = {
            "time": sim.now,
            "obs": np.array(self.policy.last_obs, dtype=np.float32, copy=True),
            "mask": np.array(self.policy.last_mask, dtype=bool, copy=True),
            "agent": agent_choice,
            "rule": rule_choice,
        }
        self._count += 1
        while len(self.snapshots) > self.keep:
            self.snapshots.popitem(last=False)
        return decision
