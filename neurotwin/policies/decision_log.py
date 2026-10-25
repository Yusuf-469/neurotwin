"""Decision logging (ROADMAP Stage 4; input to the post-hoc explanation layer, Stage 10).

Every decision, including HOLD while placements were feasible, is logged with the observable
context it was made in. The log contains only decision-time information plus the decision, so
explanations can be generated from facts actually available to the policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation, PlacementPolicy


class DecisionLog:
    """Ordered list of decision records with JSONL export."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def save_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record) + "\n")


class LoggingPolicy:
    """Wraps a policy and logs each decision. Logging is read-only (tested)."""

    def __init__(self, policy: PlacementPolicy, log: DecisionLog | None = None) -> None:
        self.policy = policy
        self.log = log or DecisionLog()

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        from ..rl.actions import Action, action_mask  # local import avoids a package cycle

        waiting = sim.waiting_requests()
        feasible = [Action(a).name for a, ok in enumerate(action_mask(sim)) if ok]
        decision = self.policy.decide(sim)
        record: dict[str, Any] = {
            "decision_id": len(self.log.records),
            "time": sim.now,
            "policy": getattr(self.policy, "name", type(self.policy).__name__),
            "free_staffed": {u.value: sim.free_staffed(u) for u in Unit},
            "n_waiting": len(waiting),
            "n_waiting_high_priority": sum(r.priority <= 2 for r in waiting),
            "waiting_by_priority": {
                str(p): sum(r.priority == p for r in waiting) for p in range(1, 6)
            },
            "oldest_wait_hours": max((sim.now - r.created_at for r in waiting), default=0.0),
            "feasible_actions": feasible,
        }
        details = getattr(self.policy, "last_details", None)
        if details:
            record["policy_details"] = details
        if decision is None:
            record["action"] = "HOLD"
        else:
            rid, unit = decision
            request = next(r for r in waiting if r.rid == rid)
            record.update(
                {
                    "action": "PLACE",
                    "rid": rid,
                    "request_type": request.rtype.value,
                    "priority": request.priority,
                    "waited_hours": sim.now - request.created_at,
                    "unit": unit.value,
                    "primary_unit": request.primary_unit.value,
                    "off_service": unit is not Unit.ICU and unit is not request.primary_unit,
                }
            )
        self.log.records.append(record)
        return decision
