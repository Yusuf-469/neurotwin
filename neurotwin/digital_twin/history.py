"""State history: observable states recorded over a run (ARCHITECTURE.md §3.3)."""

from __future__ import annotations

import json
from pathlib import Path

from ..simulation.entities import Unit
from ..simulation.hospital import HospitalSimulation, PlacementPolicy
from .state import HospitalState, observe


class StateHistory:
    """Ordered list of observable states."""

    def __init__(self) -> None:
        self.states: list[HospitalState] = []

    def record(self, sim: HospitalSimulation) -> HospitalState:
        state = observe(sim)
        self.states.append(state)
        return state

    def __len__(self) -> int:
        return len(self.states)

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self.states], fh)


class RecordingPolicy:
    """Wraps a policy and records the observable state at every decision, before deciding.

    Recording is read-only, so wrapping a policy never changes its decisions (tested).
    """

    def __init__(self, policy: PlacementPolicy, history: StateHistory | None = None) -> None:
        self.policy = policy
        self.history = history or StateHistory()

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None:
        self.history.record(sim)
        return self.policy.decide(sim)
