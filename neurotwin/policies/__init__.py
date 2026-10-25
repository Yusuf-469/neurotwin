"""Allocation policies (RESEARCH.md v2 §6.8-§6.9). The PPO policy is added in Stage 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import AllocationPolicy
from .decision_log import DecisionLog, LoggingPolicy
from .heuristics import (FCFSAllocator, LoadBalancingAllocator, PriorityAllocator,
                         TrunkReservationAllocator)
from .optimization import OptimizationAllocator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

POLICY_CLASSES: dict[str, type[AllocationPolicy]] = {
    cls.name: cls for cls in (FCFSAllocator, PriorityAllocator, LoadBalancingAllocator,
                              TrunkReservationAllocator, OptimizationAllocator)
}


def make_policy(name: str, params: dict[str, Any] | None = None) -> AllocationPolicy:
    """Build a baseline by name. Picklable inputs (name, params) make this usable from worker
    processes during tuning."""
    clean = {k: v for k, v in (params or {}).items() if not k.startswith("_")}
    return POLICY_CLASSES[name](**clean)


def load_baseline_params(tuned: bool = True) -> dict[str, dict[str, Any]]:
    """Default parameters, overridden by tuned values when ``tuned`` and the file exists."""
    params = json.loads((CONFIG_DIR / "policies_default.json").read_text(encoding="utf-8"))
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    tuned_path = CONFIG_DIR / "baselines_tuned.json"
    if tuned and tuned_path.exists():
        for name, entry in json.loads(tuned_path.read_text(encoding="utf-8")).items():
            if not name.startswith("_"):
                params[name] = {**params[name], **entry["params"]}
    return params


__all__ = ["AllocationPolicy", "FCFSAllocator", "PriorityAllocator", "LoadBalancingAllocator",
           "TrunkReservationAllocator", "OptimizationAllocator", "DecisionLog", "LoggingPolicy",
           "POLICY_CLASSES", "make_policy", "load_baseline_params"]
