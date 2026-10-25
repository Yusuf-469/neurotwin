"""Discrete-event hospital simulation (RESEARCH.md v2 §6.1-§6.7, §6.17)."""

from .config import (
    SCENARIOS,
    DistributionSpec,
    ScenarioConfig,
    SimulationConfig,
    load_default_config,
)
from .entities import GENERAL_UNITS, RequestType, RequestView, Unit
from .hospital import (
    Accumulators,
    HospitalSimulation,
    PlacementPolicy,
    SimulationResult,
    run_episode,
    run_until,
)

__all__ = [
    "SCENARIOS",
    "DistributionSpec",
    "ScenarioConfig",
    "SimulationConfig",
    "load_default_config",
    "GENERAL_UNITS",
    "RequestType",
    "RequestView",
    "Unit",
    "Accumulators",
    "HospitalSimulation",
    "PlacementPolicy",
    "SimulationResult",
    "run_episode",
    "run_until",
]
