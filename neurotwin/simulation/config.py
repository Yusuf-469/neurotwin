"""Simulation and scenario configuration (RESEARCH.md v2 §6.1, §6.6, §6.22).

Design decisions
----------------
* Model parameters live in ``config/simulation_default.json``, not in code (AGENTS.md: all
  configuration parameters explicit). The dataclass has no defaults for model parameters so
  that the JSON file is the single source of truth.
* Unknown keys are rejected so a typo cannot silently fall back to a default.
* Scenarios (A-E) only scale arrivals, staffing and physical capacity. They never change the
  stochastic model itself, which keeps the train/test scenarios comparable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "simulation_default.json"

UNIT_NAMES = ("MED", "SURG", "ICU")
NUM_PRIORITIES = 5
_SUPPORTED_DISTS = ("lognormal", "exponential", "fixed")


def round_half_up(x: float) -> int:
    """Round to the nearest integer, halves up (Python's round() is half-to-even)."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class DistributionSpec:
    """A positive duration distribution, parameterised by mean and standard deviation (hours)."""

    dist: str
    mean: float
    sd: float = 0.0

    def __post_init__(self) -> None:
        if self.dist not in _SUPPORTED_DISTS:
            raise ValueError(f"unsupported distribution {self.dist!r}; use one of {_SUPPORTED_DISTS}")
        if self.mean < 0:
            raise ValueError("mean must be non-negative")
        if self.dist == "lognormal" and (self.mean <= 0 or self.sd <= 0):
            raise ValueError("lognormal needs mean > 0 and sd > 0")
        if self.dist == "exponential" and self.mean <= 0:
            raise ValueError("exponential needs mean > 0")

    @classmethod
    def from_value(cls, value: Any) -> "DistributionSpec":
        if isinstance(value, DistributionSpec):
            return value
        return cls(**value)


@dataclass(frozen=True)
class ScenarioConfig:
    """Operating-condition multipliers (RESEARCH.md §6.22)."""

    name: str
    arrival_multiplier: float = 1.0
    staff_multiplier: float = 1.0
    capacity_multiplier: float = 1.0


# Scenario values are fixed by RESEARCH.md v2 §6.22. A is the only training scenario.
SCENARIOS: dict[str, ScenarioConfig] = {
    "A": ScenarioConfig("A_normal"),
    "B": ScenarioConfig("B_high_demand", arrival_multiplier=1.35),
    "C": ScenarioConfig("C_staff_shortage", staff_multiplier=0.65),
    "D": ScenarioConfig("D_capacity_loss", capacity_multiplier=0.6),
    "E": ScenarioConfig("E_combined", arrival_multiplier=1.25, staff_multiplier=0.7,
                        capacity_multiplier=0.6),
}


@dataclass(frozen=True)
class SimulationConfig:
    """All model parameters. See ``config/simulation_default.json`` for meanings and defaults."""

    horizon_hours: float
    warmup_hours: float
    review_interval_hours: float
    shift_length_hours: float

    emergency_base_rate: float
    elective_base_rate: float
    hourly_profile: tuple[float, ...]
    weekly_profile: tuple[float, ...]
    elective_weekdays: tuple[int, ...]
    elective_hours: tuple[int, int]

    priority_probs_emergency: tuple[float, ...]
    priority_probs_elective: tuple[float, ...]
    icu_first_prob: tuple[float, ...]
    specialty_prob_med_emergency: float
    specialty_prob_med_elective: float

    beds: Mapping[str, int]
    nurses: Mapping[str, int]
    patients_per_nurse: Mapping[str, int]
    doctors: int
    staff_absence_prob: float
    ed_capacity: int | None

    ed_treatment: DistributionSpec | None
    general_los: DistributionSpec
    icu_los: DistributionSpec
    escalation_icu_los: DistributionSpec
    post_icu_general_los: DistributionSpec
    deterioration_hazard_per_day: float
    off_service_los_multiplier: float

    priority_weights: tuple[float, ...]
    step_down_priority: int
    escalation_priority: int

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationConfig":
        """Build from a plain mapping. Keys starting with '_' are comments and are ignored."""
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        unknown = set(clean) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        missing = known - set(clean)
        if missing:
            raise ValueError(f"missing configuration keys: {sorted(missing)}")
        return cls(**_coerce(clean))

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable mapping that ``from_dict`` reconstructs exactly."""
        out = asdict(self)
        for key in ("beds", "nurses", "patients_per_nurse"):
            out[key] = dict(getattr(self, key))
        return out

    def with_overrides(self, **overrides: Any) -> "SimulationConfig":
        """Return a copy with some fields replaced (values are coerced and validated)."""
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return replace(self, **_coerce(overrides))

    # ------------------------------------------------------------------ validation

    def __post_init__(self) -> None:
        if self.horizon_hours <= 0 or not 0 <= self.warmup_hours < self.horizon_hours:
            raise ValueError("need horizon > 0 and 0 <= warmup < horizon")
        if self.review_interval_hours <= 0 or self.shift_length_hours <= 0:
            raise ValueError("review interval and shift length must be positive")
        if self.emergency_base_rate < 0 or self.elective_base_rate < 0:
            raise ValueError("arrival rates must be non-negative")
        _check_profile(self.hourly_profile, 24, "hourly_profile")
        _check_profile(self.weekly_profile, 7, "weekly_profile")
        if not all(0 <= d <= 6 for d in self.elective_weekdays):
            raise ValueError("elective_weekdays must be in 0..6 (0 = Monday)")
        start, end = self.elective_hours
        if not 0 <= start < end <= 24:
            raise ValueError("elective_hours must satisfy 0 <= start < end <= 24")
        if self.elective_base_rate > 0 and not self.elective_weekdays:
            raise ValueError("elective arrivals need at least one elective weekday")
        for name in ("priority_probs_emergency", "priority_probs_elective"):
            _check_probs(getattr(self, name), name)
        if len(self.icu_first_prob) != NUM_PRIORITIES or not all(
                0 <= p <= 1 for p in self.icu_first_prob):
            raise ValueError("icu_first_prob needs 5 probabilities in [0, 1]")
        for name in ("specialty_prob_med_emergency", "specialty_prob_med_elective",
                     "staff_absence_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("beds", "nurses", "patients_per_nurse"):
            mapping = getattr(self, name)
            if set(mapping) != set(UNIT_NAMES) or any(v < 0 for v in mapping.values()):
                raise ValueError(f"{name} needs non-negative values for exactly {UNIT_NAMES}")
        if self.doctors < 0:
            raise ValueError("doctors must be non-negative")
        if self.ed_capacity is not None and self.ed_capacity < 1:
            raise ValueError("ed_capacity must be >= 1 or null (unlimited)")
        if self.deterioration_hazard_per_day < 0 or self.off_service_los_multiplier <= 0:
            raise ValueError("invalid deterioration hazard or off-service multiplier")
        if len(self.priority_weights) != NUM_PRIORITIES or any(w < 0 for w in self.priority_weights):
            raise ValueError("priority_weights needs 5 non-negative values")
        for name in ("step_down_priority", "escalation_priority"):
            if not 1 <= getattr(self, name) <= NUM_PRIORITIES:
                raise ValueError(f"{name} must be in 1..5")


def load_default_config() -> SimulationConfig:
    return SimulationConfig.from_json(DEFAULT_CONFIG_PATH)


# ---------------------------------------------------------------------- helpers

_DIST_FIELDS = ("ed_treatment", "general_los", "icu_los", "escalation_icu_los",
                "post_icu_general_los")
_TUPLE_FIELDS = ("hourly_profile", "weekly_profile", "elective_weekdays", "elective_hours",
                 "priority_probs_emergency", "priority_probs_elective", "icu_first_prob",
                 "priority_weights")


def _coerce(values: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(values)
    for key in _DIST_FIELDS:
        if key in out and out[key] is not None:
            out[key] = DistributionSpec.from_value(out[key])
    for key in _TUPLE_FIELDS:
        if key in out:
            out[key] = tuple(out[key])
    # Profiles are relative shapes; normalise to mean 1 so base rates are true mean rates.
    # Already-normalised profiles are left bit-for-bit unchanged, so a config that round-trips
    # through a snapshot reproduces exactly the same arrival rates.
    for key in ("hourly_profile", "weekly_profile"):
        if key in out:
            profile = out[key]
            total = sum(profile)
            if total <= 0:
                raise ValueError(f"{key} must have a positive sum")
            if abs(total / len(profile) - 1.0) > 1e-12:
                out[key] = tuple(len(profile) * v / total for v in profile)
    for key in ("beds", "nurses", "patients_per_nurse"):
        if key in out:
            out[key] = {k: int(v) for k, v in out[key].items()}
    return out


def _check_profile(profile: tuple[float, ...], length: int, name: str) -> None:
    if len(profile) != length or any(v < 0 for v in profile):
        raise ValueError(f"{name} needs {length} non-negative values")


def _check_probs(probs: tuple[float, ...], name: str) -> None:
    if len(probs) != NUM_PRIORITIES or any(p < 0 for p in probs) or abs(sum(probs) - 1) > 1e-9:
        raise ValueError(f"{name} needs {NUM_PRIORITIES} probabilities summing to 1")
