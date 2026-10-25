"""Patients, bed requests and units (RESEARCH.md v2 §6.1; ARCHITECTURE.md §5.1).

Patients carry *hidden* attributes (LOS, treatment time, deterioration delay) that are drawn at
arrival for Common Random Numbers. Policies never see Patient objects; they see RequestView,
which contains only information available at decision time (RESEARCH.md §6.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Unit(str, Enum):
    MED = "MED"
    SURG = "SURG"
    ICU = "ICU"


GENERAL_UNITS: tuple[Unit, ...] = (Unit.MED, Unit.SURG)


class RequestType(Enum):
    """Bed request types (RESEARCH.md §6.1 "Bed requests").

    ADMIT_GENERAL / ADMIT_ICU correspond to the "ED general" / "ED ICU" rows of the spec and
    also cover elective admissions (``from_ed`` distinguishes them).
    """

    ADMIT_GENERAL = "admit_general"
    ADMIT_ICU = "admit_icu"
    STEP_DOWN = "step_down"
    ESCALATION = "escalation"

    @property
    def needs_icu(self) -> bool:
        return self in (RequestType.ADMIT_ICU, RequestType.ESCALATION)


@dataclass(slots=True)
class Patient:
    pid: int
    arrival_time: float
    is_emergency: bool
    priority: int                      # 1 = most urgent
    specialty: Unit                    # primary ward: MED or SURG
    icu_first: bool
    # Hidden attributes (never exposed to policies).
    ed_treatment_time: float
    icu_los: float
    general_los: float
    escalation_icu_los: float
    post_icu_general_los: float
    deterioration_delay: float         # hours after a general placement; inf = never
    # Dynamic state.
    location: Unit | None = None
    off_service: bool = False
    has_deteriorated: bool = False
    diverted: bool = False
    discharged_at: float = math.nan


@dataclass(slots=True)
class BedRequest:
    rid: int
    patient: Patient
    rtype: RequestType
    priority: int
    created_at: float
    from_ed: bool
    placed_at: float = math.nan
    unit: Unit | None = None

    @property
    def eligible_units(self) -> tuple[Unit, ...]:
        # General requests may go to either general ward (primary or off-service).
        return (Unit.ICU,) if self.rtype.needs_icu else GENERAL_UNITS

    @property
    def primary_unit(self) -> Unit:
        return Unit.ICU if self.rtype.needs_icu else self.patient.specialty

    def view(self) -> "RequestView":
        return RequestView(self.rid, self.rtype, self.priority, self.created_at,
                           self.primary_unit, self.eligible_units, self.from_ed)


@dataclass(frozen=True, slots=True)
class RequestView:
    """What a policy may know about a waiting request (no LOS, no future information)."""

    rid: int
    rtype: RequestType
    priority: int
    created_at: float
    primary_unit: Unit
    eligible_units: tuple[Unit, ...] = field(default=())
    from_ed: bool = True
