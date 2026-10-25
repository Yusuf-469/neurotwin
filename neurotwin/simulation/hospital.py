"""Discrete-event hospital simulation engine (RESEARCH.md v2 §6.1-§6.7).

Design decisions
----------------
* **Continuous time in hours, event-driven.** No fixed time step; the event list drives time.
* **Placement primitives, not policies.** The engine exposes ``place(request, unit)`` and
  decision epochs. Action semantics (RESEARCH.md §6.3) and policies live in other packages and
  act through these primitives, so every policy faces identical mechanics.
* **Decision epochs** (RESEARCH.md §6.2): after a trigger event (new bed request, freed bed,
  shift change, periodic review), an epoch opens if at least one placement is feasible. Within
  an epoch the policy places requests one at a time (atomic decisions, Δt = 0) until it stops
  (HOLD) or nothing more is feasible.
* **Exact cost accounting.** Between events the set of waiting requests is constant, so the
  time integrals used by the reward (weighted request-hours, RESEARCH.md §6.10) and by the
  metrics are accumulated exactly as (sum of weights) x (event gap). No sampling error.
* **CRN.** All patient attributes are drawn at arrival from their own stream with a fixed
  number of uniforms per patient (see ``_new_patient``). Staffing draws happen at fixed shift
  times. No draw depends on a policy decision.

Simplifying assumptions (documented, RESEARCH.md §6.1/§6.6)
-----------------------------------------------------------
* Arrivals are patients who will need an inpatient bed; ED patients sent home are not modelled.
* Triage is instantaneous at arrival; priority is drawn then.
* A patient deteriorates at most once; after escalation and ICU, the general phase uses
  ``post_icu_general_los`` and cannot deteriorate again.
* If staffing falls below current occupancy at a shift change, patients already in beds stay;
  no new placements are possible in that unit until occupancy drops below staffed capacity.
* Electives are not subject to ED diversion and do not count towards ED occupancy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import heapq

from .arrivals import ThinningArrivalProcess, elective_rate, emergency_rate
from .config import SCENARIOS, ScenarioConfig, SimulationConfig, round_half_up
from .distributions import exponential_from_uniform, sample_duration
from .entities import GENERAL_UNITS, BedRequest, Patient, RequestType, RequestView, Unit
from .events import EventKind, EventQueue
from .rng import RandomStreams

# Uniforms consumed per patient from the patient_attributes stream (fixed for CRN alignment):
# priority, specialty, ICU pathway, ED treatment, ICU LOS, general LOS, escalation ICU LOS,
# post-ICU general LOS, deterioration delay.
_UNIFORMS_PER_PATIENT = 9


@dataclass
class Accumulators:
    """Exact time integrals and counters. One instance covers [0, now] (used for rewards), a
    second covers only the measurement window [warmup, horizon] (used for metrics)."""

    weighted_request_hours: float = 0.0   # integral of sum of omega over waiting requests
    request_hours: float = 0.0            # integral of number of waiting requests
    ed_boarding_hours: float = 0.0        # integral of waiting admission requests from the ED
    diversions: int = 0
    placements: int = 0
    off_service_placements: int = 0

    def add_time(self, dt: float, weight_sum: float, n_waiting: int, n_boarding: int) -> None:
        self.weighted_request_hours += weight_sum * dt
        self.request_hours += n_waiting * dt
        self.ed_boarding_hours += n_boarding * dt

    def copy(self) -> "Accumulators":
        return Accumulators(**vars(self))


@dataclass
class SimulationResult:
    """Metrics over the measurement window [warmup, horizon] (RESEARCH.md §6.22)."""

    scenario: str
    seed: int
    measured_hours: float
    emergency_arrivals: int
    diversions: int
    boarding_times: list[float] = field(repr=False)
    elective_waits: list[float] = field(repr=False)
    ed_waits: list[float] = field(repr=False)
    general_placements: int = 0
    off_service_placements: int = 0
    censored_requests: int = 0
    time_avg_boarding: float = 0.0

    @property
    def mean_boarding_hours(self) -> float:
        return _mean(self.boarding_times)

    @property
    def p90_boarding_hours(self) -> float:
        return _quantile(self.boarding_times, 0.9)

    @property
    def diversion_rate_per_100(self) -> float:
        return 100.0 * self.diversions / self.emergency_arrivals if self.emergency_arrivals else math.nan

    @property
    def off_service_rate_per_100(self) -> float:
        return (100.0 * self.off_service_placements / self.general_placements
                if self.general_placements else math.nan)

    @property
    def mean_ed_wait_hours(self) -> float:
        return _mean(self.ed_waits)


class HospitalSimulation:
    """One simulated episode for a given configuration, scenario and seed."""

    def __init__(self, config: SimulationConfig, scenario: ScenarioConfig | None = None,
                 seed: int = 0, trace_interval_hours: float | None = None) -> None:
        """``trace_interval_hours`` optionally records occupancy snapshots (warm-up analysis).
        Tracing uses no random numbers and never triggers decisions, so it cannot change a run."""
        self.config = config
        self.scenario = scenario or SCENARIOS["A"]
        self.seed = seed
        self._streams = RandomStreams(seed)
        self._attr_rng = self._streams["patient_attributes"]
        self._staff_rng = self._streams["staffing"]

        sc = self.scenario
        self.physical_beds = {u: round_half_up(config.beds[u.value] * sc.capacity_multiplier)
                              for u in Unit}
        self._nominal_nurses = {u: round_half_up(config.nurses[u.value] * sc.staff_multiplier)
                                for u in Unit}
        self._nominal_doctors = round_half_up(config.doctors * sc.staff_multiplier)
        self._em_arrivals = ThinningArrivalProcess(emergency_rate(config, sc),
                                                   self._streams["arrivals_emergency"])
        self._el_arrivals = ThinningArrivalProcess(elective_rate(config, sc),
                                                   self._streams["arrivals_elective"])

        self.now = 0.0
        self._queue = EventQueue()
        self._patients: list[Patient] = []
        self._occupied = {u: 0 for u in Unit}
        self._nurses_on_shift = {u: 0 for u in Unit}
        self._doctors_on_shift = 0
        self._doctors_busy = 0
        self._doctor_queue: list[tuple[int, float, int, Patient]] = []

        self._requests: dict[int, BedRequest] = {}
        self._request_log: list[BedRequest] = []
        self._next_rid = 0
        self._n_waiting_general = 0
        self._n_waiting_icu = 0
        self._n_boarding = 0
        self._weight_sum = 0.0

        self.acc = Accumulators()
        self._measured = Accumulators()
        self._m_emergency_arrivals = 0
        self._m_general_placements = 0
        self._boarding_times: list[float] = []
        self._elective_waits: list[float] = []
        self._ed_waits: list[float] = []

        self._epoch_open = False
        self._finished = False
        self._trace_interval = trace_interval_hours
        self.trace: list[tuple[float, dict[Unit, int], int]] = []   # (time, occupancy, waiting)
        if trace_interval_hours is not None:
            if trace_interval_hours <= 0:
                raise ValueError("trace interval must be positive")
            self._queue.push(trace_interval_hours, EventKind.TRACE)

        # Initial events. Shift change at t = 0 draws the first shift's staffing.
        self._queue.push(0.0, EventKind.SHIFT_CHANGE)
        self._push_next_arrival(self._em_arrivals, 0.0, EventKind.EMERGENCY_ARRIVAL)
        self._push_next_arrival(self._el_arrivals, 0.0, EventKind.ELECTIVE_ARRIVAL)
        self._queue.push(config.review_interval_hours, EventKind.PERIODIC_REVIEW)

    # ================================================================== public: queries

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def epoch_open(self) -> bool:
        return self._epoch_open

    def staffed_capacity(self, unit: Unit) -> int:
        ratio = self.config.patients_per_nurse[unit.value]
        return min(self.physical_beds[unit], self._nurses_on_shift[unit] * ratio)

    def occupied(self, unit: Unit) -> int:
        return self._occupied[unit]

    def free_staffed(self, unit: Unit) -> int:
        return max(0, self.staffed_capacity(unit) - self._occupied[unit])

    @property
    def ed_count(self) -> int:
        """Patients in the ED: waiting for a doctor, in treatment, or boarding."""
        return len(self._doctor_queue) + self._doctors_busy + self._n_boarding

    @property
    def doctors_on_shift(self) -> int:
        return self._doctors_on_shift

    @property
    def nurses_on_shift(self) -> dict[Unit, int]:
        return dict(self._nurses_on_shift)

    @property
    def nominal_staff(self) -> tuple[dict[Unit, int], int]:
        return dict(self._nominal_nurses), self._nominal_doctors

    def waiting_requests(self) -> list[RequestView]:
        """Waiting requests in creation order (only decision-time information)."""
        return [r.view() for r in self._requests.values()]

    def has_feasible_placement(self) -> bool:
        general_free = any(self.free_staffed(u) > 0 for u in GENERAL_UNITS)
        return ((self._n_waiting_general > 0 and general_free)
                or (self._n_waiting_icu > 0 and self.free_staffed(Unit.ICU) > 0))

    # ================================================================== public: control

    def next_epoch(self) -> bool:
        """Advance until a decision epoch opens (True) or the horizon is reached (False).

        Calling this also ends the current epoch: any requests still waiting are held.
        """
        self._epoch_open = False
        if self._finished:
            return False
        horizon = self.config.horizon_hours
        while True:
            t_next = self._queue.peek_time()
            if t_next is None or t_next > horizon:
                break
            t, kind, payload = self._queue.pop()
            self._advance_to(t)
            if self._handle(kind, payload) and self.has_feasible_placement():
                self._epoch_open = True
                return True
        self._advance_to(horizon)
        self._finished = True
        return False

    def place(self, rid: int, unit: Unit) -> None:
        """Place waiting request ``rid`` in ``unit`` at the current time (atomic decision)."""
        if not self._epoch_open:
            raise RuntimeError("placements are only allowed during an open decision epoch")
        request = self._requests.get(rid)
        if request is None:
            raise KeyError(f"request {rid} is not waiting")
        if unit not in request.eligible_units:
            raise ValueError(f"unit {unit.value} is not eligible for request {rid}")
        if self.free_staffed(unit) <= 0:
            raise ValueError(f"no free staffed bed in {unit.value}")

        self._remove_request(request)
        request.placed_at = self.now
        request.unit = unit
        patient = request.patient
        if request.rtype in (RequestType.STEP_DOWN, RequestType.ESCALATION):
            self._occupied[patient.location] -= 1          # source bed is freed
        self._occupied[unit] += 1
        patient.location = unit

        if unit is Unit.ICU:
            patient.off_service = False
            stay = (patient.escalation_icu_los if request.rtype is RequestType.ESCALATION
                    else patient.icu_los)
            self._queue.push(self.now + stay, EventKind.ICU_PHASE_END, patient)
            off_service = False
        else:
            off_service = unit is not patient.specialty
            patient.off_service = off_service
            los = (patient.post_icu_general_los if patient.has_deteriorated
                   else patient.general_los)
            if off_service:
                los *= self.config.off_service_los_multiplier
            if not patient.has_deteriorated and patient.deterioration_delay < los:
                self._queue.push(self.now + patient.deterioration_delay,
                                 EventKind.DETERIORATION, patient)
            else:
                self._queue.push(self.now + los, EventKind.DISCHARGE, patient)

        self.acc.placements += 1
        self.acc.off_service_placements += off_service
        if self.now >= self.config.warmup_hours:
            self._measured.placements += 1
            self._measured.off_service_placements += off_service
            if unit is not Unit.ICU:
                self._m_general_placements += 1
        if request.created_at >= self.config.warmup_hours:
            wait = self.now - request.created_at
            if request.rtype in (RequestType.ADMIT_GENERAL, RequestType.ADMIT_ICU):
                (self._boarding_times if request.from_ed else self._elective_waits).append(wait)

    # ================================================================== public: results

    def result(self) -> SimulationResult:
        if not self._finished:
            raise RuntimeError("simulation has not reached the horizon")
        warmup = self.config.warmup_hours
        censored = sum(1 for r in self._requests.values() if r.created_at >= warmup)
        measured_hours = self.config.horizon_hours - warmup
        return SimulationResult(
            scenario=self.scenario.name, seed=self.seed, measured_hours=measured_hours,
            emergency_arrivals=self._m_emergency_arrivals,
            diversions=self._measured.diversions,
            boarding_times=list(self._boarding_times),
            elective_waits=list(self._elective_waits),
            ed_waits=list(self._ed_waits),
            general_placements=self._m_general_placements,
            off_service_placements=self._measured.off_service_placements,
            censored_requests=censored,
            time_avg_boarding=self._measured.ed_boarding_hours / measured_hours,
        )

    @property
    def measured(self) -> Accumulators:
        return self._measured.copy()

    def request_log(self) -> Iterator[BedRequest]:
        """All requests ever created (for accounting tests and analysis, not for policies)."""
        return iter(self._request_log)

    def patient_log(self) -> Iterator[Patient]:
        """All patients ever generated, including diverted ones (tests and analysis only)."""
        return iter(self._patients)

    # ================================================================== internals: time

    def _advance_to(self, t: float) -> None:
        dt = t - self.now
        if dt < 0:
            raise RuntimeError("time cannot go backwards")
        if dt > 0:
            n_waiting = self._n_waiting_general + self._n_waiting_icu
            self.acc.add_time(dt, self._weight_sum, n_waiting, self._n_boarding)
            lo = max(self.now, self.config.warmup_hours)
            hi = min(t, self.config.horizon_hours)
            if hi > lo:
                self._measured.add_time(hi - lo, self._weight_sum, n_waiting, self._n_boarding)
        self.now = t

    # ================================================================== internals: events

    def _handle(self, kind: EventKind, payload: Patient | None) -> bool:
        """Process one event; return True if it is a decision trigger (RESEARCH.md §6.2)."""
        if kind is EventKind.EMERGENCY_ARRIVAL:
            return self._on_emergency_arrival()
        if kind is EventKind.ELECTIVE_ARRIVAL:
            return self._on_elective_arrival()
        if kind is EventKind.ED_TREATMENT_END:
            self._doctors_busy -= 1
            self._create_admission_request(payload, from_ed=True)
            self._start_treatments()
            return True
        if kind is EventKind.DISCHARGE:
            self._occupied[payload.location] -= 1
            payload.location = None
            payload.discharged_at = self.now
            return True
        if kind is EventKind.ICU_PHASE_END:
            self._create_request(payload, RequestType.STEP_DOWN,
                                 self.config.step_down_priority, from_ed=False)
            return True
        if kind is EventKind.DETERIORATION:
            payload.has_deteriorated = True
            self._create_request(payload, RequestType.ESCALATION,
                                 self.config.escalation_priority, from_ed=False)
            return True
        if kind is EventKind.SHIFT_CHANGE:
            self._on_shift_change()
            return True
        if kind is EventKind.PERIODIC_REVIEW:
            self._queue.push(self.now + self.config.review_interval_hours,
                             EventKind.PERIODIC_REVIEW)
            return True
        if kind is EventKind.TRACE:
            self.trace.append((self.now, dict(self._occupied),
                               self._n_waiting_general + self._n_waiting_icu))
            self._queue.push(self.now + self._trace_interval, EventKind.TRACE)
            return False
        raise AssertionError(f"unhandled event kind {kind}")

    def _on_emergency_arrival(self) -> bool:
        patient = self._new_patient(is_emergency=True)
        self._push_next_arrival(self._em_arrivals, self.now, EventKind.EMERGENCY_ARRIVAL)
        measured = self.now >= self.config.warmup_hours
        if measured:
            self._m_emergency_arrivals += 1
        cap = self.config.ed_capacity
        if cap is not None and self.ed_count >= cap:
            patient.diverted = True
            self.acc.diversions += 1
            if measured:
                self._measured.diversions += 1
            return False
        if self.config.ed_treatment is None:
            self._create_admission_request(patient, from_ed=True)
            return True
        heapq.heappush(self._doctor_queue,
                       (patient.priority, patient.arrival_time, patient.pid, patient))
        self._start_treatments()
        return False

    def _on_elective_arrival(self) -> bool:
        patient = self._new_patient(is_emergency=False)
        self._push_next_arrival(self._el_arrivals, self.now, EventKind.ELECTIVE_ARRIVAL)
        self._create_admission_request(patient, from_ed=False)
        return True

    def _on_shift_change(self) -> None:
        # Fixed draw order (MED, SURG, ICU, doctors) every shift keeps the staffing stream
        # aligned across policies. Each nominal staff member is absent independently.
        keep = 1.0 - self.config.staff_absence_prob
        for unit in Unit:
            self._nurses_on_shift[unit] = int(self._staff_rng.binomial(self._nominal_nurses[unit], keep))
        self._doctors_on_shift = int(self._staff_rng.binomial(self._nominal_doctors, keep))
        self._queue.push(self.now + self.config.shift_length_hours, EventKind.SHIFT_CHANGE)
        self._start_treatments()

    def _start_treatments(self) -> None:
        # ED treatment order is a fixed rule (priority, then arrival) for every policy: ED
        # treatment is not a decision under study (RESEARCH.md §6.22, ED wait is secondary).
        while self._doctor_queue and self._doctors_busy < self._doctors_on_shift:
            _, _, _, patient = heapq.heappop(self._doctor_queue)
            self._doctors_busy += 1
            if patient.arrival_time >= self.config.warmup_hours:
                self._ed_waits.append(self.now - patient.arrival_time)
            self._queue.push(self.now + patient.ed_treatment_time,
                             EventKind.ED_TREATMENT_END, patient)

    # ================================================================== internals: entities

    def _push_next_arrival(self, process: ThinningArrivalProcess, t: float,
                           kind: EventKind) -> None:
        t_next = process.next_arrival(t)
        if math.isfinite(t_next) and t_next <= self.config.horizon_hours:
            self._queue.push(t_next, kind)

    def _new_patient(self, is_emergency: bool) -> Patient:
        cfg = self.config
        u = self._attr_rng.random(_UNIFORMS_PER_PATIENT)
        probs = cfg.priority_probs_emergency if is_emergency else cfg.priority_probs_elective
        priority = _categorical(probs, u[0]) + 1
        p_med = cfg.specialty_prob_med_emergency if is_emergency else cfg.specialty_prob_med_elective
        specialty = Unit.MED if u[1] < p_med else Unit.SURG
        icu_first = bool(u[2] < cfg.icu_first_prob[priority - 1])
        ed_time = (sample_duration(cfg.ed_treatment, u[3])
                   if cfg.ed_treatment is not None else 0.0)
        patient = Patient(
            pid=len(self._patients), arrival_time=self.now, is_emergency=is_emergency,
            priority=priority, specialty=specialty, icu_first=icu_first,
            ed_treatment_time=ed_time,
            icu_los=sample_duration(cfg.icu_los, u[4]),
            general_los=sample_duration(cfg.general_los, u[5]),
            escalation_icu_los=sample_duration(cfg.escalation_icu_los, u[6]),
            post_icu_general_los=sample_duration(cfg.post_icu_general_los, u[7]),
            deterioration_delay=exponential_from_uniform(
                cfg.deterioration_hazard_per_day / 24.0, u[8]),
        )
        self._patients.append(patient)
        return patient

    def _create_admission_request(self, patient: Patient, from_ed: bool) -> None:
        rtype = RequestType.ADMIT_ICU if patient.icu_first else RequestType.ADMIT_GENERAL
        self._create_request(patient, rtype, patient.priority, from_ed=from_ed)

    def _create_request(self, patient: Patient, rtype: RequestType, priority: int,
                        from_ed: bool) -> None:
        request = BedRequest(self._next_rid, patient, rtype, priority, self.now, from_ed)
        self._next_rid += 1
        self._requests[request.rid] = request
        self._request_log.append(request)
        if rtype.needs_icu:
            self._n_waiting_icu += 1
        else:
            self._n_waiting_general += 1
        if from_ed:
            self._n_boarding += 1
        self._weight_sum += self.config.priority_weights[priority - 1]

    def _remove_request(self, request: BedRequest) -> None:
        del self._requests[request.rid]
        if request.rtype.needs_icu:
            self._n_waiting_icu -= 1
        else:
            self._n_waiting_general -= 1
        if request.from_ed:
            self._n_boarding -= 1
        self._weight_sum -= self.config.priority_weights[request.priority - 1]
        if not self._requests:
            self._weight_sum = 0.0      # remove floating-point drift when the queue empties


class PlacementPolicy(Protocol):
    """Anything that, at an open epoch, returns a placement (rid, unit) or None for HOLD."""

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None: ...


def run_episode(sim: HospitalSimulation, policy: PlacementPolicy) -> SimulationResult:
    """Run ``sim`` to its horizon with ``policy`` making every decision."""
    while sim.next_epoch():
        while sim.has_feasible_placement():
            decision = policy.decide(sim)
            if decision is None:
                break
            sim.place(*decision)
    return sim.result()


# ---------------------------------------------------------------------- helpers

def _categorical(probs: tuple[float, ...], u: float) -> int:
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return i
    return len(probs) - 1           # guards against probabilities summing to 1 - eps


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
