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

import copy
import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Protocol

from .arrivals import ThinningArrivalProcess, elective_rate, emergency_rate
from .config import SCENARIOS, ScenarioConfig, SimulationConfig, round_half_up
from .distributions import conditional_residual, exponential_from_uniform, sample_duration
from .entities import GENERAL_UNITS, BedRequest, Patient, RequestType, RequestView, Unit
from .events import EventKind, EventQueue
from .rng import BRANCH_NAMESPACE, STREAM_NAMES, RandomStreams

# Uniforms consumed per patient from the patient_attributes stream (fixed for CRN alignment):
# priority, specialty, ICU pathway, ED treatment, ICU LOS, general LOS, escalation ICU LOS,
# post-ICU general LOS, deterioration delay.
_UNIFORMS_PER_PATIENT = 9
# Recent-event histories (arrivals, discharges, diversions) are kept for this many hours; the
# RL state uses 1 h, 4 h and 24 h windows (RESEARCH.md §6.5).
RECENT_WINDOW_HOURS = 24.0
SNAPSHOT_SCHEMA = "neurotwin.sim_snapshot/1"
_PATIENT_EVENTS = (
    EventKind.DISCHARGE,
    EventKind.ICU_PHASE_END,
    EventKind.DETERIORATION,
    EventKind.ED_TREATMENT_END,
)


@dataclass
class Accumulators:
    """Exact time integrals and counters. One instance covers [0, now] (used for rewards), a
    second covers only the measurement window [warmup, horizon] (used for metrics)."""

    weighted_request_hours: float = 0.0  # integral of sum of omega over waiting requests
    request_hours: float = 0.0  # integral of number of waiting requests
    ed_boarding_hours: float = 0.0  # integral of waiting admission requests from the ED
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
        return (
            100.0 * self.diversions / self.emergency_arrivals
            if self.emergency_arrivals
            else math.nan
        )

    @property
    def off_service_rate_per_100(self) -> float:
        return (
            100.0 * self.off_service_placements / self.general_placements
            if self.general_placements
            else math.nan
        )

    @property
    def mean_ed_wait_hours(self) -> float:
        return _mean(self.ed_waits)


class HospitalSimulation:
    """One simulated episode for a given configuration, scenario and seed."""

    def __init__(
        self,
        config: SimulationConfig,
        scenario: ScenarioConfig | None = None,
        seed: int = 0,
        trace_interval_hours: float | None = None,
    ) -> None:
        """``trace_interval_hours`` optionally records occupancy snapshots (warm-up analysis).
        Tracing uses no random numbers and never triggers decisions, so it cannot change a run."""
        self.config = config
        self.scenario = scenario or SCENARIOS["A"]
        self.seed = seed
        self.branch_seed: int | None = None  # set when this run is a what-if branch
        self._streams = RandomStreams(seed)
        self._attr_rng = self._streams["patient_attributes"]
        self._staff_rng = self._streams["staffing"]

        sc = self.scenario
        self.physical_beds = {
            u: round_half_up(config.beds[u.value] * sc.capacity_multiplier) for u in Unit
        }
        self._nominal_nurses = {
            u: round_half_up(config.nurses[u.value] * sc.staff_multiplier) for u in Unit
        }
        self._nominal_doctors = round_half_up(config.doctors * sc.staff_multiplier)
        self._em_arrivals = ThinningArrivalProcess(
            emergency_rate(config, sc), self._streams["arrivals_emergency"]
        )
        self._el_arrivals = ThinningArrivalProcess(
            elective_rate(config, sc), self._streams["arrivals_elective"]
        )

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
        self._recent: dict[str, deque[float]] = {
            "arrivals": deque(),
            "discharges": deque(),
            "diversions": deque(),
        }

        self._epoch_open = False
        self._finished = False
        self._trace_interval = trace_interval_hours
        self.trace: list[tuple[float, dict[Unit, int], int]] = []  # (time, occupancy, waiting)
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

    def recent_event_times(self, kind: str) -> tuple[float, ...]:
        """Times of recent 'arrivals' (emergency and elective, incl. diverted), 'discharges'
        or 'diversions' within the last RECENT_WINDOW_HOURS (past information only)."""
        self._prune_recent()
        return tuple(self._recent[kind])

    @property
    def doctor_queue_length(self) -> int:
        return len(self._doctor_queue)

    @property
    def doctors_busy(self) -> int:
        return self._doctors_busy

    @property
    def off_service_in_beds(self) -> int:
        return sum(
            1
            for p in self._patients
            if p.off_service and p.location is not None and p.location is not Unit.ICU
        )

    def has_feasible_placement(self) -> bool:
        general_free = any(self.free_staffed(u) > 0 for u in GENERAL_UNITS)
        return (self._n_waiting_general > 0 and general_free) or (
            self._n_waiting_icu > 0 and self.free_staffed(Unit.ICU) > 0
        )

    # ================================================================== public: control

    def next_epoch(self, until: float | None = None) -> bool:
        """Advance until a decision epoch opens (True) or the horizon is reached (False).

        With ``until``, also stop (returning False, ``finished`` still False) once the next
        event would be after ``until``; time is then advanced exactly to ``until``. Used to
        take snapshots at a chosen time. Calling this ends the current epoch: any requests
        still waiting are held.
        """
        self._epoch_open = False
        if self._finished:
            return False
        horizon = self.config.horizon_hours
        limit = horizon if until is None else min(until, horizon)
        if limit < self.now:
            raise ValueError("cannot advance to a time in the past")
        while True:
            t_next = self._queue.peek_time()
            if t_next is None or t_next > limit:
                break
            t, kind, payload = self._queue.pop()
            self._advance_to(t)
            if self._handle(kind, payload) and self.has_feasible_placement():
                self._epoch_open = True
                return True
        self._advance_to(limit)
        if limit >= horizon:
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
            self._occupied[patient.location] -= 1  # source bed is freed
        self._occupied[unit] += 1
        patient.location = unit
        patient.activity_start = self.now

        if unit is Unit.ICU:
            patient.off_service = False
            stay = (
                patient.escalation_icu_los
                if request.rtype is RequestType.ESCALATION
                else patient.icu_los
            )
            self._queue.push(self.now + stay, EventKind.ICU_PHASE_END, patient)
            off_service = False
        else:
            off_service = unit is not patient.specialty
            patient.off_service = off_service
            los = patient.post_icu_general_los if patient.has_deteriorated else patient.general_los
            if off_service:
                los *= self.config.off_service_los_multiplier
            if not patient.has_deteriorated and patient.deterioration_delay < los:
                self._queue.push(
                    self.now + patient.deterioration_delay, EventKind.DETERIORATION, patient
                )
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
            scenario=self.scenario.name,
            seed=self.seed,
            measured_hours=measured_hours,
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

    # ================================================================== digital twin support
    #
    # The full simulation state (including hidden patient attributes, RNG stream states and the
    # event list) can be cloned, serialised and branched. These live in the engine because they
    # need its internals; the digital_twin package builds file I/O, observable state, history
    # and warm-start pools on top of them (ARCHITECTURE.md §3.3).

    def clone(self) -> "HospitalSimulation":
        """Independent exact copy: continuing it reproduces this run exactly."""
        return copy.deepcopy(self)

    def to_snapshot(self) -> dict[str, Any]:
        """Complete state as a JSON-serialisable dict (Python JSON: allows Infinity/NaN).

        Restoring with ``from_snapshot`` and continuing gives results identical to continuing
        this object (tested). Contains hidden information, so it must never be given to a policy.
        """

        def pid(p: Patient | None) -> int | None:
            return None if p is None else p.pid

        return {
            "schema": SNAPSHOT_SCHEMA,
            "config": self.config.to_dict(),
            "scenario": asdict(self.scenario),
            "seed": self.seed,
            "branch_seed": self.branch_seed,
            "now": self.now,
            "epoch_open": self._epoch_open,
            "finished": self._finished,
            "rng": {name: self._streams[name].bit_generator.state for name in STREAM_NAMES},
            "rng_namespace": self._streams.namespace,
            "patients": [_patient_to_dict(p) for p in self._patients],
            "requests": [_request_to_dict(r) for r in self._request_log],
            "waiting": list(self._requests),
            "events": [[t, int(k), s, pid(p)] for t, k, s, p in self._queue.entries()],
            "event_next_seq": self._queue.next_seq,
            "occupied": {u.value: n for u, n in self._occupied.items()},
            "nurses_on_shift": {u.value: n for u, n in self._nurses_on_shift.items()},
            "doctors_on_shift": self._doctors_on_shift,
            "doctors_busy": self._doctors_busy,
            "doctor_queue": [[pr, at, p_id] for pr, at, p_id, _ in self._doctor_queue],
            "next_rid": self._next_rid,
            "n_waiting_general": self._n_waiting_general,
            "n_waiting_icu": self._n_waiting_icu,
            "n_boarding": self._n_boarding,
            "weight_sum": self._weight_sum,
            "acc": vars(self.acc).copy(),
            "measured": vars(self._measured).copy(),
            "m_emergency_arrivals": self._m_emergency_arrivals,
            "m_general_placements": self._m_general_placements,
            "boarding_times": list(self._boarding_times),
            "elective_waits": list(self._elective_waits),
            "ed_waits": list(self._ed_waits),
            "recent": {k: list(v) for k, v in self._recent.items()},
            "trace_interval": self._trace_interval,
            "trace": [[t, {u.value: n for u, n in occ.items()}, w] for t, occ, w in self.trace],
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "HospitalSimulation":
        if snap.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError(f"unsupported snapshot schema {snap.get('schema')!r}")
        sim = cls(
            SimulationConfig.from_dict(snap["config"]),
            ScenarioConfig(**snap["scenario"]),
            seed=snap["seed"],
            trace_interval_hours=snap["trace_interval"],
        )
        # Everything __init__ set up dynamically is overwritten below.
        if snap["rng_namespace"] != 0:
            sim._set_streams(RandomStreams(snap["branch_seed"], snap["rng_namespace"]))
        for name, state in snap["rng"].items():
            sim._streams[name].bit_generator.state = state
        sim.branch_seed = snap["branch_seed"]
        sim.now = snap["now"]
        sim._epoch_open = snap["epoch_open"]
        sim._finished = snap["finished"]
        patients = [_patient_from_dict(d) for d in snap["patients"]]
        sim._patients = patients
        sim._request_log = [_request_from_dict(d, patients) for d in snap["requests"]]
        by_rid = {r.rid: r for r in sim._request_log}
        sim._requests = {rid: by_rid[rid] for rid in snap["waiting"]}
        sim._queue.replace_all(
            [
                (t, EventKind(k), s, None if p_id is None else patients[p_id])
                for t, k, s, p_id in snap["events"]
            ],
            snap["event_next_seq"],
        )
        sim._occupied = {Unit(u): n for u, n in snap["occupied"].items()}
        sim._nurses_on_shift = {Unit(u): n for u, n in snap["nurses_on_shift"].items()}
        sim._doctors_on_shift = snap["doctors_on_shift"]
        sim._doctors_busy = snap["doctors_busy"]
        sim._doctor_queue = [
            (pr, at, p_id, patients[p_id]) for pr, at, p_id in snap["doctor_queue"]
        ]
        sim._next_rid = snap["next_rid"]
        sim._n_waiting_general = snap["n_waiting_general"]
        sim._n_waiting_icu = snap["n_waiting_icu"]
        sim._n_boarding = snap["n_boarding"]
        sim._weight_sum = snap["weight_sum"]
        sim.acc = Accumulators(**snap["acc"])
        sim._measured = Accumulators(**snap["measured"])
        sim._m_emergency_arrivals = snap["m_emergency_arrivals"]
        sim._m_general_placements = snap["m_general_placements"]
        sim._boarding_times = list(snap["boarding_times"])
        sim._elective_waits = list(snap["elective_waits"])
        sim._ed_waits = list(snap["ed_waits"])
        sim._recent = {k: deque(v) for k, v in snap["recent"].items()}
        sim.trace = [(t, {Unit(u): n for u, n in occ.items()}, w) for t, occ, w in snap["trace"]]
        return sim

    def branch(
        self, seed: int, resample_in_progress: bool = True, scenario: ScenarioConfig | None = None
    ) -> "HospitalSimulation":
        """What-if branch (RESEARCH.md §6.21): same present, new future randomness.

        With ``scenario``, the branch also switches to new operating conditions from now on:
        future arrival rates, nominal staffing (applied from the next shift change) and physical
        beds. If beds are removed while occupied, the patients stay and no new placements are
        possible in that unit until occupancy falls below capacity (same rule as staffing).

        The returned simulation shares this run's present state but draws all *future*
        randomness from streams for ``seed`` in the branch namespace: future arrivals, staffing
        draws and every not-yet-started duration of patients already in the system. With
        ``resample_in_progress`` (default), durations already in progress (ED treatment, ICU
        stay, ward stay) are also re-drawn from their distribution *conditional on the time
        already elapsed*, because the true remaining time is unknown at the branch point.

        Branches with the same seed are identical whichever policy is later used (CRN), so
        policies can be compared from the same state on the same futures.
        """
        if self._finished:
            raise RuntimeError("cannot branch a finished simulation")
        sim = self.clone()
        sim.branch_seed = seed
        if scenario is not None:
            sim._apply_scenario(scenario)
        sim._set_streams(RandomStreams(seed, BRANCH_NAMESPACE))
        entries = [
            e
            for e in sim._queue.entries()
            if e[1] not in (EventKind.EMERGENCY_ARRIVAL, EventKind.ELECTIVE_ARRIVAL)
        ]
        entries = sim._redraw_hidden_attributes(entries, resample_in_progress)
        sim._queue.replace_all(entries, sim._queue.next_seq)
        sim._push_next_arrival(sim._em_arrivals, sim.now, EventKind.EMERGENCY_ARRIVAL)
        sim._push_next_arrival(sim._el_arrivals, sim.now, EventKind.ELECTIVE_ARRIVAL)
        return sim

    def set_measurement_window(self, start: float, end: float) -> None:
        """Measure from ``start`` to ``end`` (what-if analysis): replaces warm-up and horizon,
        and resets every measured accumulator, counter and metric list. Only valid before any
        part of the new window has elapsed (start >= now)."""
        if start < self.now or end <= start:
            raise ValueError("need now <= start < end")
        self.config = self.config.with_overrides(warmup_hours=start, horizon_hours=end)
        self._measured = Accumulators()
        self._m_emergency_arrivals = 0
        self._m_general_placements = 0
        self._boarding_times, self._elective_waits, self._ed_waits = [], [], []
        self._finished = False

    def _apply_scenario(self, scenario: ScenarioConfig) -> None:
        cfg = self.config
        self.scenario = scenario
        self.physical_beds = {
            u: round_half_up(cfg.beds[u.value] * scenario.capacity_multiplier) for u in Unit
        }
        self._nominal_nurses = {
            u: round_half_up(cfg.nurses[u.value] * scenario.staff_multiplier) for u in Unit
        }
        self._nominal_doctors = round_half_up(cfg.doctors * scenario.staff_multiplier)

    def _set_streams(self, streams: RandomStreams) -> None:
        self._streams = streams
        self._attr_rng = streams["patient_attributes"]
        self._staff_rng = streams["staffing"]
        self._em_arrivals = ThinningArrivalProcess(
            emergency_rate(self.config, self.scenario), streams["arrivals_emergency"]
        )
        self._el_arrivals = ThinningArrivalProcess(
            elective_rate(self.config, self.scenario), streams["arrivals_elective"]
        )

    def _redraw_hidden_attributes(self, entries: list, resample_in_progress: bool) -> list:
        """Re-draw future (and optionally in-progress) durations of patients in the system.

        Every active patient consumes exactly _UNIFORMS_PER_PATIENT uniforms in pid order, so
        the redraw is identical for every policy that later runs from the branch (CRN).
        Attributes already realised (priority, specialty, pathway) and completed activities
        are kept.
        """
        cfg = self.config
        pending = {p.pid: i for i, (_, k, _, p) in enumerate(entries) if k in _PATIENT_EVENTS}
        in_doctor_queue = {pid for _, _, pid, _ in self._doctor_queue}
        entries = list(entries)
        for patient in self._patients:
            if patient.diverted or not math.isnan(patient.discharged_at):
                continue
            u = self._attr_rng.random(_UNIFORMS_PER_PATIENT)
            idx = pending.get(patient.pid)
            kind = entries[idx][1] if idx is not None else None
            elapsed = self.now - patient.activity_start if idx is not None else 0.0
            in_progress = resample_in_progress and idx is not None
            loc, det = patient.location, patient.has_deteriorated
            in_general = loc is not None and loc is not Unit.ICU
            placed = loc is not None

            def fresh(spec, ui):
                return sample_duration(spec, ui)

            # ED treatment
            if patient.pid in in_doctor_queue and cfg.ed_treatment is not None:
                patient.ed_treatment_time = fresh(cfg.ed_treatment, u[3])
            elif kind is EventKind.ED_TREATMENT_END and in_progress:
                r = conditional_residual(cfg.ed_treatment, elapsed, u[3])
                patient.ed_treatment_time = elapsed + r
                entries[idx] = (self.now + r,) + tuple(entries[idx][1:])
            # Initial ICU stay (ICU-first pathway, before any deterioration)
            if not placed:
                patient.icu_los = fresh(cfg.icu_los, u[4])
            elif loc is Unit.ICU and not det and kind is EventKind.ICU_PHASE_END and in_progress:
                r = conditional_residual(cfg.icu_los, elapsed, u[4])
                patient.icu_los = elapsed + r
                entries[idx] = (self.now + r,) + tuple(entries[idx][1:])
            # Escalation ICU stay
            if not det or (det and in_general):
                patient.escalation_icu_los = fresh(cfg.escalation_icu_los, u[6])
            elif loc is Unit.ICU and kind is EventKind.ICU_PHASE_END and in_progress:
                r = conditional_residual(cfg.escalation_icu_los, elapsed, u[6])
                patient.escalation_icu_los = elapsed + r
                entries[idx] = (self.now + r,) + tuple(entries[idx][1:])
            # Ward stays and deterioration
            mult = cfg.off_service_los_multiplier if patient.off_service else 1.0
            ward_in_progress = in_general and kind in (EventKind.DISCHARGE, EventKind.DETERIORATION)
            if not det:
                if ward_in_progress and in_progress:
                    base_elapsed = elapsed / mult
                    patient.general_los = base_elapsed + conditional_residual(
                        cfg.general_los, base_elapsed, u[5]
                    )
                    # Constant hazard: the residual time to deterioration is memoryless.
                    patient.deterioration_delay = elapsed + exponential_from_uniform(
                        cfg.deterioration_hazard_per_day / 24.0, u[8]
                    )
                    los = patient.general_los * mult
                    if patient.deterioration_delay < los:
                        entries[idx] = (
                            patient.activity_start + patient.deterioration_delay,
                            EventKind.DETERIORATION,
                        ) + tuple(entries[idx][2:])
                    else:
                        entries[idx] = (patient.activity_start + los, EventKind.DISCHARGE) + tuple(
                            entries[idx][2:]
                        )
                elif not ward_in_progress:
                    patient.general_los = fresh(cfg.general_los, u[5])
                    patient.deterioration_delay = exponential_from_uniform(
                        cfg.deterioration_hazard_per_day / 24.0, u[8]
                    )
                patient.post_icu_general_los = fresh(cfg.post_icu_general_los, u[7])
            else:
                if in_general and kind is EventKind.DISCHARGE and in_progress:
                    base_elapsed = elapsed / mult
                    patient.post_icu_general_los = base_elapsed + conditional_residual(
                        cfg.post_icu_general_los, base_elapsed, u[7]
                    )
                    entries[idx] = (
                        patient.activity_start + patient.post_icu_general_los * mult,
                    ) + tuple(entries[idx][1:])
                elif not (in_general and kind is EventKind.DISCHARGE):
                    patient.post_icu_general_los = fresh(cfg.post_icu_general_los, u[7])
        return entries

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
        self._prune_recent()

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
            self._recent["discharges"].append(self.now)
            return True
        if kind is EventKind.ICU_PHASE_END:
            self._create_request(
                payload, RequestType.STEP_DOWN, self.config.step_down_priority, from_ed=False
            )
            return True
        if kind is EventKind.DETERIORATION:
            payload.has_deteriorated = True
            self._create_request(
                payload, RequestType.ESCALATION, self.config.escalation_priority, from_ed=False
            )
            return True
        if kind is EventKind.SHIFT_CHANGE:
            self._on_shift_change()
            return True
        if kind is EventKind.PERIODIC_REVIEW:
            self._queue.push(
                self.now + self.config.review_interval_hours, EventKind.PERIODIC_REVIEW
            )
            return True
        if kind is EventKind.TRACE:
            self.trace.append(
                (self.now, dict(self._occupied), self._n_waiting_general + self._n_waiting_icu)
            )
            self._queue.push(self.now + self._trace_interval, EventKind.TRACE)
            return False
        raise AssertionError(f"unhandled event kind {kind}")

    def _on_emergency_arrival(self) -> bool:
        patient = self._new_patient(is_emergency=True)
        self._push_next_arrival(self._em_arrivals, self.now, EventKind.EMERGENCY_ARRIVAL)
        self._recent["arrivals"].append(self.now)
        measured = self.now >= self.config.warmup_hours
        if measured:
            self._m_emergency_arrivals += 1
        cap = self.config.ed_capacity
        if cap is not None and self.ed_count >= cap:
            patient.diverted = True
            self.acc.diversions += 1
            self._recent["diversions"].append(self.now)
            if measured:
                self._measured.diversions += 1
            return False
        if self.config.ed_treatment is None:
            self._create_admission_request(patient, from_ed=True)
            return True
        heapq.heappush(
            self._doctor_queue, (patient.priority, patient.arrival_time, patient.pid, patient)
        )
        self._start_treatments()
        return False

    def _on_elective_arrival(self) -> bool:
        patient = self._new_patient(is_emergency=False)
        self._push_next_arrival(self._el_arrivals, self.now, EventKind.ELECTIVE_ARRIVAL)
        self._recent["arrivals"].append(self.now)
        self._create_admission_request(patient, from_ed=False)
        return True

    def _on_shift_change(self) -> None:
        # Fixed draw order (MED, SURG, ICU, doctors) every shift keeps the staffing stream
        # aligned across policies. Each nominal staff member is absent independently.
        keep = 1.0 - self.config.staff_absence_prob
        for unit in Unit:
            self._nurses_on_shift[unit] = int(
                self._staff_rng.binomial(self._nominal_nurses[unit], keep)
            )
        self._doctors_on_shift = int(self._staff_rng.binomial(self._nominal_doctors, keep))
        self._queue.push(self.now + self.config.shift_length_hours, EventKind.SHIFT_CHANGE)
        self._start_treatments()

    def _start_treatments(self) -> None:
        # ED treatment order is a fixed rule (priority, then arrival) for every policy: ED
        # treatment is not a decision under study (RESEARCH.md §6.22, ED wait is secondary).
        while self._doctor_queue and self._doctors_busy < self._doctors_on_shift:
            _, _, _, patient = heapq.heappop(self._doctor_queue)
            self._doctors_busy += 1
            patient.activity_start = self.now
            if patient.arrival_time >= self.config.warmup_hours:
                self._ed_waits.append(self.now - patient.arrival_time)
            self._queue.push(
                self.now + patient.ed_treatment_time, EventKind.ED_TREATMENT_END, patient
            )

    # ================================================================== internals: entities

    def _prune_recent(self) -> None:
        cutoff = self.now - RECENT_WINDOW_HOURS
        for times in self._recent.values():
            while times and times[0] < cutoff:
                times.popleft()

    def _push_next_arrival(
        self, process: ThinningArrivalProcess, t: float, kind: EventKind
    ) -> None:
        t_next = process.next_arrival(t)
        if math.isfinite(t_next) and t_next <= self.config.horizon_hours:
            self._queue.push(t_next, kind)

    def _new_patient(self, is_emergency: bool) -> Patient:
        cfg = self.config
        u = self._attr_rng.random(_UNIFORMS_PER_PATIENT)
        probs = cfg.priority_probs_emergency if is_emergency else cfg.priority_probs_elective
        priority = _categorical(probs, u[0]) + 1
        p_med = (
            cfg.specialty_prob_med_emergency if is_emergency else cfg.specialty_prob_med_elective
        )
        specialty = Unit.MED if u[1] < p_med else Unit.SURG
        icu_first = bool(u[2] < cfg.icu_first_prob[priority - 1])
        ed_time = sample_duration(cfg.ed_treatment, u[3]) if cfg.ed_treatment is not None else 0.0
        patient = Patient(
            pid=len(self._patients),
            arrival_time=self.now,
            is_emergency=is_emergency,
            priority=priority,
            specialty=specialty,
            icu_first=icu_first,
            ed_treatment_time=ed_time,
            icu_los=sample_duration(cfg.icu_los, u[4]),
            general_los=sample_duration(cfg.general_los, u[5]),
            escalation_icu_los=sample_duration(cfg.escalation_icu_los, u[6]),
            post_icu_general_los=sample_duration(cfg.post_icu_general_los, u[7]),
            deterioration_delay=exponential_from_uniform(
                cfg.deterioration_hazard_per_day / 24.0, u[8]
            ),
        )
        self._patients.append(patient)
        return patient

    def _create_admission_request(self, patient: Patient, from_ed: bool) -> None:
        rtype = RequestType.ADMIT_ICU if patient.icu_first else RequestType.ADMIT_GENERAL
        self._create_request(patient, rtype, patient.priority, from_ed=from_ed)

    def _create_request(
        self, patient: Patient, rtype: RequestType, priority: int, from_ed: bool
    ) -> None:
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
            self._weight_sum = 0.0  # remove floating-point drift when the queue empties


class PlacementPolicy(Protocol):
    """Anything that, at an open epoch, returns a placement (rid, unit) or None for HOLD."""

    def decide(self, sim: HospitalSimulation) -> tuple[int, Unit] | None: ...


def run_until(sim: HospitalSimulation, policy: PlacementPolicy, until: float | None = None) -> None:
    """Run ``sim`` with ``policy`` until time ``until`` (or the horizon if None).

    If an epoch is already open (e.g. a branch taken at a decision point), the policy decides
    in it first.
    """
    while True:
        if sim.epoch_open:
            while sim.has_feasible_placement():
                decision = policy.decide(sim)
                if decision is None:
                    break
                sim.place(*decision)
        if not sim.next_epoch(until):
            return


def run_episode(sim: HospitalSimulation, policy: PlacementPolicy) -> SimulationResult:
    """Run ``sim`` to its horizon with ``policy`` making every decision."""
    run_until(sim, policy)
    return sim.result()


# ---------------------------------------------------------------------- helpers


def _patient_to_dict(p: Patient) -> dict[str, Any]:
    d = asdict(p)
    d["specialty"] = p.specialty.value
    d["location"] = None if p.location is None else p.location.value
    return d


def _patient_from_dict(d: dict[str, Any]) -> Patient:
    d = dict(d)
    d["specialty"] = Unit(d["specialty"])
    d["location"] = None if d["location"] is None else Unit(d["location"])
    return Patient(**d)


def _request_to_dict(r: BedRequest) -> dict[str, Any]:
    return {
        "rid": r.rid,
        "pid": r.patient.pid,
        "rtype": r.rtype.value,
        "priority": r.priority,
        "created_at": r.created_at,
        "from_ed": r.from_ed,
        "placed_at": r.placed_at,
        "unit": None if r.unit is None else r.unit.value,
    }


def _request_from_dict(d: dict[str, Any], patients: list[Patient]) -> BedRequest:
    return BedRequest(
        d["rid"],
        patients[d["pid"]],
        RequestType(d["rtype"]),
        d["priority"],
        d["created_at"],
        d["from_ed"],
        d["placed_at"],
        None if d["unit"] is None else Unit(d["unit"]),
    )


def _categorical(probs: tuple[float, ...], u: float) -> int:
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return i
    return len(probs) - 1  # guards against probabilities summing to 1 - eps


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
