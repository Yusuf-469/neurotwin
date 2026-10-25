# Stage 3 Report: Digital Twin State Management

**Date:** 2026-10-25
**Code:** `neurotwin/digital_twin/`, plus snapshot, branch and clone support in `neurotwin/simulation/hospital.py`
**Tests:** `tests/test_digital_twin.py` (20 tests). Full suite: 70 passing.

## 1. What was built

| Capability | Where | Purpose |
|---|---|---|
| `HospitalState` / `observe(sim)` | `digital_twin/state.py` | Policy-visible state only (decision-time information, RESEARCH.md §6.2). JSON-serialisable for the API and dashboard |
| `StateHistory`, `RecordingPolicy` | `digital_twin/history.py` | State history over a run; recording never changes decisions |
| Exact snapshot / restore | `HospitalSimulation.to_snapshot / from_snapshot`, `digital_twin/snapshot_io.py` | Complete state (hidden attributes, RNG streams, event list, accumulators) to JSON / `.json.gz` |
| `clone()` | engine | Independent in-memory copy |
| `branch(seed)` | engine | What-if branch: same present, new future randomness (RESEARCH.md §6.21) |
| `WarmStartPool` | `digital_twin/warm_start.py` | Steady-state starting points for RL training episodes |
| Seed registry | `experiments/seeds.py` | Disjoint train / validation / test ranges, asserted in code |

Observable state and snapshots are kept apart on purpose. A snapshot contains hidden information (realised lengths of stay, for example) and must never be given to a policy.

## 2. Exactness

A run that is snapshotted mid-way, serialised to JSON, restored and continued produces **byte-identical** results to the uninterrupted run. Checked:
- with a non-idling policy and with a holding policy;
- with snapshots taken between epochs and inside an open decision epoch;
- in memory and through a gzip file.

The comparison covers every patient's attributes, discharge time and diversion flag, all metric lists, and all exact cost integrals.

Two details were needed for exactness:
- **Event sequence counter:** stored as an integer, because `itertools.count` cannot be serialised.
- **Arrival-profile normalisation:** made idempotent. Re-normalising an already normalised profile changed the last floating-point bits, which could flip a thinning decision.

## 3. What-if branching

`branch(seed)` keeps the present state (occupancy, staffing, waiting requests and history) and replaces all future randomness with streams for `seed` in a separate **branch namespace**. Branch seed 7 therefore never replays the future of episode seed 7 (tested). Re-drawn on branching:
- future arrivals and staffing draws;
- not-yet-started durations of patients already in the system (e.g. the ward stay of a patient still in the ICU);
- **durations already in progress**, re-drawn from the distribution **conditional on time already elapsed**: `T | T > e`, sampled by inverse CDF. For the exponential distribution this reduces to memorylessness (tested). For the lognormal, a two-sample KS test against rejection sampling passes at elapsed times of 60 h and 200 h for ward stays and 4 h for ED treatment.

Every active patient consumes a fixed number of uniforms in patient order. Two branches with the same seed are therefore identical whichever policy later runs, which gives CRN for what-if comparisons (tested).

### Why conditional resampling matters: the inspection paradox

Patients found in a bed at a random moment have **longer** stays than a typical patient, because long stays are more likely to be in progress (length-biased sampling). Treating an in-progress stay as if it had just started ignores both the time already served and this bias.

**Mutation test.** Scenario A, greedy policy; 30 warm-started episodes vs 30 full episodes from an empty start:

| Branch method | Time-avg boarding (patients) | Off-service per 100 | Welch t-test vs full episodes |
|---|---|---|---|
| Full episodes (reference) | 1.250 | 14.33 | — |
| Conditional resampling (implemented) | 1.183 | 14.75 | p = 0.85 / 0.86 (no detectable bias) |
| Naive restart of in-progress stays | 3.444 | 27.12 | p = 0.0018 / 2.7 × 10⁻⁵ (biased) |

The naive method nearly triples boarding and doubles off-service placement. The regression test `test_warm_start_episodes_are_not_biased_relative_to_full_episodes` detects it.

## 4. Warm starts for RL training

From an empty start, the hospital needs a 21-day warm-up (Stage 2 report §3). Training on full 35-day episodes would put 60% of the agent's experience in unmeasured, unrepresentative transient states.

**Decision:**
- **Training episodes** start from a `WarmStartPool`:
  - The pool holds states saved at the end of the warm-up of runs on **training seeds only** (asserted).
  - Each state was produced by a reference greedy placement policy.
  - Each episode branches a randomly chosen pool state with its own training seed.
- **Evaluation** (validation and test) does **not** use the pool. Test episodes follow the full protocol from an empty start, so reported results never depend on the reference warm-up policy.

**Evidence:** warm-started episodes reproduce the measured-window statistics of full episodes under the same policy (table above).

**Limitation:** pool states reflect the greedy policy's steady state. A learning agent's own steady state may differ, so a training episode starts somewhat off the agent's stationary distribution. If this matters, the pool can be regenerated with the trained agent in a later training phase. That option is recorded and not yet needed.

## 5. Not done in this stage (by design)

- The what-if *engine* (policy comparison over many branch seeds, with paired statistics) is Stage 8. Stage 3 provides the primitive (`branch`) and its correctness.
- Persistence in SQLite is not needed yet. Snapshots are files, and history is JSON. ARCHITECTURE.md §14 allows SQLite "when needed".
