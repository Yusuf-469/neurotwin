# Stage 2 Validation Report — Simulation Engine

**Date:** 2026-10-25
**Code:** `neurotwin/simulation/` (package version 0.2.0)
**Environment:** Python 3.11.9, numpy 2.4.6, scipy 1.17.1, Windows 11
**Scope:** Evidence that the discrete-event engine implements its stochastic model correctly (RESEARCH.md v2 §6.17), plus the pilot study that sets the warm-up length and checks the load calibration (§6.11, §6.20).

> These checks show **implementation correctness** in reduced configurations. They do **not** show that the model resembles any real hospital. All numbers below are simulation outputs.

## 1. How to reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest                                   # 50 tests (incl. slow)
.\.venv\Scripts\python.exe -m neurotwin.simulation.validation.report --heavy
.\.venv\Scripts\python.exe -m neurotwin.simulation.validation.warmup --days 70 --reps 30
```

Seeds: replications use seeds 0–29. Results are deterministic for a given seed.

## 2. Analytical validation (production engine, reduced configurations)

**Reduction:** homogeneous Poisson arrivals, one ward with c = 10 beds, exponential LOS (mean 10 h), no ED treatment stage, no staffing limit, no deterioration, a single priority class, and a non-idling placement policy. With ED capacity `Q_max`, the system is exactly M/M/c/K with K = c + Q_max. Without an ED cap, it is M/M/c.

**Why M/M/c/K instead of M/M/c/c (Erlang B):** a diverted patient is one who finds the ED full, and the ED always has a waiting room (`Q_max ≥ 1`), so a pure loss system cannot be configured. The formula module is tested to reduce exactly to Erlang B at K = c and to Erlang C as K → ∞ (`tests/test_queueing_formulas.py`).

30 replications per case; each replication is 4,000 h (30,000 h for ρ = 0.95) with a 10% warm-up.

| System | rho (offered) | Quantity | Analytical | Simulated mean | 95% CI | In CI |
|---|---|---|---|---|---|---|
| M/M/10 | 0.5 | W_q (h) | 0.0722 | 0.0803 | [0.0605, 0.1002] | yes |
| M/M/10 | 0.5 | P(wait) | 0.0361 | 0.0375 | [0.0324, 0.0426] | yes |
| M/M/10 | 0.8 | W_q (h) | 2.0459 | 2.1604 | [1.8620, 2.4589] | yes |
| M/M/10 | 0.8 | P(wait) | 0.4092 | 0.4155 | [0.3954, 0.4357] | yes |
| M/M/10 | 0.95 | W_q (h) | 16.5117 | 16.0418 | [14.4372, 17.6463] | yes |
| M/M/10 | 0.95 | P(wait) | 0.8256 | 0.8188 | [0.8077, 0.8299] | yes |
| M/M/10/15 | 0.9 | Blocking | 0.0613 | 0.0617 | [0.0580, 0.0654] | yes |
| M/M/10/15 | 0.9 | W_q (h) | 1.2628 | 1.2603 | [1.2066, 1.3139] | yes |
| M/M/10/15 | 0.9 | P(wait) | 0.4526 | 0.4565 | [0.4434, 0.4696] | yes |
| M/M/10/15 | 1.2 | Blocking | 0.2033 | 0.2045 | [0.1998, 0.2091] | yes |
| M/M/10/15 | 1.2 | W_q (h) | 2.5631 | 2.5733 | [2.5228, 2.6237] | yes |
| M/M/10/15 | 1.2 | P(wait) | 0.7630 | 0.7676 | [0.7593, 0.7759] | yes |
| M/M/10/12 | 1.0 | Blocking | 0.1501 | 0.1502 | [0.1457, 0.1546] | yes |
| M/M/10/12 | 1.0 | W_q (h) | 0.5300 | 0.5329 | [0.5195, 0.5463] | yes |
| M/M/10/12 | 1.0 | P(wait) | 0.3533 | 0.3575 | [0.3511, 0.3639] | yes |

**15 of 15 analytical values lie inside the 95% CI.** The regression tests use 99% CIs; the reason is given in `tests/test_validation_mmck.py`.

### Does the check have power? (mutation test)

The same M/M/10, ρ = 0.8 check was rerun with a deliberate model error. The analytical value was kept at the correct model's value.

| Injected error | W_q 99% CI | P(wait) 99% CI | Detected |
|---|---|---|---|
| None (correct model) | [1.758, 2.563] contains 2.046 | [0.388, 0.443] contains 0.409 | — |
| Mean LOS +10% | [4.613, 7.459] | [0.590, 0.662] | yes (both) |
| Arrival rate +5% | [2.651, 3.975] | [0.479, 0.543] | yes (both) |

A 5% error in a single input is detected, so passing the checks is informative.

### Other engine checks (all passing)

| Check | Test |
|---|---|
| Non-homogeneous arrivals match the hour-of-week rate profile (chi-square, 60 simulated weeks) | `test_arrivals.py` |
| Exact sample-path identity: ∫(number waiting) dt = Σ individual waits (unweighted, priority-weighted, ED boarding, measurement window). This is the identity behind Little's law **and** the RL reward | `test_engine.py` |
| CRN integrity: two different policies with the same seed see identical patients (arrival times and all hidden attributes) and identical staffing | `test_engine.py` |
| Non-vacuity: those two policies produce different outcomes | `test_engine.py` |
| Off-service placement multiplies LOS by `m_os`; deterioration produces escalation → ICU → step-down | `test_engine.py` |
| Invariants at every decision: free staffed beds consistent, ED never above capacity, policies cannot see hidden patient data | `test_engine.py` |
| Occupancy tracing does not change a run | `test_engine.py` |

## 3. Warm-up (Welch's method)

**Setup:** scenario A, default parameters, greedy placement policy, 30 replications × 70 days, hourly total occupancy averaged across replications.

**First attempt (1-day window) was misleading.** It suggested 11 days, but occupancy has a **weekly cycle** (electives arrive on weekdays only), and the weekly peaks kept rising until about week 4. Welch's method needs a smoothing window at least as long as the longest cycle.

**With a 1-week window**, the smoothed curve rises steeply for about 10 days, then slowly until day ~21. From then on it fluctuates between about 130 and 136 occupied beds. Selected values:

| Day | 0 | 7 | 14 | 21 | 28 | 35 | 42 | 49 | 56 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|
| Smoothed occupancy | 13.5 | 111.3 | 126.0 | 129.2 | 131.9 | 134.5 | 134.4 | 131.0 | 131.6 | 133.3 |

The reproducible rule "first day reaching 98% of the final-third level" gives **21 days**.

**Decision:** warm-up = 21 days (a whole number of weeks, so measurement always starts on the same weekday), followed by 14 measured days. The episode horizon is **35 days** (840 h).

**Consequence for RL (Stage 5):** acting through 21 unmeasured days per episode is wasteful. Stage 3 (digital-twin snapshots) will make it possible to start episodes from saved warm states, drawn from a pool generated with training seeds only. This is an option to evaluate, not a decision yet.

## 4. Load calibration (scenario A, greedy policy, same 30 × 70-day runs)

| Quantity | Value |
|---|---|
| Staffed occupancy at end of run: MED / SURG / ICU | 0.88 / 0.80 / 0.85 |
| Mean boarding time | 1.36 h |
| 90th-percentile boarding time | 2.25 h |
| Diversions per 100 emergency arrivals | 0.01 |
| Off-service placements per 100 general placements | 16.7 |

This is consistent with the ~85% staffed-occupancy target in RESEARCH.md §6.20. Scenario A is moderately loaded: congestion is present but rarely severe. The greedy policy places off-service immediately, which explains the high off-service rate. That is exactly the trade-off the study policies must manage.

A single-seed pilot of scenarios B–E (14-day horizon, 1-day warm-up, before the warm-up was fixed) showed congestion rising as intended: mean boarding 4.8 h (B), 7.6 h (C), 12.1 h (D) and 16.5 h (E), with diversions in all four. These figures are indicative only and will be re-measured with the corrected warm-up and 50 seeds in Stage 6.

## 5. Performance

A 14-day episode with the greedy policy runs in about 0.05 s, roughly 2.4 × 10⁷ times faster than real time. This far exceeds the 10× real-time requirement (PROJECT_SPEC §6).

## 6. Known limitations

- Validation covers the queueing core (arrivals, beds, waiting, diversion). The ED doctor stage, shift-based staffing and the step-down/escalation pathways are covered by mechanics tests and invariants, not by closed-form results (no simple closed forms exist for them).
- The calibration is a plausibility target, not a fit to data.
- The warm-up was determined with the greedy policy. Policies that hold beds may change the steady state slightly; the Welch check will be repeated for the final PPO policy in Stage 7.
