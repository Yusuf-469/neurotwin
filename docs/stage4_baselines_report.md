# Stage 4 Report: Baseline Policies, Scenario Recalibration and Tuning

**Date:** 2026-10-26
**Code:** `neurotwin/policies/`, `neurotwin/experiments/{objective,load,tuning}.py`
**Config:** `config/policies_default.json` (untuned starting values), `config/baselines_tuned.json` (tuned, with provenance), `config/reward_default.json` (shared cost weights)
**Tests:** 94 passing (`tests/test_policies.py` adds 23)

> All numbers are simulation results from scenario A validation seeds. They are not test-set results and say nothing about real hospitals.

## 1. Baselines implemented

| Policy | Rule |
|---|---|
| FCFS | Oldest request first; primary ward if free; off-service after waiting W_os |
| Priority | Most urgent first (ties by wait); same off-service rule |
| Load-Balancing | Priority order; off-service when the primary ward's staffed occupancy exceeds the other's by more than θ_lb |
| Trunk-Reservation | Priority order; priority 3–5 placed in unit k only if more than r_k staffed beds are free (Miller, 1969) |
| MIP | Exact myopic transportation problem (`scipy.optimize.milp`), re-solved at every atomic decision |

All policies act only through the engine's public queries and placement primitive, and never see hidden patient data. A request that cannot be placed does not block requests for other bed types.

**MIP exactness:** on 300 random instances, the MILP optimum equals both independent oracles:
- the LP relaxation, whose solution is integral, as total unimodularity predicts;
- the Hungarian algorithm on expanded bed slots.

**Decision log:** `LoggingPolicy` records every placement and every HOLD with its observable context (free beds, queue mix, wait, off-service flag), as JSONL. This is the input to the post-hoc explanation layer. Logging is read-only (tested).

## 2. Scenario recalibration (a design correction, made before any RL work)

### What the first baseline runs showed

With the v2.1 scenarios, one validation seed per scenario:
- **Scenario A was almost congestion-free under sensible policies:** mean boarding 0.03–0.19 h. Stage 2 had calibrated A to ~85% occupancy under a *greedy* policy that overflows immediately; off-service stays are 10% longer, which inflated congestion. Under sensible policies there was little to decide, so an RL agent trained on A would learn little.
- **The stress scenarios were hopelessly overloaded.** In E, every policy diverted 42–59% of arrivals and the cost was dominated by diversions. No policy can matter much there, so robustness comparisons would be uninformative.

### The fix: calibrate on offered load, not on policy outcomes

Choosing scenarios by looking at which policy wins would let the scenario choice favour a policy. Instead, `neurotwin/experiments/load.py` computes the **analytical offered load** ρ from model parameters alone:
- expected bed-hours per patient (including deterioration, computed by quadrature);
- expected staffed beds (exact over the binomial distribution of nurse absences).

**Targets were stated before examining any policy result:**
- A ≈ 0.93: heavily loaded but stable.
- B, C, D ≈ 1.02: just past saturation, the heavy-traffic regime where allocation matters most.
- E ≈ 1.10.

| Scenario | v2.1 multipliers → ρ | v2.3 multipliers → ρ general / ICU |
|---|---|---|
| A | base → 0.879 | base rates +5.8% → 0.930 / 0.821 |
| B | arrivals 1.35 → 1.187 | arrivals 1.10 → 1.023 / 0.903 |
| C | staff 0.65 → 1.328 | staff 0.88 → 1.068 / 0.985 |
| D | capacity 0.6 → 1.410 | capacity 0.88 → 1.020 / 0.878 |
| E | (1.25, 0.7, 0.6) → 1.769 | (1.08, 0.90, 0.90) → 1.098 / 1.064 |

C cannot hit 1.02 exactly: nurse counts are integers, so 0.90× gives 0.989 and anything from 0.84× to 0.89× gives 1.068. The old values are kept as a severe-overload extension.

### Consequences
- **Warm-up re-measured.** Welch's method (30 × 84 days, 1-week window) at ρ = 0.93 suggests 29 days. Heavier load relaxes more slowly. Rounded up to whole weeks, the warm-up is **35 days and the horizon 49 days**. Greedy reference policy at steady state: staffed occupancy MED 0.94 / SURG 0.88 / ICU 0.80, mean boarding 3.99 h, 90th percentile 9.15 h, off-service 29.4%.
- **PPO-DR training range re-derived.** Checking each axis separately is not enough: a candidate range with each multiplier milder than every test value still had a corner load of 1.046, heavier than B and D. The adopted range has a worst-corner load of 0.981 / 0.854, below the mildest test scenario.
- **Warm-start pool size.** At the heavier load, a warm-start test using 10 pool states failed: mean boarding 5.4 h vs 3.6 h. A 200-run diagnostic showed this was **unrepresentative sampling, not bias**:

  | Sample | Mean time-avg boarding [95% CI] |
  |---|---|
  | Full episodes (200) | 3.85 [3.29, 4.42] |
  | Warm, 200 distinct states | 3.86 [3.30, 4.43] (p = 0.98) |
  | Warm, 10 states × 20 branches | 5.89 [5.23, 6.54] |

  Near saturation the state is very persistent, so a few states misrepresent the stationary distribution. Training pools now need ≥ 200 distinct states. The test uses one branch per distinct state and still detects the naive-restart bug (p ≈ 10⁻¹³).

## 3. Tuning

**Protocol (fixed in `neurotwin/experiments/tuning.py` before running):**
- Scenario A, validation seeds 100000–100019 only (asserted), full evaluation protocol.
- Objective: **J** = α·(priority-weighted request-hours) + β·diversions + κ·off-service placements, per measured day. This is the same objective the RL reward optimises.
- Every candidate is run on the same seeds (CRN); selection is by lowest mean J.

| Policy | Grid | Selected | J [95% CI] | Runner-up: paired ΔJ [95% CI] |
|---|---|---|---|---|
| FCFS | W_os ∈ {2, 4, 8, 12} | W_os = 4 | 276.8 [164.8, 388.9] | W_os = 8: +7.7 [−16.8, 32.1] |
| Priority | W_os ∈ {2, 4, 8, 12} | W_os = 12 | 168.7 [116.7, 220.8] | W_os = 8: +3.7 [−6.0, 13.2] |
| Load-Balancing | θ ∈ {0.05, 0.1, 0.2, 0.3} | θ = 0.10 | 164.4 [109.4, 219.3] | θ = 0.05: +4.7 [−4.6, 14.0] |
| Trunk-Reservation | r ∈ {0..5}³, W_os from Priority | r = (0, 0, 0) | 168.7 (= Priority) | r_ICU = 1: +9.3 [−0.8, 19.3] |
| MIP | κ_MIP ∈ {2..80}, Δ ∈ {1, 4}, β_b ∈ {0, 10} | κ_MIP = 20, Δ = 4, β_b = 10 | 179.8 [114.6, 244.9] | κ_MIP = 10, β_b = 0: +2.7 [−2.4, 7.9] |

Runner-up differences are not significant. The selected values are best-estimates, and the choice among top candidates matters little.

### Findings

1. **Ordering matters a lot.** FCFS scores J ≈ 277 against ≈ 165–169 for priority-ordered policies.
2. **Trunk reservation tunes to zero reserve, which theory predicts.** Reservation is optimal in *loss* systems (Miller, 1969). In a *delay* system with class weights and equal service rates, the cμ rule (serve the highest-weighted patient first, never idle) is optimal, and that is Priority. Diversions are rare in A (≈0.1 per 100 arrivals), so A behaves as a delay system. Reservation can only help where diversions are frequent (stress). The RQ4 hypothesis is kept as pre-registered. The A-tuned reserve is not retuned on stress scenarios: that would use test-scenario information.
3. **The exactly solved myopic optimiser loses to simple rules.**
   - First tuning run: κ_MIP was fixed to the reward's κ = 2, and only the breach cost was tuned. All four breach values gave identical J = 213, a strawman.
   - With κ_MIP tuned as a surrogate for the downstream cost of off-service stays, which the one-step model cannot see, it reached J = 180. That is still behind Priority and Load-Balancing.
   - This is the clearest evidence so far for RQ1's premise: the value of a decision depends on future capacity, and one-step optimisation cannot see that.
4. **Common random numbers work.** Confidence intervals of *paired* differences between candidates are 5–20× narrower than the intervals of the candidates' own means (for example ±9.6 against ±52 for Priority). Section 6.24 will quantify this formally.

## 4. Fairness checklist

- [x] Same observable information for every policy (public queries only; tested no hidden fields)
- [x] Same decision epochs and placement primitive
- [x] Same tuning data (A, validation seeds) and same objective as the RL reward
- [x] Every baseline's decisive parameter tuned, including the MIP's off-service cost
- [x] No tuning on stress scenarios or test seeds (asserted in code)

## 5. Limitations

- Grids are coarse (4–24 points, plus 216 for trunk reservation). Runner-up differences are within noise, so finer grids are unlikely to change conclusions.
- Trunk reservation was tuned in two stages (W_os fixed from Priority, then r), not jointly.
- The MIP's horizon Δ and breach cost interact with κ_MIP; the grid covers only a few combinations.
- Tuning used 20 validation seeds. The final comparison (Stage 6) uses 50 independent test seeds.
