# Stage 1 Design Review and Technical-Depth Plan

**Date:** 2026-10-25
**Scope:** Review of the locked experimental design in `RESEARCH.md` §6 before any code is written, plus the plan for the parts of NeuroTwin that are implemented in-house rather than taken from libraries.
**Status:** **Approved by the project owner on 2026-10-25 and applied in `RESEARCH.md` v2.0.** The full rewrite also fixed further defects found while applying these (A10–A16: redundant and non-decision actions, missing deterioration process, validation-set leakage, a duplicate state feature, the ambiguous "overflow" term, CRN implementation requirements, and conditional test selection). It also corrected or removed unverifiable citations. See the `RESEARCH.md` changelog. Some details below were superseded by v2 (for example, the MIP is now a transportation problem with staffing folded into unit capacities); **`RESEARCH.md` v2 is authoritative.**

---

## Part A — Defects found in the locked design

Each item states the problem, why it matters, and a proposed fix. Items are ordered by how much they would affect the validity of results.

### A1. Difference-based reward does not penalise sustained bad states (`RESEARCH.md` §6.8)

**Problem.** The reward is `R_t = C(s_t) − C(s_{t+1})`. The cost `C` mixes two kinds of term:

- *Counter terms* (cumulative overflow count, cumulative throughput). For these, `C(s_t) − C(s_{t+1})` is the per-step increment, which is the right signal.
- *Level terms* (average waiting time, average ED wait, utilisation variance). For these, the difference only rewards *changes*. A queue that stays long for 10 hours costs the agent nothing during those hours.

Summing the reward over an episode telescopes. With γ = 1 the return is `C(s_0) − C(s_T)`, so only the final state's level terms matter. With γ < 1 the return is

```
C(s_0) − (1−γ) Σ_{t=1}^{T−1} γ^{t−1} C(s_t) − γ^{T−1} C(s_T)
```

so the time-integrated cost survives only with weight `(1−γ)` (0.01 at γ = 0.99), while the terminal state dominates.

**Why it matters.** The primary metric is average patient waiting time over the measurement period, which is an integral over time. The reward as specified is not aligned with that metric, and the "Why difference-based" argument in §6.8 (scale invariance, gradient stability) does not address this. The structure is close to potential-based shaping (Ng, Harada & Russell, 1999), which by construction leaves the optimal policy determined by the *base* reward. Here the base reward is effectively zero for the level terms.

**Proposed fix.** Use a direct per-step cost accrued over the interval between decisions:

```
R_t = −[ α · ∫_{τ_t}^{τ_{t+1}} Q_wait(τ) dτ        # patient-hours of waiting accrued
       + β · (new overflow events in (τ_t, τ_{t+1}])
       + δ · (transfers made at t)
       − ε · (patients admitted/discharged in (τ_t, τ_{t+1}])
       + ζ · ∫ utilisation_imbalance(τ) dτ ]
```

`∫ Q_wait dτ` is the total patient-waiting time accrued in the interval. Summed over an episode it equals total waiting time, which is exactly the primary metric. This follows from the same accounting as Little's law. The episode return therefore lines up with what is evaluated.

Decisions are event-driven, so intervals `Δt = τ_{t+1} − τ_t` vary. This makes the problem a semi-Markov decision process. Discounting should use elapsed time: `γ^{Δt / Δt_ref}`. PPO's GAE then uses a per-step discount, which the in-house implementation (B2) supports directly. Off-the-shelf stable-baselines3 does not without modification.

### A2. MIP waiting-time constraint is dimensionally wrong (`RESEARCH.md` §6.7, constraint 4)

**Problem.** `w_i ≥ arrival_time_i − (current_time × y_i)`.
- If `y_i = 1`: `w_i ≥ arrival_time_i − current_time`, which is ≤ 0, so `w_i = 0`. Fine.
- If `y_i = 0`: `w_i ≥ arrival_time_i`, an absolute timestamp. A patient who arrived on day 13 is penalised 13× more than one who arrived on day 1, regardless of how long either has waited.

**Proposed fix.** In a myopic problem, the elapsed wait `e_i = current_time − arrival_time_i` is data, not a decision. Holding cost is linear in `y_i` and needs no `w_i` variable:

```
holding cost_i = (1 − y_i) · h_i,   h_i = priority_weight_i · (e_i + Δ_review)
```

`Δ_review` is the expected time until the next decision point (≤ 4 h by the periodic trigger).

### A3. MIP overflow definition is not linear (`RESEARCH.md` §6.7, constraint 5)

**Problem.** "`o_i = 1` if patient i waits > max_wait_threshold" is a logical condition, not a linear constraint.

**Proposed fix.** Because `e_i` is data, the condition is known before solving. For patients with `e_i + Δ_review > max_wait_threshold`, add `o_i ≥ 1 − y_i`; otherwise fix `o_i = 0`. No big-M constant is needed.

### A4. "RL is also myopic" is incorrect (`RESEARCH.md` §6.7, justification 1)

**Problem.** PPO optimises discounted return with γ up to 0.999 (the tuning range in "Remaining Experimental Hyperparameters"). It is explicitly non-myopic. An examiner who knows RL will reject this justification.

**Proposed fix.** Keep the myopic MIP, but justify it honestly. It is the strongest *one-step* optimiser, so the comparison PPO vs MIP directly tests the research-relevant question: **does learned look-ahead beat optimal myopic allocation?** That is a stronger framing than claiming parity.

Optional stronger baseline (Stage 4 stretch): a rolling-horizon MIP using the *configured expected* arrival rate `λ(t)`. This is not future information leakage, because the RL agent also observes arrival-rate features. Decide in Stage 4 based on time.

### A5. MIP structure: the core problem is an assignment problem (informational, affects solver choice)

Without the staffing constraint (constraint 6), the MIP is a bipartite assignment problem. Its constraint matrix is totally unimodular, so the LP relaxation has integral optima and it can be solved exactly with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). The staffing constraint is a knapsack-type side constraint that breaks total unimodularity, so the full problem needs a MILP solver.

**Proposed solver:** `scipy.optimize.milp` (HiGHS backend, SciPy ≥ 1.9). SciPy is already a required dependency of scikit-learn, which is approved, so **no new dependency** is added. The Hungarian solution of the relaxed problem serves as a test oracle: when staffing is non-binding, both solvers must return the same objective value.

Constraint 6 should also count staff already committed to occupied beds, not just the new assignments.

### A6. Priority scale contradicts itself

`RESEARCH.md` §6.5 says "Priority level (1-5, **5 most urgent**)". §6.4 ("Priority 1-2 patients waiting" = high-priority) and §6.9 ("Priority 1-2: 15%") treat **1 as most urgent**. Triage systems such as ESI use 1 = most urgent.

**Proposed fix:** 1 = most urgent everywhere; correct §6.5.

### A7. PPO is never evaluated on the Normal scenario, but the robustness metric needs it

The experiment matrix (§6.16) marks PPO × Normal(A) as "excluded (TRAIN)". The degradation metric (§6.13) needs `Metric(p, normal)` for every policy, including PPO.

**Proposed fix:** Evaluate PPO on scenario A using **held-out test seeds** that are disjoint from training and validation seeds. This is an in-distribution test and is not training on the test scenario. Mark the cell "included (held-out seeds)".

### A8. Degradation formula takes an absolute value, then claims a sign (§6.13)

`|M(stress) − M(normal)| / M(normal)` loses direction, but the text says the sign indicates improvement or deterioration.

**Proposed fix:**

```
Degradation_m(p) = s_m · (M(p, stress) − M(p, normal)) / M(p, normal) × 100%
```

with `s_m = +1` for lower-is-better and `−1` for higher-is-better metrics. Then positive always means worse. Report a paired bootstrap CI on the degradation itself (same seeds across scenarios under CRN).

### A9. RL training variance is not part of the seed protocol (§6.11, §6.16)

The seed protocol covers *environment* seeds. PPO results also vary substantially across *training* seeds (network initialisation, minibatch order); see Henderson et al., "Deep Reinforcement Learning That Matters", AAAI 2018. One trained agent is one sample.

**Proposed fix:**
- Train **K ≥ 5 independent PPO agents** (different training seeds), each drawing a fresh environment seed per episode from a training-only seed range.
- Report test results per agent and aggregated. The paired bootstrap resamples over (agent, test seed) pairs, or uses a hierarchical bootstrap.

---

## Part B — In-house technical core (applied to ARCHITECTURE.md and ROADMAP.md)

### B1. Simulator validation against queueing theory (Stage 2)

Before adding hospital complexity, the DES engine is checked against closed-form results using a reduced configuration.

| Check | Reduced configuration | Analytical result |
|---|---|---|
| M/M/c delay system | Homogeneous Poisson arrivals λ, exponential LOS with mean 1/μ, c identical beds, infinite FIFO queue, no staff limits, no overflow | Erlang C: `P(wait)`, `W_q = C(c, λ/μ) / (cμ − λ)` |
| M/M/c/c loss system | As above, but arrivals finding all beds busy are diverted (overflow) | Erlang B blocking probability = simulated overflow fraction |
| Little's law | Full hospital model, any policy | `L = λ_eff · W` for each queue, over the measurement window |
| Non-homogeneous arrivals | Arrival generator alone, rate `λ(t)` | Counts per hour-of-day bin match `∫ λ(t) dt` (chi-square goodness of fit) |

**Acceptance:** for each check, the analytical value lies inside the 95% CI of the simulated estimate over ≥ 30 seeds, at several loads (ρ = 0.5, 0.8, 0.95). The checks become permanent regression tests, so later changes to the engine cannot silently break it.

### B2. PPO implemented in-house (Stage 5)

PPO is written in PyTorch inside `rl/ppo/`. stable-baselines3 is retained **only as a reference implementation** for validation, not as the training path.

Components implemented and documented:
- Actor-critic MLP (separate or shared trunk, configurable), orthogonal initialisation.
- **Action masking** at the logit level (masked logits set to −∞ before softmax), matching the masking rules in `RESEARCH.md` §6.2.
- Rollout buffer with per-step discount (supports A1's semi-Markov discounting).
- Generalised Advantage Estimation (Schulman et al., 2016).
- Clipped surrogate objective, clipped value loss, entropy bonus (Schulman et al., 2017).
- Advantage normalisation, gradient-norm clipping, learning-rate annealing.
- Observation normalisation with running statistics (frozen at evaluation).

Implementation details follow Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" (ICLR Blog Track, 2022). Each detail used is cited in a code comment.

**Validation (must pass before hospital training results are reported):**
1. **CartPole-v1:** mean return over 5 training seeds reaches the solved threshold within a fixed budget, comparable to stable-baselines3 PPO under the same hyperparameters (both curves reported).
2. **Hospital environment:** in-house PPO and stable-baselines3 PPO (masked variant from sb3-contrib *only if* approved; otherwise unmasked on a mask-free reduced configuration) trained with matched hyperparameters reach overlapping learning-curve CIs.
3. Unit tests: GAE against a hand-computed example; the clipped loss equals the unclipped loss when the ratio is inside [1−ε, 1+ε]; masked actions have zero probability.

### B3. Constrained PPO extension (Stage 7, optional)

Instead of hand-tuning the overflow weight β, treat overflow as a constraint. This is a constrained MDP:

```
maximise  E[return without overflow term]
subject to  E[overflow rate per day] ≤ c
```

It is solved with a Lagrangian multiplier updated by dual ascent (Tessler, Mankowitz & Mannor, "Reward Constrained Policy Optimization", ICLR 2019; Ray, Achiam & Amodei, "Benchmarking Safe Exploration in Deep RL", 2019).

**Experiment:** PPO with fixed β vs PPO-Lagrangian at the same overflow budget `c`. Measure constraint satisfaction under the stress scenarios, where a fixed β tuned on Normal may no longer be enough. This links directly to the robustness research question and reduces one arbitrary reward weight.

This is an additional policy variant. It is evaluated only if Stages 5–7 core results are complete.

---

## Decisions requested

1. **Approve or reject A1–A9 individually.** Approved items will be written into `RESEARCH.md` §6 with a changelog entry.
2. **Approve B3** (constrained PPO) as an optional Stage 7 extension.
3. **sb3-contrib** (for a masked PPO reference in B2.2): approve as a dev/test-only dependency, or skip it and validate on a mask-free reduced configuration.

## References

- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *AAAI*.
- Tessler, C., Mankowitz, D. J., & Mannor, S. (2019). Reward constrained policy optimization. *ICLR*.
- Ray, A., Achiam, J., & Amodei, D. (2019). Benchmarking safe exploration in deep reinforcement learning. OpenAI.
- Huang, S., Dossa, R. F. J., Raffin, A., Kanervisto, A., & Wang, W. (2022). The 37 implementation details of proximal policy optimization. *ICLR Blog Track*.
