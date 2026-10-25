# Stage 7 Report: Robustness, Variance Reduction, Policy Structure and Ablations

**Date:** 2026-10-26
**Data:** `docs/data/stage7/` (one JSON file per analysis; file names are given in each section)
**Reproduce:** each section names its command. All of them read the Stage 6 test episodes or rerun them with the same seeds.

> **Simulation results only.** The model is not calibrated to any real hospital, and nothing here says anything about real-world performance.

## 1. Robustness under stress (RESEARCH.md §6.15; `robustness.json`)
**Command:** `python -m neurotwin.experiments.robustness` (Stage 6 data, no new episodes)

Degradation is the relative change from scenario A to each stress scenario: Deg = (mean_S − mean_A) / mean_A, where positive means worse. The 95% CIs come from a bootstrap that resamples the 50 seeds jointly across scenarios and, for PPO, the 5 agents. The Robustness Index (RI) is the mean degradation over B–E and four metrics (J, mean boarding, p90 boarding, off-service).

**Disclosed deviation.** Diversion is reported as an *absolute* change per 100 arrivals. Its scenario-A value is about 0.1–0.6, so a relative change would divide by almost zero.

| Policy | RI (%) | 95% CI | PPO − policy (RI points) | 95% CI |
|---|---|---|---|---|
| PPO | 311 | [235, 424] | — | — |
| FCFS | 295 | [228, 391] | +15.8 | [−18.5, 63.8] |
| Priority | 390 | [311, 497] | **−78.8** | [−104.2, −49.3] |
| Load-Balancing | 316 | [240, 429] | −5.4 | [−27.2, 17.0] |
| Trunk-Reservation | 390 | [311, 497] | **−78.8** | [−104.2, −49.3] |
| MIP | 300 | [226, 412] | +11.4 | [−11.8, 35.2] |

**Findings**
- **PPO degrades less than Priority** (CI excludes 0). PPO's degradation is statistically indistinguishable from Load-Balancing, MIP and FCFS.
- **A lower RI does not mean a better policy.** FCFS has a low *relative* degradation only because its scenario-A cost is already high (J = 338 vs PPO's 208). In absolute terms it is the worst policy in every scenario (Stage 6). The RI should be read together with the absolute levels.
- **Trunk-Reservation coincides exactly with Priority.** Its tuned reserve is r = 0, as Stage 4 §90 explains: with rare diversions in A, the system behaves as a delay system, where the cμ rule is optimal. Retuning on stress scenarios would use test information, so it was not done.
- **Diversion increase in E (per 100 arrivals):**

  | Policy | Increase | 95% CI |
  |---|---|---|
  | PPO | +14.0 | [11.5, 16.8] |
  | Load-Balancing | +11.8 | [10.0, 13.7] |
  | Priority | +14.4 | [12.4, 16.5] |
  | FCFS | +22.1 | [19.6, 24.5] |

  PPO's diversions are not better controlled than Load-Balancing's under combined stress. This motivates the constrained-PPO study (§6).

## 2. Measured variance reduction from Common Random Numbers (§6.24; `crn_check.json`)
**Command:** `python -m neurotwin.experiments.crn_check`

Setup:
- For each baseline, the scenario-A difference PPO − baseline is estimated twice:
  - **paired:** both policies on the same 50 test seeds;
  - **independent:** the baseline on a disjoint seed block, used only here.
- PPO is agent-averaged per seed.

| Metric | Baseline | Paired CI ± | Independent CI ± | Width ratio | Per-seed correlation |
|---|---|---|---|---|---|
| J/day | FCFS | 39.2 | 126.3 | 3.2× | 0.971 |
| J/day | Priority | 7.1 | 84.1 | 11.8× | 0.993 |
| J/day | Load-Balancing | 6.5 | 81.0 | **12.5×** | 0.995 |
| J/day | MIP | 22.4 | 91.7 | 4.1× | 0.986 |
| Mean boarding (h) | Load-Balancing | 0.12 | 1.27 | 10.8× | 0.994 |

(Trunk-Reservation equals Priority, as noted in §1.)

**Finding.** CRN makes the CIs up to 12.5× narrower, which is equivalent to roughly 150× more independent seeds for the same precision. Without CRN, none of the PPO-vs-Load-Balancing or PPO-vs-Priority differences in Stage 6 would be resolvable at n = 50. The gain is smallest for FCFS, whose dynamics differ most from PPO's, so its per-seed outcomes are least correlated.

## 3. Does PPO learn bed reservation? (RQ4, §6.20; `structure.json`)
**Command:** `python -m neurotwin.experiments.structure`

**Reservation states** are states where:
- a general admission could be placed in its primary ward; and
- no urgent (priority 1–2) patient is waiting.

HOLD then means keeping a bed free for a possible urgent arrival. The analysis bins P(HOLD) by the number of free staffed beds b, pooling all 5 agents over the first 20 test seeds.

| Scenario | States | Spearman ρ(b, P(HOLD)) | Isotonic R² vs flat | Share of HOLD decisions |
|---|---|---|---|---|
| A (train) | 30,267 | **−0.51** (p ≈ 0) | **0.98** | 4.7% |
| E (stress) | 24,422 | +0.21 | 0.01 | 35.5% |

**P(HOLD) by free beds in A:**

| b | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥8 |
|---|---|---|---|---|---|---|---|---|
| P(HOLD) | 0.088 | 0.107 | 0.058 | 0.037 | 0.032 | 0.023 | 0.016 | 0.005 |

**Findings**
- **In the training scenario, PPO has learned a soft trunk-reservation rule.** The probability of holding a bed falls almost monotonically as free capacity grows (isotonic R² = 0.98). The best *hard* threshold baseline tunes to r = 0, so this soft, state-dependent reservation is learned behaviour that the threshold family does not contain. It is also consistent with PPO beating Priority (§1, Stage 6).
- **The structure breaks down out of distribution.** In E, P(HOLD) is high (≈0.3) at every b and shows no decreasing structure. The policy holds far more often in states it never saw in training. This is the mechanistic explanation for its diversion increase in §1, and it is reported as a negative result.

## 4. Ablations (§6.16; `ablation_*.json`)
**Commands:** `python -m neurotwin.experiments.ablations train`, then `python -m neurotwin.experiments.ablations evaluate`. The second produces `ablation_episodes.json` and `ablation_comparisons.json`.

**Design**
- Each variant retrains PPO with one change and the same hyperparameters, seeds and scenario (A).
- Removed observation features are set to a constant 0, which keeps the network shape fixed.
- Evaluation is on the 50 test seeds in A and E.
- Each variant is compared with the full model (final_A), so Δ = ablated − full, and positive means removing the component hurts.
- Holm correction is applied over all 80 comparisons (8 variants × 2 scenarios × 5 metrics).

**Disclosed deviations and limits**
- **K = 3 agents per variant** instead of 5, for compute reasons.
- **The full model's agent variance is not resampled:** it enters agent-averaged per seed. Its CIs are therefore somewhat optimistic.
- **The full model includes agent 0,** whose rare collapses in stressed episodes (Stage 6) raise its scenario-E means. The ablations use agents 0–2 with the same seeds, but their training differs, so this cannot be matched exactly.

| Variant (what is removed) | ΔJ in A | 95% CI | ΔJ in E | 95% CI |
|---|---|---|---|---|
| B: demand history (3 features) | +0.4 | [−4.9, 5.8] | +18.4 | [−19.2, 57.0] |
| C: time of day / week (4) | +4.0 | [−0.8, 9.0] | +15.0 | [−24.2, 52.8] |
| D: staffing (2) | −0.9 | [−6.7, 5.7] | −49.2 | [−117.1, 17.3] |
| E: priority aggregates (3) | −2.0 | [−6.0, 2.3] | +9.4 | [−39.9, 58.9] |
| F1: diversion penalty (β = 0) | +0.5 | [−5.2, 5.9] | −27.1 | [−161.4, 69.5] |
| F2: off-service penalty (κ = 0) | +13.4 | [4.1, 24.5] | −65.3 | [−157.0, 5.8] |
| F3: uniform priority weights in reward | **+23.5*** | [13.2, 34.4] | −80.2 | [−181.1, 34.8] |
| G: ICU-flow state (2) | +15.6 | [1.5, 37.3] | +32.8 | [−34.4, 89.7] |

\* significant after Holm. The full model's J is 208.1 in A and 1453.5 in E. Every metric is in `ablation_comparisons.json`.

**Findings**
1. **The priority-weighted reward is the component that matters most.** Training with uniform priority weights (F3) is the only variant whose J in A is significantly worse after Holm: +23.5, or +11%, with mean boarding also significantly worse. The reward structure therefore carries more of the policy's performance than any single observation group.

   In E, F3 has a significantly *lower* p90 boarding time: −13.7 h, 95% CI [−19.0, −8.2]. Weighting all patients equally shortens the longest waits, at the cost of priority-weighted waiting. This is the expected trade-off, and shows that J encodes a value judgement.
2. **The off-service penalty does what it is designed to do.** Without κ (F2), off-service placements rise significantly in A (+3.2 per 100). The nominal J increase (+13.4) does not survive Holm. This is a sanity check that the reward terms are wired correctly.
3. **No single observation group is necessary in A.** Removing demand history, temporal, staffing or priority-aggregate features changes J in A by less than 5 per day, and no change is significant. The features are partly redundant: the remaining inputs carry much of the same information. ICU-flow state (G) has the largest nominal effect (+15.6), but it does not survive Holm.
4. **The diversion penalty learned in A has no measurable effect, even under stress.** Training with β = 0 (F1) leaves diversions in A unchanged. In E, F1 has nominally *fewer* diversions (−1.2 per 100), which is not significant after Holm.

   Diversions are rare in A (about 0.3 per 100 arrivals), so the β term gives almost no training signal. A fixed penalty learned in the training scenario therefore cannot control diversions under stress. This is direct evidence for the constrained-PPO redesign in §6.
5. **No feature group helps under stress.** In E, no ΔJ is significant. Several observation ablations are associated with significantly fewer off-service placements (C, E, G) and diversions (B, C).

   One untested explanation is that fewer inputs mean less extrapolation outside the training distribution, which would be consistent with the structure breakdown in §3. Agent-level differences (see the limits above) could also contribute. We do not claim a causal effect.

## 5. Domain-randomised PPO (PPO-DR, §6.23; `ppo_dr_*.json`)
**Status: pending.** The training range has been checked to be strictly milder than every test scenario (`ppo_dr check`; tested in `tests/test_stage7.py`).

## 6. λ-conditioned constrained PPO (v2.6, §6.19; `constrained_*.json`)
**Status: pending.** The design was pre-registered before any run (RESEARCH.md v2.6). A single policy is trained across λ ∈ [0, 100]. At deployment, online dual ascent moves λ toward a budget of 3 diversions per 100 arrivals. The question is whether this policy keeps diversions within budget under stress, unlike fixed-β PPO (§1), and how much waiting cost that costs.
