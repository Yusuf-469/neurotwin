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
**Commands:** `python -m neurotwin.experiments.ppo_dr train --workers 5`, then `python -m neurotwin.experiments.ppo_dr evaluate`. The output is `ppo_dr_episodes.json` and `ppo_dr_comparisons.json`.

**Design**
- **Training:** each training episode draws its arrival multiplier from [0.95, 1.04] and its staff and capacity multipliers from [0.97, 1.0]. Everything else is identical to the final PPO: K = 5, the same agent and episode seeds, and warm starts branched into the sampled conditions.
- **Leakage guard:** the harshest training corner (ρ_general = 0.981) is strictly milder than the mildest test scenario (1.020). This is asserted before training and tested.
- **Evaluation:** test seeds in A–E, paired against PPO, with Holm correction over 25 comparisons.

| Scenario | ΔJ (PPO-DR − PPO) | 95% CI | Δ off-service per 100 | 95% CI |
|---|---|---|---|---|
| A | +1.1 | [−2.3, 4.9] | −0.3 | [−0.8, 0.1] |
| B | −45.0 | [−128.9, 17.8] | −0.6 | [−1.3, 0.1] |
| C | −30.2 | [−90.7, 20.3] | −1.1 | [−1.8, −0.4] |
| D | −38.1 | [−89.8, 3.9] | −0.6 | [−1.1, −0.1] |
| E | −60.8 | [−147.8, 13.2] | **−1.8*** | [−2.8, −0.9] |

\* significant after Holm.

**Finding: a consistent but unconfirmed benefit at no cost.**
- PPO-DR is no worse in the training scenario. Under every stress scenario its J is nominally lower, by 3–5%, but no ΔJ survives correction.
- Its only significant effect is fewer off-service placements under combined stress.
- Randomising over conditions milder than the tests therefore does not by itself close the generalisation gap.
- A wider training range would need to approach the test conditions, which risks leakage. This is why the range was fixed in advance, and it is the obvious next experiment.

## 6. λ-conditioned constrained PPO (v2.6, §6.19; `constrained_*.json`)
**Commands:** `python -m neurotwin.experiments.constrained train --workers 5`, then `python -m neurotwin.experiments.constrained evaluate`. The output is `constrained_episodes.json` and `constrained_comparisons.json`.

**Design** (pre-registered in RESEARCH.md v2.6 before any run)
- **Training:** one policy is trained across λ ∈ [0, 100], with λ/100 as an input. Training is in scenario A with the same seeds as the final PPO, K = 5.
- **Deployment:** online dual ascent starts at λ0 = 20. Every 24 h it moves λ toward a budget of 3 diversions per 100 arrivals, with step size 5.
- **Evaluation:** test seeds in A–E, paired against fixed-β PPO, with Holm correction over 25 comparisons.

| Scenario | Mean λ (measured) | Δ diversions per 100 (λ-PPO − PPO) | 95% CI | Episodes within budget: PPO / λ-PPO | ΔJ | 95% CI |
|---|---|---|---|---|---|---|
| A | 0.5 | −0.12 | [−0.32, 0.01] | 96.8% / 97.2% | +5.6 | [0.4, 11.3] |
| B | 29.1 | +0.04 | [−1.23, 1.49] | 58.8% / 55.6% | +27.2 | [−54.1, 95.6] |
| C | 56.4 | +0.42 | [−0.81, 2.33] | 24.0% / 20.4% | +77.5 | [23.9, 131.5] |
| D | 28.6 | −0.80 | [−1.90, 0.07] | 48.8% / 50.0% | +2.5 | [−40.2, 35.6] |
| E | 77.7 | +1.80 | [−1.64, 7.99] | 7.2% / 6.4% | +75.2 | [−7.0, 168.8] |

No comparison is significant after Holm. The ΔJ values in A and C exclude 0 only nominally.

**Finding: a negative result. The controller behaves as designed, but the policy does not respond.**
1. **The dual ascent works as intended.** λ falls toward 0 where the budget is met (A) and rises with stress (B/D ≈ 29, C ≈ 56, E ≈ 78).
2. **Diversions do not fall.** The share of episodes within budget is unchanged in every scenario. ΔJ is nominally worse in A and C, which is what a policy pays when it is pushed by a large λ that it cannot act on effectively.
3. **Why it fails:**
   - **Training signal.** The policy was trained only in A, where diversions are about 0.3 per 100 arrivals. Whatever λ it was given, its training episodes contained almost no diversions, so it learned almost nothing about how its actions affect them. The λ input therefore barely changes its behaviour. This is the same mechanism as ablation F1 (§4.4): β = 0 vs β = 20 made no difference.
   - **Infeasible budget.** The budget is unreachable under heavy stress for *every* policy: in Stage 6, the lowest diversion rates were 7.9 (Load-Balancing, C) and 12.0 (Load-Balancing, E) per 100. When ED capacity is exhausted, diversions are driven by capacity rather than by the allocation decision.
4. **What would be needed:** training episodes in which diversions actually occur, such as constrained training on randomised stress, which needs a careful leakage design. A budget should also be defined relative to what is achievable in each scenario. Both are left as future work and stated as such.

This experiment is reported as it came out. It was designed to fix the flaw of a frozen multiplier, and it did: the multiplier now adapts online. It also showed that a constraint method cannot control an outcome the policy never learned to influence. This is a useful negative result for constrained RL in rare-event settings.

## 7. Safety shield against the collapse mode (v2.7, §6.25; `shield_comparisons.json`)
**Command:** `python -m neurotwin.experiments.shield --workers 10`. The output is `shield_episodes.json` and `shield_comparisons.json`.

**Design** (pre-registered in RESEARCH.md v2.7, commit f574cf7, before any run)
- **Motivation, disclosed:** the idea came from the Stage 6 collapse finding, so it is post-hoc in motivation. The Stage 6 results are unchanged.
- **Rule:** while the ED is full (exactly the state in which arrivals are diverted) and a placement is feasible, HOLD is removed from the agent's options. The agent takes its own best remaining action.
- **No tunable parameter, and no retraining:** the five final agents are used as they are. Outside ED overflow the policy is identical.
- **Evaluation:** the same 50 test seeds in A–E, compared with the Stage 6 rows for unshielded PPO and Load-Balancing (deterministic simulator, same seeds). Holm correction over 10 cost comparisons.

**Primary outcome: collapses** (measured diversion above 50 per 100 arrivals)

| Policy | Collapsed episodes |
|---|---|
| PPO, unshielded (Stage 6) | 5 of 1,250 |
| **PPO with shield** | **0 of 1,250** |
| Load-Balancing (Stage 6) | 0 of 1,250 |

The worst episode's diversion rate fell from 99 to 34 per 100 arrivals (scenario E). Agent 0's worst cost per day fell as follows:

| Scenario | Unshielded | Shielded |
|---|---|---|
| B | 8,331 | 1,764 |
| D | 3,646 | 1,764 |
| E | 3,908 | 2,145 |

The shield intervened on average 1.9 times per episode in A, and 51–176 times in B–E.

**Secondary outcome: cost J per day** (shielded − comparator; positive = shield worse)

| Scenario | vs unshielded PPO | 95% CI | vs Load-Balancing | 95% CI |
|---|---|---|---|---|
| A | +1.0 (+0.5%) | [−4.9, 7.5] | +4.1 (+2.0%) | [−3.9, 13.5] |
| B | +12.4 (+1.5%) | [−59.8, 60.7] | +15.7 (+1.9%) | [−12.8, 43.6] |
| C | **+51.7 (+4.4%)*** | [24.2, 81.7] | +11.8 (+1.0%) | [−29.5, 56.1] |
| D | −1.4 (−0.1%) | [−45.7, 35.1] | +29.4 (+3.3%) | [−3.1, 62.7] |
| E | +16.5 (+1.1%) | [−30.3, 65.4] | +37.9 (+2.6%) | [−11.9, 90.6] |

\* significant after Holm.

**Descriptive mechanism** (means over 250 episodes; not significance-tested):
- **Fewer turned-away ambulances:** diversions fell in every scenario, e.g. 14.3 → 12.9 per 100 in E and 5.3 → 4.2 in D.
- **Longer tail waits:** 90th-percentile boarding rose in B, C and E, e.g. 48.9 → 52.3 h in C.

Forcing a placement while the ED is full fills a bed that a more urgent later patient would otherwise have taken.

**Findings**
1. **The parameter-free shield eliminates the collapse mode** (5 → 0 of 1,250) without retraining, and removes the extreme worst cases.
2. **It is not free.** In the staff-shortage scenario the average cost rises by 4.4% (significant). Elsewhere the change is not significant. The shielded agent remains statistically indistinguishable from Load-Balancing in every scenario.
3. **Interpretation:** a learned policy plus a simple rule-based guard gives the robustness of the rules with the agent's behaviour everywhere else. This is the practical recommendation from this study: deploy learned policies only behind guards that rule out known failure states. Here, the trade is slightly worse average cost under one kind of stress in exchange for the removal of a catastrophic tail.

## 8. Can a learned policy beat Load-Balancing? Imitation-initialised PPO and ensembles (v2.8, §6.26)
**Commands:** `python -m neurotwin.experiments.bc_ppo clone`, then `finetune`, `validate` and `test`.
The output is `bc_ppo_validation.json`, `bc_ppo_test.json` and `bc_ppo_test_episodes.json`.

**Design** (pre-registered in RESEARCH.md v2.8, commit 3c07d4d, before any run)
- **Behaviour cloning (BC):** Load-Balancing's decisions are cloned into the PPO actor network. 30 training-seed episodes per agent give about 65,000 decisions each; 3.3% of the rule's decisions cannot be expressed as one of the 7 actions and are skipped.
- **Fine-tuning (BC+PPO):** PPO starts from the cloned network, with learning rate 1e-4, the observation normaliser frozen, 30 critic-only warm-up updates and 3M steps. K = 5.
- **Ensembles:** soft-vote ensembles average the masked action probabilities of five agents.
- **Selection:** on the 20 validation seeds, scenario A, by lowest mean cost J.
- **One-time final test:** on a fresh seed block (400,000–400,049) never used before, scenarios A–E. The code refuses to run it twice.

**Cloning accuracy:** 96.0–96.4% on held-out demonstrations.

**Validation** (scenario A, mean J per day; selection among the last three candidates)

| Candidate | Validation J |
|---|---|
| BC only | 260.4 |
| Load-Balancing | 164.4 |
| PPO (final_A) | 151.2 |
| ENS(PPO) | 148.1 |
| BC+PPO (mean of 5) | 144.5 |
| **ENS(BC+PPO), selected** | **144.1** |

The selection was committed (14d3a99) before the fresh test.

**Fresh test:** selected candidate − Load-Balancing, J per day (negative = learned policy better). Holm correction over 5 scenarios.

| Scenario | ENS(BC+PPO) | Load-Balancing | Difference | 95% CI | Holm p |
|---|---|---|---|---|---|
| A | 195.3 | 213.4 | **−18.1 (−8.5%)** | [−24.1, −12.9] | **0.0005** |
| B | 839.0 | 853.6 | −14.6 (−1.7%) | [−35.1, 6.8] | 0.53 |
| C | 1223.4 | 1217.8 | +5.6 (+0.5%) | [−26.3, 40.5] | 1.0 |
| D | 877.8 | 876.2 | +1.6 (+0.2%) | [−22.9, 27.1] | 1.0 |
| E | 1493.3 | 1442.2 | +51.2 (+3.5%) | [6.1, 96.9] | 0.10 |

**Secondary: compared with the original PPO** (final_A, same fresh seeds)
- Better in A: −11.6, significant.
- Worse in E: +54.3, significant.
- B–D: no significant difference.

**Collapses:** 0 of 250 for the ensemble, 0 of 250 for Load-Balancing, and 1 of 1,250 for the original PPO agents.

**Findings**
1. **A learned policy now beats the strongest tuned rule in the training scenario:** −8.5% cost in A, significant after Holm, on seeds never used before. The original PPO only tied it.
2. **Under unseen stress it ties Load-Balancing in B, C and D.** In E it is nominally 3.5% worse, which is not significant after correction. Compared with the original PPO it gives up some robustness under combined stress (significantly worse in E) in exchange for the gain in A.
3. **Pure imitation fails.** BC alone agrees with the rule on 96% of decisions but costs 58% more than the rule on validation, because small errors compound into states the rule never visited. This is the known distribution-shift problem of behaviour cloning. The RL fine-tuning is what turns the clone into a policy better than its teacher.
4. **The claim is limited to what was tested:** a significant win in normal conditions, and no significant loss under stress. It does not say that RL beats the rule everywhere.

### 8.1 Full comparison of the final agent (v2.9 addendum)
**Command:** `python -m neurotwin.experiments.final_eval`. The output is in `docs/data/final/`, in the same formats as Stage 6.

The remaining baselines were run on the same fresh block, and the Stage 6 analysis was repeated unchanged. Nothing was selected or tuned: the agent is the frozen §6.26 selection. 63 of the 100 primary comparisons are significant after Holm.

Cost J relative to each baseline (final agent − baseline, %; descriptive; negative = agent cheaper):

| Baseline | A | B | C | D | E |
|---|---|---|---|---|---|
| Load-Balancing | −8.5 | −1.7 | +0.5 | +0.2 | +3.5 |
| Priority (= Trunk-Reservation) | −14.4 | −8.5 | −0.9 | −10.1 | +1.9 |
| MIP | −17.4 | −14.7 | −15.0 | −16.0 | −18.1 |
| FCFS | −45.3 | −38.0 | −33.6 | −42.0 | −33.1 |

The 95% CI excludes 0 in every cell except:
- B, C and D against Load-Balancing;
- C and E against Priority.

The E difference against Load-Balancing is not significant in the pre-registered Holm test (§8).

The dashboard's Results page now shows this final agent by default in chapters 1–3, with the development version (v1, Stage 6) available for comparison.

## 9. Crisis-trained agent v3 (v2.10, §6.27)
**Commands:** `python -m neurotwin.experiments.v3 finetune`, then `validate` and `test`. The output is in `docs/data/v3/`.

**Design** (pre-registered, commit 555a19f, before any run)
- **Training:** PPO fine-tuning continued from each v2 (BC+PPO) agent, with per-episode domain randomisation covering every stress scenario: arrivals ×0.95–1.15, staff ×0.85–1.00 and beds ×0.85–1.00. K = 5, soft-vote ensemble.
- **Disclosed:** situations 2–5 are therefore in-distribution for v3. Its results answer "can a crisis-trained agent beat the rules in crises?", not the generalisation question that v1 and v2 address.
- **Selection:** v2 vs v3 on the 20 validation seeds in all five situations. The score is the mean relative J against Load-Balancing: v2 −1.2%, v3 −2.0%, so **v3 was selected** (committed 24ce77c before the test).
- **One-time test:** a second fresh block (500,000–500,049), all baselines.

**Fresh test: v3 − Load-Balancing, cost J per day** (Holm over 5)

| Situation | Difference | 95% CI | Holm p |
|---|---|---|---|
| 1 Normal day | **−11.2 (−5.1%)** | [−18.6, −3.5] | **0.024** |
| 2 Demand surge | −13.0 (−1.6%) | [−35.2, 11.2] | 0.52 |
| 3 Staff shortage | **−65.6 (−5.3%)** | [−97.1, −34.9] | **0.0005** |
| 4 Bed loss | −9.8 (−1.1%) | [−33.4, 14.6] | 0.52 |
| 5 Everything at once | +23.1 (+1.6%) | [−6.3, 53.1] | 0.36 |

**Against the other baselines** (relative J):
- **Priority:** −15.5, −9.5, −5.6, −9.6 and −1.6% in situations 1–5.
- **MIP:** −18 to −22% in every situation.
- **FCFS:** −35 to −48% in every situation.

**Collapses:** 0 in 1,500 test episodes, for every policy.

**Findings**
1. **v3 beats the strongest rule significantly in two situations**, normal day and staff shortage, **and is not significantly worse in any**. Its point estimate is lower than Load-Balancing's in four of five situations.
2. **Crisis training closed most of v2's crisis gap.** Staff shortage moved from a tie to a significant win. Under combined stress the gap fell from +3.5% (v2, first fresh block) to +1.6% (v3, second fresh block). The two blocks differ, so this comparison is indicative, not a paired test.
3. **These crisis results are in-distribution for v3** and are reported as such.
