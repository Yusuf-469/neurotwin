# Stage 6 Report: Primary Evaluation on Test Seeds

**Date:** 2026-10-26
**Protocol:** committed in b762841 *before* the test run; amended in 9a1726f (§2) before recomputation.
**Data:** `docs/data/stage6/episodes.csv` (2,500 episodes), `comparisons.json` (100 primary comparisons), `J_comparisons.json` (secondary), `summary.json`
**Reproduce:** `python -m neurotwin.experiments.evaluate --split test --out docs/data/stage6`. This needs the five agents in `results/ppo/final_A/`, which `python -m neurotwin.experiments.train_ppo --name final_A --agents 5` regenerates deterministically.

> **Simulation results only.** The model is not calibrated to any real hospital, and nothing here says anything about real-world performance.

## 1. Design
- **Policies:** PPO (5 independently trained agents), FCFS, Priority, Load-Balancing, Trunk-Reservation, MIP. Baselines use their tuned parameters (Stage 4); PPO uses its selected configuration (Stage 5).
- **Scenarios:** A (normal, the training scenario) and B–E (stress, **never seen in training**).
- **Seeds:** 50 test seeds (200000–200049), the same seeds for every policy (CRN).
- **Protocol:** full episodes, starting from an empty hospital, with a 35-day warm-up and 14 measured days.
- **Statistics (RESEARCH.md §6.14):** hierarchical paired bootstrap (resampling agents, then seeds; B = 10,000) and Holm correction over 100 primary comparisons (PPO vs 5 baselines × 5 scenarios × 4 metrics). Effect sizes d_z and Cliff's δ, with Wilcoxon as a sensitivity check (all in `comparisons.json`).

## 2. Protocol amendment (deviation, disclosed)
The first analysis of the test run was stopped by the NaN guard: one PPO episode had no placed ED patient from the measurement window, so the pre-registered mean boarding time was undefined. A definedness audit (counts only, no performance values) showed that patients still waiting at the horizon occur in **84–100% of stressed episodes for every policy**. The pre-registered placed-only boarding metric ignored them, which *rewards* leaving hard patients waiting. Boarding metrics were changed to **censoring-inclusive**: still-waiting patients count with their wait up to the horizon, a lower bound. The change was:
- applied uniformly to all policies;
- committed before recomputation (9a1726f);
- recomputed deterministically, so the episodes themselves are unchanged.

The placed-only metric is kept as a secondary column. J was unaffected.

## 3. Headline result: overall cost J per measured day (secondary, descriptive)
Mean over 50 test seeds; PPO is averaged over its 5 agents. Lower is better.

| Scenario | PPO | Load-Bal. | Priority | Trunk-Res. | MIP | FCFS |
|---|---|---|---|---|---|---|
| A (train) | 208.1 | **205.0** | 227.8 | 227.8 | 242.9 | 338.2 |
| B demand | 838.9 | **835.6** | 922.4 | 922.4 | 939.2 | 1317.5 |
| C staff | **1187.6** | 1227.5 | 1247.9 | 1247.9 | 1460.4 | 1855.4 |
| D capacity | 930.4 | **899.6** | 1014.5 | 1014.5 | 1098.9 | 1571.1 |
| E combined | 1453.5 | **1432.2** | 1511.9 | 1511.9 | 1811.5 | 2300.4 |

**Paired difference PPO − baseline [95% hierarchical bootstrap CI]:**

| Scenario | vs Load-Balancing | vs Priority | vs MIP | vs FCFS |
|---|---|---|---|---|
| A | +3.1 [−4.4, 11.5] | **−19.8 [−28.2, −11.4]** | **−34.9 [−59.0, −15.3]** | **−130.1** |
| B | +3.3 [−56.0, 133.3] | −83.5 [−145.8, 43.8] | −100.3 [−185.4, 30.5] | **−478.6** |
| C | −39.9 [−97.9, 17.4] | **−60.3 [−117.5, −4.5]** | **−272.8** | **−667.8** |
| D | +30.8 [−9.8, 90.0] | **−84.1 [−128.4, −25.5]** | **−168.5** | **−640.8** |
| E | +21.3 [−40.9, 78.5] | −58.4 [−122.7, 2.4] | **−358.0** | **−846.9** |

Bold: the CI excludes 0.

**Reading:**
- **PPO is statistically indistinguishable from Load-Balancing, the best tuned heuristic, in every scenario** (all CIs include 0; point estimates from −3.2% to +3.4%).
- PPO is clearly better than FCFS everywhere and better than the exactly solved MIP in A, C, D and E. It is better than Priority in A, C and D, and probably in B and E, where the CIs only just include 0.
- **The validation advantage over Load-Balancing (Stage 5: 151 vs 164) did not hold on test seeds.** This is the optimistic bias of model selection that a separate test set is designed to expose.

## 4. Primary metrics (Holm-corrected; 45 of 100 comparisons significant)
Means over the 50 test seeds (boarding is censoring-inclusive):

| Scenario | Metric | PPO | LB | Priority | MIP | FCFS |
|---|---|---|---|---|---|---|
| A | Mean boarding (h) | 3.11 | 3.07 | 3.46 | 3.29 | 4.55 |
| A | Off-service /100 | 12.97 | 9.29 | 5.52 | 7.89 | 18.62 |
| C | 90th-pct boarding (h) | 48.88 | 55.92 | 72.05 | 42.00 | 38.43 |
| E | Mean boarding (h) | 19.08 | 17.77 | 19.37 | 22.55 | 27.29 |
| E | Diversion /100 | 14.27 | 11.98 | 14.58 | 15.07 | 22.72 |

Significant after Holm correction, in summary:
- **PPO vs Priority / Trunk-Reservation:**
  - lower 90th-percentile boarding in B, C and E, by 13.5–23.2 h, with large effects (d_z −0.87 to −1.30);
  - lower off-service rates under every stress scenario, by 6–11 per 100;
  - lower mean boarding in A;
  - **but** more off-service placement in A (+7.5 per 100, d_z 1.46).
- **PPO vs Load-Balancing:** no significant difference in boarding or diversion. **Load-Balancing has significantly fewer off-service placements** in A, B, D and E (2.8–5.2 per 100).
- **PPO vs MIP:** lower mean boarding in C; but the MIP has a **lower 90th-percentile boarding** in B, D and E. The MIP's urgency-weighted objective protects the tail.
- **PPO vs FCFS:** better on nearly everything. FCFS has a lower 90th percentile in C, D and E, because it serves the longest waiters first, at a large cost to average boarding.

The full table, with CIs, effect sizes and Wilcoxon p-values, is in `comparisons.json`.

**Trade-off profile:** PPO doesn't dominate any strong baseline. It sits on a different point of the trade-off curve, with much better tail boarding than Priority, at the price of more off-service placement than Load-Balancing.

## 5. Robustness finding: a learned-policy failure mode
**5 of 1,250 PPO episodes collapsed** (diversion above 50%). **No baseline episode collapsed** (0 of 1,250). All five collapses came from **one agent (agent 0)**, in scenarios B, D and E: 2% of that agent's stressed episodes.

**Timeline of the worst case** (B, seed 200000):
- Normal operation until about day 8.
- The ED fills to capacity around day 10 and diversions begin. This is a state far outside the training distribution.
- The agent then increasingly chooses HOLD. Occupied beds fall from 137 to 3 while the queue grows to 331.
- 99% of arrivals are diverted (J = 8,331, against ~800 normally).

This is a self-reinforcing out-of-distribution failure. An unfamiliar state produces holding, which makes the state more unfamiliar.

**Agent 0 had the best validation score of the five (147.9).** The best-looking agent was the most fragile, a concrete argument for (a) training and reporting several independent agents and (b) stress-testing beyond the training distribution.

**Sensitivity (not pre-registered; context only):** excluding agent 0, PPO against Load-Balancing in J ranges from −1.8% to +2.7%, with all CIs including 0. The headline conclusion doesn't depend on the fragile agent.

## 6. What this means for the research questions (preliminary; Stage 7 completes them)
- **RQ1 (learned look-ahead vs optimal myopic):** supported in cost terms. PPO beats the exactly solved MIP in J in 4 of 5 scenarios, while the MIP keeps a better 90th-percentile boarding in three stress scenarios.
- **Primary hypothesis** (PPO reduces boarding vs heuristic and optimisation baselines): **partly supported.** Yes against Priority, Trunk-Reservation, FCFS and MIP on most measures. **Not supported against Load-Balancing**, which PPO only matches.
- **RQ3 (generalisation):** PPO's average performance relative to the baselines carries over from A to the unseen stress scenarios. But a learned policy has a **failure mode the heuristics lack** (§5).

## 7. Limitations
- Results are for this simulated hospital, these parameter ranges, and PPO with a roughly half-day effective look-ahead (Stage 5).
- J comparisons are secondary and not multiplicity-corrected.
- Censoring-inclusive boarding is a lower bound for patients still waiting at the horizon.
- One protocol amendment was made after the run (§2). It was disclosed, applied uniformly, and triggered by definedness alone.
