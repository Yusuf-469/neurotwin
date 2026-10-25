# Viva and Presentation Guide

Preparation notes for presenting NeuroTwin. Every number here comes from the stage reports; cite them, not this file.

## 1. The two-minute version

Hospitals decide continuously who gets a free bed, whether to place a patient in the wrong specialty ward, and whether to keep a bed free for urgent arrivals. NeuroTwin is a validated discrete-event digital twin of an ED, two wards and an ICU, built to test whether a reinforcement-learning policy can make these decisions better than rules or optimisation, and whether it stays effective under stress it never trained on.

I wrote the simulator from scratch and validated it against queueing theory. I implemented PPO from scratch with time-dependent discounting and action masking, checked it against stable-baselines3, and compared it with five tuned baselines, including an exactly solved optimisation model. The evaluation used 2,500 test episodes with pre-registered statistics.

**Result:** PPO matches the best tuned rule (Load-Balancing) and beats Priority, FCFS and the one-step optimiser on cost in most scenarios, including unseen stress. But one of five agents had a catastrophic failure mode that no rule-based policy has. Under normal load, the agent independently learned a soft bed-reservation policy, as queueing theory predicts.

## 2. Suggested slide outline (12–14 slides)

1. **Problem:** bed allocation decisions (who, where, hold or place); why fixed rules struggle under stress.
2. **Research questions:** look-ahead vs myopic optimisation; robustness; generalisation; learned structure.
3. **Related work and gap:** Sun, Shi & Dai (PPO for overflow) is the closest prior work; the gap is generalisation, strong baselines, and a validated simulator.
4. **Hospital model:** units, staffed capacity, pathways (diagram from ARCHITECTURE.md §5.4).
5. **Simulator validation:** Erlang C / M/M/c/K table (15/15 inside CI) and the mutation test.
6. **Digital twin:** snapshot/branch; the inspection-paradox finding (naive restart triples boarding).
7. **Baselines:** the five policies; the MIP as a transportation problem (total unimodularity, two oracles); fair tuning.
8. **Scenario calibration:** offered load, targets stated before any results; why the original scenarios were hopeless.
9. **In-house PPO:** semi-Markov GAE (γ^Δt), masking, validation against stable-baselines3 and MaskablePPO.
10. **Making PPO learn:** the low signal-to-noise diagnosis, and the horizon trade-off (about half a day).
11. **Results:** the Stage 6 cost table and paired CIs; PPO ≈ Load-Balancing; validation optimism.
12. **Robustness:** agent-0 collapse timeline; robustness index; learned reservation in A vs E.
13. **Methodology wins:** CRN (12.5× narrower CIs); protocol committed before the test run; disclosed amendment.
14. **Platform demo:** dashboard, what-if, explanations (facts vs generated text).
15. **Limitations and future work.**

## 3. Likely questions and honest answers

**"What did you implement yourself?"**
The discrete-event simulator, the digital twin (snapshots, branching, conditional resampling), all baselines including the MIP formulation, the Gymnasium environment, PPO itself (networks, GAE with γ^Δt, losses, masking, training loop), the statistics (hierarchical bootstrap, Holm), the API and the dashboard. Libraries: NumPy, SciPy (HiGHS solver, statistics), PyTorch (autograd), Gymnasium (interface only). stable-baselines3 was used only to check my PPO.

**"How do you know your simulator is correct?"**
In reduced configurations it is exactly M/M/c or M/M/c/K. All 15 analytical values (wait probability, mean wait, blocking, at three loads) fall inside the 95% CI of 30 replications. A deliberate 5% error in the arrival rate is detected, so the check has power. Exact sample-path identities (∫ queue dt = Σ waits) hold to 1e-9. This shows correct implementation, *not* realism. The parameters are plausible assumptions.

**"Why didn't PPO beat Load-Balancing?"**
It did on validation seeds (151 vs 164), but not on fresh test seeds, where the two are statistically indistinguishable. The validation numbers were also used to choose PPO's settings, so they were optimistically biased; the test set exists to remove that bias. Load-Balancing is also a strong, tuned baseline that directly targets the dominant trade-off. I report the honest result.

**"Then what is the value of RL here?"**
1. It beats the exactly solved one-step optimiser, which supports the value of look-ahead (RQ1).
2. It found a different trade-off: much better tail waits than Priority (13–23 h lower 90th percentile under stress).
3. It discovered reservation behaviour on its own.
4. The study shows *how* to evaluate RL credibly, including where it fails.

**"Why γ = 0.95 per hour, isn't that short-sighted?"**
Yes, the effective look-ahead is about half a day, and I state that. With γ ≥ 0.98 the policy could not learn: a single placement changes total cost very little relative to the noise from future arrivals, so the gradient signal vanished. Both my PPO and the reference implementation struggled the same way. Shorter horizons reduce that noise. Queueing-informed value approximations (as in Sun, Shi & Dai) are the natural next step.

**"Agent 0 collapsed. Isn't that a bug?"**
The same code produced four agents that never collapsed. Agent 0 met a state far outside its training distribution (ED full, diversions spiking), and its policy output there was "hold", which made the state worse. That's a known out-of-distribution failure of learned policies, and it's why I trained and reported five agents and stress-tested. It is also the best argument for keeping rule-based fallbacks or guardrails.

**"What are common random numbers and why do they matter?"**
Every policy faces exactly the same patients and staffing for a given seed, because every random source has its own stream and patient attributes are drawn at arrival. Paired differences then cancel most of the noise: CIs were up to 12.5× narrower than with independent seeds. Without CRN, PPO's 9% advantage over Priority would have been undetectable.

**"Did you change anything after seeing test results?"**
One thing, and it is disclosed. The NaN guard stopped the analysis because the pre-registered boarding metric ignored patients still waiting at the end. I checked only definedness counts (no performance values), made the metric censoring-inclusive for every policy, and committed the change before recomputing. Git history shows the order (b762841 → 9a1726f → b6047a5).

**"Is the explanation layer an LLM making decisions?"**
No. Explanations are generated after the fact from the decision log, and the explanation package cannot import the simulator or the policies (a test enforces this). The default explainer is deterministic. The LLM interface checks every number in generated text against the logged facts. No external LLM is connected.

**"Can this be used in a hospital?"**
No. It is a research simulation with assumed parameters, not validated on real data, and not a medical device. A real study would need real data, calibration, clinical input and governance.

## 4. Demo checklist
1. Start the API and dashboard (README "Running the platform").
2. Simulation page: scenario A, policy `ppo`, seed 3. Step one week (warm-up), then one more week.
3. Open a few decisions and explain them. Point out facts vs generated text, and the reliability indicator.
4. Run what-if: Priority vs Load-Balancing vs PPO, arrivals ×1.3, 72 h, 20 futures. Show the paired CIs.
5. Study results page: cost table with CIs, CRN narrowing, reservation table.
