# NeuroTwin V2 - Project Specification

## 1. Problem Statement

Hospitals face complex operational challenges in resource allocation, including bed management, staff scheduling, and patient flow optimization. Traditional allocation policies often fail to adapt to changing demand patterns, leading to extended patient wait times, underutilized resources, and poor throughput. There is a need for an intelligent, adaptive approach to hospital resource management that can learn from patterns and optimize decisions under varying operational conditions.

## 2. Motivation

- Healthcare operational efficiency directly impacts patient outcomes and hospital costs
- Discrete-event simulation is a standard tool for hospital operations research
- Reinforcement learning offers the potential for adaptive policy learning under changing conditions
- A controlled simulated environment enables reproducible experiments without risking real patient safety
- The research focuses on robustness and generalization of adaptive policies across scenarios

## 3. Target Users

- **Primary**: Final-year CSE-AIML students conducting research in AI-driven hospital operations
- **Secondary**: Researchers in healthcare operations management and clinical informatics
- **Tertiary**: Hospital administrators seeking to understand AI-driven optimization concepts

## 4. System Goals

1. Simulate a hospital environment with patient flow, staff allocation, and resource utilization
2. Maintain a digital twin representation with state tracking, history, and what-if evaluation capability
3. Implement multiple baseline allocation policies (heuristic and optimization-based)
4. Create a reinforcement learning agent that learns adaptive allocation policies
5. Evaluate policies across normal and stressed operational scenarios with statistical rigor
6. Support reproducible experiments with configurable scenarios and multi-seed evaluation
7. Provide counterfactual/what-if analysis for comparing policies from comparable states
8. Visualize hospital operations through an interactive dashboard
9. Generate explanations for allocation decisions (LLM-based, post-hoc only)

## 5. Functional Requirements

### 5.1 Simulation Engine
- Discrete-event simulation of hospital operations
- Model: Emergency Department, triage/priority queue, two specialty general wards (medical,
  surgical), ICU, doctors, nurses (staffed capacity)
- Simulate: emergency and elective arrivals, triage, ED treatment, bed requests, primary and
  off-service placement, ICU step-down, deterioration and escalation, discharges, diversion
- Model: staff availability and shift schedules
- Support: configurable simulation parameters (arrival rates, service times, resource counts, staffing levels)
- Validation: reduced configurations must reproduce closed-form queueing results (Erlang B/C,
  Little's law) before any policy result is reported (RESEARCH.md §6.17)
- Common Random Numbers via independent RNG streams per stochastic source
- **Note**: Operating rooms are out of scope for V1; may be added as future extension

### 5.2 Digital Twin State Representation
- Maintain current state of all hospital entities and resources
- Track resource utilization and patient queues in real-time
- Store historical state for analysis
- Provide state snapshot export/import for reproducibility
- Support what-if/counterfactual evaluation: simulate alternative futures from comparable states

### 5.3 Resource Allocation Policies
- **Baseline Policies**:
  - FCFS (First-Come, First-Served)
  - Priority-based allocation (severity/urgency)
  - Load-balancing heuristic
  - Trunk-reservation (tuned bed-reservation thresholds; Miller, 1969)
  - Optimization-based allocation: exactly solved myopic assignment (RESEARCH.md §6.9)
- All policies operate on comparable simulation states
- All policies evaluated using identical metrics and scenarios

### 5.4 Reinforcement Learning Agent
- Gymnasium-compatible environment interface (semi-Markov: variable step duration)
- PPO (Proximal Policy Optimization) as primary algorithm, **implemented in-house in PyTorch**
  and validated against a reference implementation (RESEARCH.md §6.18)
- Compact, masked, sequential placement actions (not individual resource assignment)
- Time-integrated cost reward aligned with the evaluation metrics (waiting, diversion,
  off-service placement)
- Training under normal conditions; evaluation under normal (held-out seeds) and stressed conditions
- Generalization testing on unseen scenarios
- K = 5 independent training seeds per configuration
- Policy-structure analysis against trunk-reservation theory
- Optional extension: constrained PPO (PPO-Lagrangian) with a diversion budget

### 5.5 Experiment and Evaluation Infrastructure
- Configurable scenario families: normal, high demand, staff shortage, capacity constraint, combined stress
- Multi-seed evaluation (30-50 seeds per policy/scenario combination for final evaluation)
- Statistical evaluation: means, standard deviations, confidence intervals, effect sizes, significance tests
- Robustness analysis: measure performance degradation under stress
- Ablation studies (where scientifically justified)

### 5.6 What-If Engine
- Counterfactual analysis from comparable initial states
- Evaluate multiple policies under alternative scenarios
- Example: "What if emergency demand increases 30%?"
- Dashboard clearly labels all results as simulation results

### 5.7 ML Prediction (Optional Extension)
- Demand forecasting integration into RL state
- Compare RL with prediction vs RL without prediction
- Report findings honestly regardless of outcome

### 5.8 AI Explanation Layer
- Generate natural language explanations for allocation decisions post-hoc
- Use actual logged state/action information
- Clearly distinguish facts from generated explanations
- Never fabricate results or control simulation
- Include uncertainty/limitations indicators

### 5.9 Interactive Dashboard
- Real-time visualization of hospital state
- Simulation control (start, pause, step, speed)
- Historical performance metrics with statistical summaries
- What-if scenario evaluation interface
- Decision explanation display

## 6. Non-Functional Requirements

- **Reproducibility**: Simulation must support seeded random number generation; all seeds documented
- **Modularity**: Components must be loosely coupled and independently testable
- **Extensibility**: Easy to add new entity types, policies, and algorithms
- **Performance**: Simulation should run at least 10x real-time speed
- **Statistical Rigor**: Experiments use sufficient seeds and appropriate statistical tests
- **Documentation**: Clear documentation of all components, APIs, and experimental methodology

## 7. Project Boundaries

### In Scope
- Discrete-event simulation of hospital operations (simplified model)
- Digital twin state management with what-if capability
- RL-based decision agent (Gymnasium-compatible, PPO primary)
- Rule-based and optimization baseline policies
- Statistical evaluation framework
- What-if/counterfactual analysis
- AI explanation layer (post-hoc, not controlling)
- Web dashboard for visualization and interaction
- Reproducible experiment infrastructure

### Out of Scope
- Actual hospital integration or deployment
- Real-time data feeds from actual hospitals
- Medical device integration
- Electronic health record (EHR) integration
- Clinical decision support for real patients
- Regulatory compliance for medical devices
- Operating room modeling (V1)
- Production deployment infrastructure (Kubernetes, AWS)

## 8. Assumptions

1. Synthetic data is sufficient for model training and evaluation
2. The simulation operates on discrete events with time-driven updates
3. All patient data is synthetic and does not represent real individuals
4. The simulation does not require real-time feeds from external systems
5. A simplified hospital model captures sufficient dynamics for the research question
6. PPO is sufficient as the primary RL algorithm; other algorithms may be added if justified
7. The optimization baseline is the myopic assignment formulation in RESEARCH.md §6.9
8. Model parameters (LOS distributions, staffing ratios, off-service LOS multiplier,
   deterioration hazard) are plausible assumptions, not estimates from real data; the most
   influential receive sensitivity analysis (RESEARCH.md §6.6, §6.20)

## 9. Limitations

1. Results from simulation may not generalize to real hospital environments
2. The system does not account for clinical nuances that affect patient outcomes
3. RL policies learned in simulation may not transfer to real-world deployment
4. The simulation uses simplified models that do not capture all real-world dynamics
5. LLM explanations may occasionally produce inaccurate or misleading descriptions
6. Computational resources may limit the scale and complexity of simulations
7. Statistical significance depends on seed count and variance; 30-50 seeds may still have limited power for rare events
