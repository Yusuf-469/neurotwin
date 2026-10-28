// Fixed policy -> colour mapping (categorical order validated in both themes; colour follows the
// entity, never its rank, so filtering never repaints a policy). Every technical name also has a
// plain-language label and a one-line description, so the dashboard reads for non-specialists;
// the technical name stays visible as secondary text.

export interface PolicyMeta {
  key: string;
  label: string;
  tech: string;
  color: string;
  desc: string;
}

export const POLICIES: PolicyMeta[] = [
  { key: "ppo", label: "RL agent v1", tech: "PPO neural network, built from scratch", color: "var(--series-1)",
    desc: "A neural network trained with reinforcement learning over 3 million simulated decisions. Nobody wrote its rules." },
  { key: "load_balancing", label: "Balance-the-wards rule", tech: "Load-Balancing", color: "var(--series-2)",
    desc: "Sends each patient to whichever suitable ward has the most room." },
  { key: "priority", label: "Most-urgent-first rule", tech: "Priority", color: "var(--series-3)",
    desc: "Always serves the most urgent patient first, into their own specialty ward." },
  { key: "trunk_reservation", label: "Keep-a-bed-free rule", tech: "Trunk-Reservation", color: "var(--series-4)",
    desc: "Like most-urgent-first, but can hold back beds for emergencies (tuning chose zero)." },
  { key: "mip", label: "Instant optimiser", tech: "MIP (one-step optimal)", color: "var(--series-5)",
    desc: "Solves the best possible assignment right now, without thinking ahead." },
  { key: "fcfs", label: "First-come-first-served", tech: "FCFS", color: "var(--series-6)",
    desc: "Whoever has waited longest gets the next bed, regardless of urgency." },
];

/** The RL agent with the safety guard (RESEARCH.md §6.25). Not part of the original six-way
 *  study, so it is kept out of POLICIES; it shares the agent's colour because it is the same
 *  trained network, and is always shown with its own label. */
export const SHIELDED: PolicyMeta = {
  key: "ppo_shielded", label: "RL agent v1 + safety guard", tech: "PPO with a no-hold-while-ED-full guard", color: "color-mix(in srgb, var(--series-1) 55%, var(--text-muted))",
  desc: "The same agent, but it may never keep a bed empty while ambulances are being turned away.",
};

/** The improved agent (RESEARCH.md §6.26): learned from the best rule, then improved with RL;
 *  five such agents vote. Same family colour as the RL agent, always shown with its own label. */
export const ENSEMBLE: PolicyMeta = {
  key: "ppo_ensemble", label: "RL agent v2", tech: "imitation + PPO fine-tuning, 5-agent vote", color: "color-mix(in srgb, var(--violet) 55%, var(--text-muted))",
  desc: "Learned from the best hand-tuned rule, then improved itself with reinforcement learning; five copies vote on each decision.",
};

/** The latest agent (RESEARCH.md §6.27): v2 further trained on varied days covering the crisis
 *  conditions; five copies vote. */
export const V3: PolicyMeta = {
  key: "ppo_v3", label: "RL agent v3 (latest)", tech: "crisis-trained: imitation + PPO + varied days, 5-agent vote", color: "var(--violet)",
  desc: "Learned from the best rule, improved with reinforcement learning, then practised on crisis days; five copies vote on each decision.",
};

/** Decision-makers offered in the live hospital and race tools. */
export const LIVE_POLICIES: PolicyMeta[] = [V3, ENSEMBLE, POLICIES[0], SHIELDED, ...POLICIES.slice(1)];

/** How decision-makers are grouped in pickers: one RL agent with its versions, then the rules. */
export const POLICY_GROUPS: { label: string; keys: string[] }[] = [
  { label: "RL agent (latest)", keys: ["ppo_v3"] },
  { label: "Earlier RL versions", keys: ["ppo_ensemble", "ppo", "ppo_shielded"] },
  { label: "Hand-designed rules", keys: ["load_balancing", "priority", "trunk_reservation", "mip", "fcfs"] },
];

export const policyMeta = (key: string): PolicyMeta =>
  LIVE_POLICIES.find((p) => p.key === key) ??
  { key, label: key, tech: key, color: "var(--text-muted)", desc: "" };

export const UNIT_COLORS: Record<string, string> = {
  MED: "var(--series-1)",
  SURG: "var(--series-2)",
  ICU: "var(--series-3)",
};

export const UNIT_LABELS: Record<string, string> = {
  MED: "Medical ward",
  SURG: "Surgical ward",
  ICU: "Intensive care",
};

/** Situations are shown as numbers 1-5; the letters A-E remain the internal keys (and the
 *  names used in RESEARCH.md and the stage reports). */
export const SCENARIO_NUM: Record<string, number> = { A: 1, B: 2, C: 3, D: 4, E: 5 };
export const scenNum = (key: string): string => String(SCENARIO_NUM[key?.[0]] ?? key);

export const SCENARIO_LABELS: Record<string, string> = {
  A: "1 · Normal day",
  B: "2 · Demand surge",
  C: "3 · Staff shortage",
  D: "4 · Bed loss",
  E: "5 · Everything at once",
};

import type { IconName } from "@/components/icons";

export const SCENARIO_INFO: Record<string, { short: string; desc: string; icon: IconName }> = {
  A: { short: "Normal day", icon: "hospital", desc: "Typical demand and full staffing. The only situation the RL agent trained on." },
  B: { short: "Demand surge", icon: "trend", desc: "10% more patients arrive than usual, like a bad flu week." },
  C: { short: "Staff shortage", icon: "staff", desc: "12% fewer nurses and doctors, so fewer beds can be staffed." },
  D: { short: "Bed loss", icon: "bed", desc: "12% of beds are out of service, e.g. closed bays." },
  E: { short: "Everything at once", icon: "alert", desc: "8% more patients, 10% fewer staff and 10% fewer beds at once: the hardest test." },
};

export interface MetricMeta {
  key: string;
  label: string;
  tech: string;
  unit: string;
  digits: number;
  desc: string;
}

export const METRICS: MetricMeta[] = [
  { key: "J_per_day", label: "Overall cost score", tech: "operational cost J", unit: "per day", digits: 0,
    desc: "One score combining waiting (urgent patients count more), turned-away ambulances and wrong-ward placements. Lower is better." },
  { key: "mean_boarding_h", label: "Average wait for a bed", tech: "mean boarding time", unit: "hours", digits: 2,
    desc: "How long emergency patients wait for a ward bed on average." },
  { key: "p90_boarding_h", label: "Longest waits", tech: "90th-percentile boarding", unit: "hours", digits: 1,
    desc: "The wait that only the unluckiest 10% of patients exceed." },
  { key: "diversion_per_100", label: "Ambulances turned away", tech: "diversions", unit: "per 100 arrivals", digits: 2,
    desc: "When the emergency department is full, new arrivals are sent elsewhere." },
  { key: "off_service_per_100", label: "Wrong-ward placements", tech: "off-service placements", unit: "per 100", digits: 1,
    desc: "Patients placed in a ward of the wrong specialty because their own was full." },
];

/** Plain names for the five urgency levels (1 = most urgent; 4-5 are planned admissions). */
export const PRIORITY_NAMES: Record<number, string> = {
  1: "Critical",
  2: "Very urgent",
  3: "Urgent",
  4: "Standard",
  5: "Planned",
};

// Urgency is ordinal; status colours are reserved for it and always shown with a text label.
export const PRIORITY_COLORS: Record<number, string> = {
  1: "var(--status-critical)",
  2: "var(--status-critical)",
  3: "var(--status-serious)",
  4: "var(--text-muted)",
  5: "var(--text-muted)",
};
