// Fixed policy -> colour mapping (categorical order validated in both themes; colour follows the
// entity, never its rank, so filtering never repaints a policy).

export interface PolicyMeta {
  key: string;
  label: string;
  color: string;
}

export const POLICIES: PolicyMeta[] = [
  { key: "ppo", label: "PPO (learned)", color: "var(--series-1)" },
  { key: "load_balancing", label: "Load-Balancing", color: "var(--series-2)" },
  { key: "priority", label: "Priority", color: "var(--series-3)" },
  { key: "trunk_reservation", label: "Trunk-Reservation", color: "var(--series-4)" },
  { key: "mip", label: "MIP (one-step optimal)", color: "var(--series-5)" },
  { key: "fcfs", label: "FCFS", color: "var(--series-6)" },
];

export const policyMeta = (key: string): PolicyMeta =>
  POLICIES.find((p) => p.key === key) ?? { key, label: key, color: "var(--text-muted)" };

export const UNIT_COLORS: Record<string, string> = {
  MED: "var(--series-1)",
  SURG: "var(--series-2)",
  ICU: "var(--series-3)",
};

export const SCENARIO_LABELS: Record<string, string> = {
  A: "A · Normal (training)",
  B: "B · Demand surge",
  C: "C · Staff shortage",
  D: "D · Bed loss",
  E: "E · Combined stress",
};

export interface MetricMeta {
  key: string;
  label: string;
  unit: string;
  digits: number;
}

export const METRICS: MetricMeta[] = [
  { key: "J_per_day", label: "Operational cost J", unit: "per day", digits: 0 },
  { key: "mean_boarding_h", label: "Mean boarding time", unit: "hours", digits: 2 },
  { key: "p90_boarding_h", label: "90th-percentile boarding", unit: "hours", digits: 1 },
  { key: "diversion_per_100", label: "Diversions", unit: "per 100 arrivals", digits: 2 },
  { key: "off_service_per_100", label: "Off-service placements", unit: "per 100", digits: 1 },
];

// Urgency is ordinal; status colours are reserved for it and always shown with a text label.
export const PRIORITY_COLORS: Record<number, string> = {
  1: "var(--status-critical)",
  2: "var(--status-critical)",
  3: "var(--status-serious)",
  4: "var(--text-muted)",
  5: "var(--text-muted)",
};
