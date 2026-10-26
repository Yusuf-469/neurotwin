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
  { key: "ppo", label: "AI agent", tech: "PPO (reinforcement learning)", color: "var(--series-1)",
    desc: "Learned by trial and error over millions of simulated decisions. Nobody wrote its rules." },
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

export const policyMeta = (key: string): PolicyMeta =>
  POLICIES.find((p) => p.key === key) ??
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

export const SCENARIO_LABELS: Record<string, string> = {
  A: "A · Normal (training)",
  B: "B · Demand surge",
  C: "C · Staff shortage",
  D: "D · Bed loss",
  E: "E · Combined stress",
};

export const SCENARIO_INFO: Record<string, { short: string; desc: string; icon: string }> = {
  A: { short: "Normal day", icon: "🏥", desc: "Typical demand and full staffing. The only situation the AI trained on." },
  B: { short: "Demand surge", icon: "📈", desc: "10% more patients arrive than usual, like a bad flu week." },
  C: { short: "Staff shortage", icon: "🧑‍⚕️", desc: "12% fewer nurses and doctors, so fewer beds can be staffed." },
  D: { short: "Bed loss", icon: "🛏️", desc: "12% of beds are out of service, e.g. closed bays." },
  E: { short: "Everything at once", icon: "⚠️", desc: "8% more patients, 10% fewer staff and 10% fewer beds at once: the hardest test." },
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
