// Typed client for the NeuroTwin API (neurotwin/api/app.py).
// Production (static export served by FastAPI): same origin, so the base URL is "".
// Development (`npm run dev`): .env.development points at the local API on port 8000.

export const API_URL: string = process.env.NEXT_PUBLIC_API_URL ?? "";

export interface Status {
  label: string;
  scenario: string;
  seed: number;
  policy: string;
  time: number;
  finished: boolean;
  horizon_hours: number;
  warmup_hours: number;
  conditions?: Conditions;
  /** Work done by the last step request (only on step responses). */
  step?: { compute_ms: number; decisions_made: number; simulated_hours: number };
}

export interface Conditions {
  arrival_multiplier: number;
  staff_multiplier: number;
  capacity_multiplier: number;
}

export interface StartResponse extends Status {
  session_id: string;
}

export interface WaitingRequest {
  rid: number;
  type: string;
  priority: number;
  created_at: number;
  primary_unit: string;
  eligible_units: string[];
  from_ed: boolean;
}

export interface HospitalState {
  time: number;
  physical_beds: Record<string, number>;
  staffed_capacity: Record<string, number>;
  occupied: Record<string, number>;
  nurses_on_shift: Record<string, number>;
  doctors_on_shift: number;
  ed_count: number;
  ed_waiting_for_doctor: number;
  ed_in_treatment: number;
  waiting_requests: WaitingRequest[];
  off_service_in_beds: number;
  cumulative_placements: number;
  cumulative_diversions: number;
  cumulative_off_service: number;
}

export interface StateResponse extends Status {
  state: HospitalState;
  episode_metrics?: Record<string, number>;
}

export interface HistoryPoint {
  time: number;
  occupied: Record<string, number>;
  staffed_capacity: Record<string, number>;
  waiting: number;
  ed_count: number;
  cumulative_diversions: number;
  cumulative_off_service: number;
}

export interface DecisionRecord {
  decision_id: number;
  time: number;
  policy: string;
  action: "PLACE" | "HOLD";
  request_type?: string;
  priority?: number;
  waited_hours?: number;
  unit?: string;
  primary_unit?: string;
  off_service?: boolean;
  n_waiting: number;
}

export interface Explanation {
  decision_id: number;
  summary: string;
  reasoning: string;
  reliability_indicator: string;
  facts_used: string[];
  alternatives_considered: string[];
  limitations: string;
  generated_by: string;
}

export interface WhatIfResult {
  label: string;
  start_time: number;
  horizon_hours: number;
  scenario: string;
  n_futures: number;
  per_policy: Record<string, Record<string, { mean: number; n: number }>>;
  comparisons: { policy: string; reference: string; metric: string; diff_mean: number; ci95: [number, number] }[];
}

export interface WhatIfRequestBody {
  horizon_hours: number;
  n_futures: number;
  arrival_multiplier: number;
  staff_multiplier: number;
  capacity_multiplier: number;
  policies: string[];
}

export interface SummaryEntry {
  scenario: string;
  policy: string;
  [metric: string]: string | [number, number, number];
}

export interface Comparison {
  scenario: string;
  metric: string;
  baseline: string;
  diff_mean: number;
  ci_low: number;
  ci_high: number;
  p_holm: number;
  significant_holm: boolean;
  d_z: number;
  ppo_mean: number;
  baseline_mean: number;
}

export interface JComparison {
  scenario: string;
  baseline: string;
  ppo_mean: number;
  baseline_mean: number;
  diff: number;
  ci_low: number;
  ci_high: number;
  rel_diff_pct: number;
}

export interface StudyResults {
  label: string;
  summary: SummaryEntry[] | null;
  comparisons: Comparison[] | null;
  J_comparisons: JComparison[] | null;
  robustness: { robustness_index: Record<string, { estimate: number; ci: [number, number] }> } | null;
  crn_check: { metric: string; baseline: string; width_ratio_independent_over_paired: number; variance_reduction: number }[] | null;
  structure: Record<string, { n_states: number; spearman_rho: number; bins: { free_beds: string | number; mean_p_hold: number }[] }> | null;
  /** Stage 7: ablated variant vs full PPO (baseline field = "full_vs_<variant>"; diff = ablated - full). */
  ablations: Comparison[] | null;
  /** Stage 7: PPO-DR vs PPO (diff = PPO-DR - PPO). */
  ppo_dr: Comparison[] | null;
  /** Stage 7: lambda-conditioned constrained PPO vs PPO, plus budget-satisfaction shares. */
  constrained: { comparisons: Comparison[]; budget: Record<string, number> } | null;
  /** Stage 7: safety shield (RESEARCH.md §6.25); collapses out of 1,250 test episodes. */
  shield: { collapses: Record<string, number>; n_episodes: number; comparisons: Comparison[] } | null;
  /** Stage 7 §6.26: selected candidate vs Load-Balancing on the fresh seed block. */
  beat_rule: { selected: string; primary_vs_load_balancing: Comparison[]; secondary_vs_ppo: Comparison[]; collapses: Record<string, number> } | null;
  beat_rule_validation: { validation_J: Record<string, { mean_J: number }>; selected: string } | null;
}

export interface RaceLane {
  name: string;
  rank: number;
  cost: number;
  cost_per_day: number;
  turned_away: number;
  wrong_ward: number;
  placements: number;
  waiting_now: number;
  compute_ms: number;
  trace: [number, number][];
}

export interface RaceState {
  elapsed_hours: number;
  horizon_hours: number;
  finished: boolean;
  lanes: RaceLane[];
  compute_ms?: number;
}

export interface RaceStartBody {
  competitors: string[];
  scenario: string;
  custom?: Conditions;
  seed: number;
  start: "busy" | "empty" | "session";
  session_id?: string;
  horizon_hours: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    // FastAPI errors: {"detail": "..."} or {"detail": [{"msg": "..."}]}; show the message only.
    let message = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string | { msg: string }[] };
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`Request failed (${res.status}): ${message}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; label: string }>("/api/health"),
  policies: () => request<{ policies: string[] }>("/api/policies"),
  scenarios: () => request<{ scenarios: Record<string, Record<string, number | string>> }>("/api/scenarios"),
  start: (scenario: string, seed: number, policy: string, custom?: Conditions) =>
    request<StartResponse>("/api/simulation/start", {
      method: "POST", body: JSON.stringify({ scenario, seed, policy, ...(custom ? { custom } : {}) }),
    }),
  step: (sid: string, hours: number) =>
    request<Status>(`/api/simulation/${sid}/step`, { method: "POST", body: JSON.stringify({ hours }) }),
  reset: (sid: string) => request<Status>(`/api/simulation/${sid}/reset`, { method: "POST" }),
  state: (sid: string) => request<StateResponse>(`/api/simulation/${sid}/state`),
  history: (sid: string) => request<{ history: HistoryPoint[] }>(`/api/simulation/${sid}/history`),
  decisions: (sid: string, limit = 30) =>
    request<{ total: number; decisions: DecisionRecord[] }>(`/api/simulation/${sid}/decisions?limit=${limit}`),
  explanation: (sid: string, id: number) =>
    request<{ explanation: Explanation }>(`/api/simulation/${sid}/explanations/${id}`),
  whatif: (sid: string, body: WhatIfRequestBody) =>
    request<WhatIfResult>(`/api/simulation/${sid}/whatif`, { method: "POST", body: JSON.stringify(body) }),
  results: () => request<StudyResults>("/api/experiments/results"),
  raceStart: (body: RaceStartBody) =>
    request<RaceState & { race_id: string; start_label: string; start_time: number; setup_ms: number }>("/api/race/start", {
      method: "POST", body: JSON.stringify(body),
    }),
  raceStep: (rid: string, hours: number) =>
    request<RaceState>(`/api/race/${rid}/step`, { method: "POST", body: JSON.stringify({ hours }) }),
};
