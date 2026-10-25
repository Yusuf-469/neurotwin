// Typed client for the NeuroTwin API (neurotwin/api/app.py).
// The API base URL comes from NEXT_PUBLIC_API_URL (default: local uvicorn on port 8000).

export const API_URL: string = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Status {
  label: string;
  scenario: string;
  seed: number;
  policy: string;
  time: number;
  finished: boolean;
  horizon_hours: number;
  warmup_hours: number;
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
}

export interface StudyResults {
  label: string;
  summary: SummaryEntry[] | null;
  comparisons: Comparison[] | null;
  robustness: { robustness_index: Record<string, { estimate: number; ci: [number, number] }> } | null;
  crn_check: { metric: string; baseline: string; width_ratio_independent_over_paired: number; variance_reduction: number }[] | null;
  structure: Record<string, { n_states: number; spearman_rho: number; bins: { free_beds: string | number; mean_p_hold: number }[] }> | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; label: string }>("/api/health"),
  policies: () => request<{ policies: string[] }>("/api/policies"),
  scenarios: () => request<{ scenarios: Record<string, Record<string, number | string>> }>("/api/scenarios"),
  start: (scenario: string, seed: number, policy: string) =>
    request<StartResponse>("/api/simulation/start", { method: "POST", body: JSON.stringify({ scenario, seed, policy }) }),
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
};
