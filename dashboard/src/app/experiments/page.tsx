"use client";

import { useEffect, useState } from "react";
import { api, type StudyResults } from "@/lib/api";

const POLICIES = ["ppo", "load_balancing", "priority", "trunk_reservation", "mip", "fcfs"];
const SCENARIOS = ["A", "B", "C", "D", "E"];
const METRICS: { key: string; label: string }[] = [
  { key: "J_per_day", label: "Cost J per day" },
  { key: "mean_boarding_h", label: "Mean boarding (h, censoring-inclusive)" },
  { key: "p90_boarding_h", label: "90th-percentile boarding (h)" },
  { key: "diversion_per_100", label: "Diversions per 100 arrivals" },
  { key: "off_service_per_100", label: "Off-service per 100 placements" },
];

export default function ExperimentsPage() {
  const [data, setData] = useState<StudyResults | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.results().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">loading…</p>;

  // Mean ± half-width of the 95% t-interval over the 50 test seeds.
  const cell = (scenario: string, policy: string, metric: string): string => {
    const e = data.summary?.find((s) => s.scenario === scenario && s.policy === policy);
    const v = e?.[metric];
    if (!Array.isArray(v)) return "—";
    const digits = metric.includes("per_100") || metric.includes("_h") ? 2 : 1;
    return `${v[0].toFixed(digits)} ± ${((v[2] - v[1]) / 2).toFixed(digits)}`;
  };
  const significant = (data.comparisons ?? []).filter((c) => c.significant_holm).length;

  return (
    <>
      <div className="panel">
        <h2>Study results (test seeds, pre-computed)</h2>
        <p className="muted">
          {data.label} 50 test seeds per scenario; PPO averaged over 5 independently trained
          agents. A is the training scenario; B–E are stress scenarios never seen in training.
          {" "}{significant} of {data.comparisons?.length ?? 0} primary PPO-vs-baseline comparisons
          are significant after Holm correction. See docs/stage6_evaluation_report.md.
        </p>
      </div>
      {METRICS.map((m) => (
        <div className="panel scroll" key={m.key}>
          <h2>{m.label} (lower is better; mean ± 95% CI half-width)</h2>
          <table>
            <thead><tr><th>scenario</th>{POLICIES.map((p) => <th key={p}>{p}</th>)}</tr></thead>
            <tbody>
              {SCENARIOS.map((s) => (
                <tr key={s}><td>{s}</td>{POLICIES.map((p) => <td key={p}>{cell(s, p, m.key)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {data.robustness && (
        <div className="panel scroll">
          <h2>Robustness index (mean % degradation from A to B–E; lower = more robust)</h2>
          <table>
            <thead><tr><th>policy</th><th>index</th><th>95% CI</th></tr></thead>
            <tbody>
              {Object.entries(data.robustness.robustness_index)
                .sort((a, b) => a[1].estimate - b[1].estimate)
                .map(([p, v]) => (
                  <tr key={p}><td>{p}</td><td>{v.estimate.toFixed(1)}%</td><td>[{v.ci[0].toFixed(1)}, {v.ci[1].toFixed(1)}]</td></tr>
                ))}
            </tbody>
          </table>
          <p className="muted">Relative degradation favours policies that are already poor in A; read with the absolute tables above.</p>
        </div>
      )}
      {data.crn_check && (
        <div className="panel scroll">
          <h2>Common random numbers: confidence-interval narrowing (scenario A)</h2>
          <table>
            <thead><tr><th>metric</th><th>PPO vs</th><th>CI narrower by</th><th>variance reduction</th></tr></thead>
            <tbody>
              {data.crn_check.map((c) => (
                <tr key={c.metric + c.baseline}>
                  <td>{c.metric}</td><td>{c.baseline}</td>
                  <td>{c.width_ratio_independent_over_paired.toFixed(1)}×</td>
                  <td>{(100 * c.variance_reduction).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data.structure && (
        <div className="panel scroll">
          <h2>Learned bed reservation: P(hold) when only low-priority patients wait</h2>
          <table>
            <thead>
              <tr><th>scenario</th>{data.structure.A?.bins.map((b) => <th key={String(b.free_beds)}>{b.free_beds} free</th>)}<th>Spearman ρ</th></tr>
            </thead>
            <tbody>
              {Object.entries(data.structure).map(([s, v]) => (
                <tr key={s}>
                  <td>{s}</td>
                  {v.bins.map((b) => <td key={String(b.free_beds)}>{b.mean_p_hold.toFixed(3)}</td>)}
                  <td>{v.spearman_rho.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
