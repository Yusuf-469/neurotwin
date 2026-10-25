"use client";

import { useState } from "react";
import { api, type WhatIfResult } from "@/lib/api";

interface Props {
  sessionId: string;
  policies: string[];
  disabled: boolean;
}

const METRICS: { key: string; label: string }[] = [
  { key: "J_per_day", label: "cost J / day" },
  { key: "mean_boarding_h", label: "mean boarding (h)" },
  { key: "p90_boarding_h", label: "90th pct boarding (h)" },
  { key: "diversion_per_100", label: "diversions / 100" },
  { key: "off_service_per_100", label: "off-service / 100" },
];

export default function WhatIfPanel({ sessionId, policies, disabled }: Props) {
  const [selected, setSelected] = useState<string[]>(["priority", "load_balancing"]);
  const [horizon, setHorizon] = useState(72);
  const [futures, setFutures] = useState(20);
  const [arrivals, setArrivals] = useState(1.0);
  const [staff, setStaff] = useState(1.0);
  const [capacity, setCapacity] = useState(1.0);
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const toggle = (p: string) =>
    setSelected((s) => (s.includes(p) ? s.filter((x) => x !== p) : [...s, p]));

  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await api.whatif(sessionId, {
        horizon_hours: horizon, n_futures: futures, arrival_multiplier: arrivals,
        staff_multiplier: staff, capacity_multiplier: capacity, policies: selected,
      }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h2>What-if from the current state</h2>
      <p className="muted">
        Every policy is run on the same random futures from this exact state (common random
        numbers). Stays already in progress are re-drawn given the time already spent.
      </p>
      <div className="row">
        {policies.map((p) => (
          <label key={p} style={{ flexDirection: "row", alignItems: "center" }}>
            <input type="checkbox" checked={selected.includes(p)} onChange={() => toggle(p)} /> {p}
          </label>
        ))}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <label>horizon (h)<input type="number" value={horizon} min={1} max={336} onChange={(e) => setHorizon(+e.target.value)} /></label>
        <label>futures<input type="number" value={futures} min={2} max={50} onChange={(e) => setFutures(+e.target.value)} /></label>
        <label>arrivals ×<input type="number" step={0.05} value={arrivals} onChange={(e) => setArrivals(+e.target.value)} /></label>
        <label>staff ×<input type="number" step={0.05} value={staff} onChange={(e) => setStaff(+e.target.value)} /></label>
        <label>beds ×<input type="number" step={0.05} value={capacity} onChange={(e) => setCapacity(+e.target.value)} /></label>
        <button onClick={run} disabled={disabled || busy || selected.length === 0}>
          {busy ? "running…" : "run what-if"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="scroll" style={{ marginTop: 12 }}>
          <p className="muted">{result.label}</p>
          <table>
            <thead>
              <tr><th>policy</th>{METRICS.map((m) => <th key={m.key}>{m.label}</th>)}</tr>
            </thead>
            <tbody>
              {Object.entries(result.per_policy).map(([p, m]) => (
                <tr key={p}>
                  <td>{p}</td>
                  {METRICS.map((x) => <td key={x.key}>{m[x.key]?.mean.toFixed(2)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ marginTop: 8 }}>
            Paired difference vs {result.comparisons[0]?.reference ?? "reference"} (95% CI over {result.n_futures} futures):
          </p>
          <table>
            <thead><tr><th>policy</th><th>metric</th><th>difference</th><th>95% CI</th></tr></thead>
            <tbody>
              {result.comparisons.map((c) => (
                <tr key={c.policy + c.metric}>
                  <td>{c.policy}</td>
                  <td>{c.metric}</td>
                  <td>{c.diff_mean.toFixed(2)}</td>
                  <td>[{c.ci95[0].toFixed(2)}, {c.ci95[1].toFixed(2)}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
