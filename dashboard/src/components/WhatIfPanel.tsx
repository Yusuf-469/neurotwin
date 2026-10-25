"use client";

import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, type WhatIfResult } from "@/lib/api";
import { METRICS, POLICIES, policyMeta } from "@/lib/policies";
import { Card, Segmented, useViewToggle } from "./ui";

interface Props {
  sessionId: string;
  policies: string[];
  disabled: boolean;
}

export default function WhatIfPanel({ sessionId, policies, disabled }: Props) {
  const [selected, setSelected] = useState<string[]>(["priority", "load_balancing", "ppo"]);
  const [horizon, setHorizon] = useState(72);
  const [futures, setFutures] = useState(20);
  const [arrivals, setArrivals] = useState(1.3);
  const [staff, setStaff] = useState(1.0);
  const [capacity, setCapacity] = useState(1.0);
  const [metric, setMetric] = useState("J_per_day");
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [asTable, viewToggle] = useViewToggle();

  const toggle = (p: string) => setSelected((s) => (s.includes(p) ? s.filter((x) => x !== p) : [...s, p]));
  const ordered = POLICIES.map((p) => p.key).filter((k) => selected.includes(k) && policies.includes(k));

  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await api.whatif(sessionId, {
        horizon_hours: horizon, n_futures: futures, arrival_multiplier: arrivals,
        staff_multiplier: staff, capacity_multiplier: capacity, policies: ordered,
      }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const meta = METRICS.find((m) => m.key === metric)!;
  const reference = result?.comparisons[0]?.reference ?? ordered[0];
  const rows = (result?.comparisons ?? []).filter((c) => c.metric === metric).map((c) => ({
    policy: c.policy, label: policyMeta(c.policy).label, diff: c.diff_mean,
    err: [c.diff_mean - c.ci95[0], c.ci95[1] - c.diff_mean] as [number, number],
    lo: c.ci95[0], hi: c.ci95[1],
  }));

  return (
    <Card title="What-if: compare policies from this exact moment"
      subtitle="Every policy faces the same random futures (common random numbers); stays in progress are re-drawn given time already spent.">
      <div className="row" style={{ marginBottom: 10 }}>
        {POLICIES.filter((p) => policies.includes(p.key)).map((p) => (
          <label key={p.key} className="chip" style={{ cursor: "pointer" }}>
            <input type="checkbox" checked={selected.includes(p.key)} onChange={() => toggle(p.key)}
              style={{ margin: 0 }} />
            <i className="dot" style={{ background: p.color }} />{p.label}
          </label>
        ))}
      </div>
      <div className="row">
        <label className="field">horizon (h)<input type="number" value={horizon} min={1} max={336} onChange={(e) => setHorizon(+e.target.value)} /></label>
        <label className="field">futures<input type="number" value={futures} min={2} max={50} onChange={(e) => setFutures(+e.target.value)} /></label>
        <label className="field">arrivals ×<input type="number" step={0.05} value={arrivals} onChange={(e) => setArrivals(+e.target.value)} /></label>
        <label className="field">staff ×<input type="number" step={0.05} value={staff} onChange={(e) => setStaff(+e.target.value)} /></label>
        <label className="field">beds ×<input type="number" step={0.05} value={capacity} onChange={(e) => setCapacity(+e.target.value)} /></label>
        <button className="btn" onClick={run} disabled={disabled || busy || ordered.length < 2}>
          {busy ? "Simulating futures…" : "Run what-if"}
        </button>
      </div>
      {ordered.length < 2 && <p className="muted">Select at least two policies.</p>}
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="stack" style={{ marginTop: 16, opacity: busy ? 0.5 : 1 }}>
          <div className="row">
            <Segmented options={METRICS.map((m) => ({ value: m.key, label: m.label }))} value={metric} onChange={setMetric} />
            <span className="spacer" />
            {viewToggle}
          </div>
          <span className="muted">
            Difference from <strong>{policyMeta(reference).label}</strong> in {meta.label.toLowerCase()} ({meta.unit}),
            mean over {result.n_futures} futures with 95% CI. Below 0 = better than the reference. {result.label}
          </span>
          {asTable ? (
            <div className="scroll">
              <table>
                <thead><tr><th>policy</th>{METRICS.map((m) => <th key={m.key}>{m.label}</th>)}</tr></thead>
                <tbody>
                  {Object.entries(result.per_policy).map(([p, m]) => (
                    <tr key={p}><td>{policyMeta(p).label}</td>
                      {METRICS.map((x) => <td key={x.key}>{m[x.key]?.mean.toFixed(x.digits)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ width: "100%", height: 60 + rows.length * 46 }}>
              <ResponsiveContainer>
                <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
                  <YAxis type="category" dataKey="label" width={150} stroke="var(--axis)"
                    tick={{ fill: "var(--text-secondary)", fontSize: 12 }} tickLine={false} />
                  <ReferenceLine x={0} stroke="var(--text-muted)" />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }}
                    contentStyle={{ background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                    formatter={(v, _n, item) => {
                      const p = item.payload as { lo: number; hi: number };
                      const val = typeof v === "number" ? v.toFixed(meta.digits) : "–";
                      return [`${val} [${p.lo.toFixed(meta.digits)}, ${p.hi.toFixed(meta.digits)}]`, "difference (95% CI)"];
                    }} />
                  <Bar dataKey="diff" barSize={18} radius={4} isAnimationActive={false}>
                    {rows.map((r) => <Cell key={r.policy} fill={policyMeta(r.policy).color} />)}
                    <ErrorBar dataKey="err" width={6} strokeWidth={1.5} stroke="var(--text-secondary)" direction="x" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
