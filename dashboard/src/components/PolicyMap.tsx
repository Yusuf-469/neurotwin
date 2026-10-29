"use client";

// Map of what the latest agent learned: each dot is one real decision, placed by the network's
// own internal representation (PCA of its last hidden layer; neurotwin/experiments/policy_map.py).
// Descriptive only: it shows how the network organises situations, not how well it performs.

import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { api, type PolicyMapData, type PolicyMapPoint } from "@/lib/api";
import { ACTION_LABELS, SCENARIO_LABELS } from "@/lib/policies";
import { useInView } from "./motion";
import { Legend, Segmented, TechDetails, Takeaway } from "./ui";

type Mode = "action" | "scenario" | "agreement";

// Fixed categorical order (colour follows the category, never its rank or count).
const ACTION_ORDER = ["ADMIT_PRIMARY_PRIORITY", "HOLD", "STEP_DOWN", "ADMIT_ICU", "ADMIT_OFF_SERVICE", "ESCALATE", "ADMIT_PRIMARY_LONGEST_WAIT"];
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)", "var(--text-muted)"];
const AGREE_GROUPS: Record<string, string> = {
  same: "Same as the best rule", both_wait: "Same as the best rule",
  same_patient_other_ward: "Different placement", different_patient: "Different placement",
  agent_waits: "One waits, the other acts", rule_waits: "One waits, the other acts",
};
const AGREE_ORDER = ["Same as the best rule", "Different placement", "One waits, the other acts"];

function groupOf(p: PolicyMapPoint, mode: Mode): string {
  if (mode === "action") return p.action;
  if (mode === "scenario") return p.scenario;
  return AGREE_GROUPS[p.agreement] ?? p.agreement;
}

function labelOf(key: string, mode: Mode): string {
  if (mode === "action") return ACTION_LABELS[key] ?? key;
  if (mode === "scenario") return SCENARIO_LABELS[key] ?? key;
  return key;
}

function MapTip({ active, payload }: { active?: boolean; payload?: { payload: PolicyMapPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="fan-tip">
      <b>{ACTION_LABELS[p.action] ?? p.action}</b>
      <span className="muted">{SCENARIO_LABELS[p.scenario]} · day {p.day.toFixed(1)}</span>
      <span>{p.waiting} waiting · wards {Math.round(p.occ[0] * 100)}% / {Math.round(p.occ[1] * 100)}% · ICU {Math.round(p.occ[2] * 100)}%</span>
      <span className="muted">vote share {(p.confidence * 100).toFixed(0)}%</span>
    </div>
  );
}

export default function PolicyMap() {
  const [ref, seen] = useInView<HTMLDivElement>();
  const [data, setData] = useState<PolicyMapData | null>(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState<Mode>("action");

  useEffect(() => {
    if (!seen || data) return;
    api.policyMap().then(setData).catch((e) => setError(String(e)));
  }, [seen, data]);

  const groups = useMemo(() => {
    if (!data) return [];
    const order = mode === "action" ? ACTION_ORDER : mode === "scenario" ? ["A", "B", "C", "D", "E"] : AGREE_ORDER;
    return order.map((key, i) => ({
      key, color: SERIES[i % SERIES.length],
      points: data.points.filter((p) => groupOf(p, mode) === key),
    })).filter((g) => g.points.length > 0);
  }, [data, mode]);

  const ax = data?.axes ?? [];
  const axisName = (k: number) => ax[k]?.top_inputs.slice(0, 2).map((t) => t.label.toLowerCase()).join(", ") ?? "";

  return (
    <div ref={ref} className="pmap">
      <div className="row" style={{ justifyContent: "space-between", gap: 10 }}>
        <Segmented options={[{ value: "action", label: "Move chosen" }, { value: "scenario", label: "Situation" },
          { value: "agreement", label: "Versus the best rule" }]} value={mode} onChange={(v) => setMode(v as Mode)} />
        {data && <span className="muted" style={{ fontSize: 13 }}>{data.points.length.toLocaleString()} decisions shown</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {!data && !error && <div className="skel skel-chart" style={{ height: 380, marginTop: 12 }} aria-hidden="true" />}
      {data && (
        <>
          <Legend items={groups.map((g) => ({ label: `${labelOf(g.key, mode)} (${g.points.length})`, color: g.color }))} />
          <div className="pmap-chart" key={mode}>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 10, right: 16, bottom: 30, left: 6 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" stroke="var(--axis)" tick={false} tickLine={false}
                  label={{ value: `more crowded →  (${axisName(0)})`, position: "insideBottom", offset: -18, fill: "var(--text-muted)", fontSize: 11.5 }} />
                <YAxis type="number" dataKey="y" stroke="var(--axis)" tick={false} tickLine={false} width={22}
                  label={{ value: "second pattern (weaker)", angle: -90, position: "insideLeft", offset: 12, fill: "var(--text-muted)", fontSize: 11.5 }} />
                <Tooltip content={<MapTip />} cursor={{ stroke: "var(--axis)", strokeDasharray: "3 3" }} />
                {groups.map((g) => (
                  <Scatter key={g.key} data={g.points} fill={g.color} fillOpacity={0.62} shape="circle" isAnimationActive={false} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <Takeaway tone="good">
            <b>The network sorts situations by how crowded the hospital is</b> (left to right), and its &quot;wait and keep beds free&quot; choices
            sit on the crowded side. Nobody programmed that grouping; it is how the trained network represents the hospital.
          </Takeaway>
          <TechDetails>
            <p>{data.n_decisions_with_choice.toLocaleString()} decisions with at least two valid moves, from runs {data.seeds.join(", ")} (training seeds; no
              validation or test run used) in all five situations; {data.points.length.toLocaleString()} sampled at random. Position = first two principal
              components of the 64 activations of the last hidden layer of network 0 (horizontal {(100 * (ax[0]?.explained_variance ?? 0)).toFixed(0)}% of the
              variance, vertical {(100 * (ax[1]?.explained_variance ?? 0)).toFixed(0)}%). Axis names are the inputs most correlated with each component
              (horizontal: {ax[0]?.top_inputs.map((t) => `${t.label} ${t.correlation.toFixed(2)}`).join("; ")}; vertical, weaker: {ax[1]?.top_inputs.map((t) => `${t.label} ${t.correlation.toFixed(2)}`).join("; ")}). Moves and vote shares are from the
              five-network vote. Descriptive only, not a performance result.</p>
          </TechDetails>
        </>
      )}
    </div>
  );
}
