"use client";

import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend as RLegend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Card, Legend, Segmented, useViewToggle } from "@/components/ui";
import { api, type StudyResults } from "@/lib/api";
import { METRICS, POLICIES, SCENARIO_LABELS, policyMeta } from "@/lib/policies";

const SCENARIOS = ["A", "B", "C", "D", "E"];
const BASELINES = ["load_balancing", "priority", "trunk_reservation", "mip", "fcfs"];
const axis = { stroke: "var(--axis)", tick: { fill: "var(--text-muted)", fontSize: 11 }, tickLine: false } as const;
const tip = {
  contentStyle: { background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 },
  cursor: { fill: "var(--surface-2)" },
};

/** Diverging colour for an effect: negative (PPO better) -> blue, positive -> red, ~0 -> gray. */
function divergeColor(value: number, max: number): string {
  const t = Math.max(-1, Math.min(1, value / max));
  const pole = t < 0 ? "var(--diverge-neg)" : "var(--diverge-pos)";
  const pct = Math.round(Math.abs(t) * 85);
  return `color-mix(in srgb, ${pole} ${pct}%, var(--diverge-mid))`;
}

export default function ExperimentsPage() {
  const [data, setData] = useState<StudyResults | null>(null);
  const [error, setError] = useState("");
  const [metric, setMetric] = useState("J_per_day");
  const [barTable, barToggle] = useViewToggle();
  const [heatTable, heatToggle] = useViewToggle();

  useEffect(() => {
    api.results().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data?.summary) return <p className="muted">Loading study results…</p>;

  const meta = METRICS.find((m) => m.key === metric)!;
  const value = (s: string, p: string): [number, number, number] | null => {
    const e = data.summary!.find((x) => x.scenario === s && x.policy === p)?.[metric];
    return Array.isArray(e) ? e : null;
  };
  const barData = SCENARIOS.map((s) => {
    const row: Record<string, number | string | [number, number]> = { scenario: s };
    for (const p of POLICIES) {
      const v = value(s, p.key);
      if (v) { row[p.key] = v[0]; row[`${p.key}_err`] = [v[0] - v[1], v[2] - v[1] - (v[0] - v[1])]; }
    }
    return row;
  });

  // Heatmap cells: PPO - baseline.
  type HeatCell = { diff: number; lo: number; hi: number; sig: boolean; effect: number; note: string };
  const heat: Record<string, Record<string, HeatCell>> = {};
  for (const b of BASELINES) {
    heat[b] = {};
    for (const s of SCENARIOS) {
      if (metric === "J_per_day") {
        const c = data.J_comparisons?.find((x) => x.scenario === s && x.baseline === b);
        if (c) heat[b][s] = { diff: c.diff, lo: c.ci_low, hi: c.ci_high, sig: c.ci_low > 0 || c.ci_high < 0,
          effect: c.rel_diff_pct / 20, note: `${c.rel_diff_pct > 0 ? "+" : ""}${c.rel_diff_pct.toFixed(1)}%` };
      } else {
        const c = data.comparisons?.find((x) => x.scenario === s && x.baseline === b && x.metric === metric);
        if (c) heat[b][s] = { diff: c.diff_mean, lo: c.ci_low, hi: c.ci_high, sig: c.significant_holm,
          effect: c.d_z, note: `d_z ${c.d_z.toFixed(2)}` };
      }
    }
  }

  const ri = Object.entries(data.robustness?.robustness_index ?? {})
    .map(([p, v]) => ({ policy: p, label: policyMeta(p).label, value: v.estimate,
      err: [v.estimate - v.ci[0], v.ci[1] - v.estimate] as [number, number] }))
    .sort((a, b) => a.value - b.value);
  const crn = (data.crn_check ?? []).filter((c) => c.metric === "J_per_day")
    .map((c) => ({ label: policyMeta(c.baseline).label, policy: c.baseline, ratio: c.width_ratio_independent_over_paired,
      reduction: c.variance_reduction }));
  const structure = data.structure
    ? data.structure.A.bins.map((b, i) => ({
      free: String(b.free_beds), A: +b.mean_p_hold.toFixed(3), E: +(data.structure!.E?.bins[i]?.mean_p_hold ?? NaN).toFixed(3),
    }))
    : [];

  return (
    <div className="stack">
      <section className="hero" style={{ padding: "26px 28px" }}>
        <span className="eyebrow">Pre-registered evaluation · 50 test seeds per scenario · 2,500 episodes</span>
        <h1 style={{ fontSize: "clamp(22px, 2.6vw, 32px)" }}>Study results</h1>
        <p style={{ marginBottom: 12 }}>
          PPO (5 independently trained agents, trained on scenario A only) against five tuned baselines in the
          training scenario and four stress scenarios it never saw. Lower is better for every metric.
          {" "}{data.label}
        </p>
        <Segmented options={METRICS.map((m) => ({ value: m.key, label: m.label }))} value={metric} onChange={setMetric} />
      </section>

      <Card title={`${meta.label} by scenario and policy`} subtitle={`mean over 50 test seeds with 95% CI · ${meta.unit}`} actions={barToggle}>
        <Legend items={POLICIES.map((p) => ({ label: p.label, color: p.color }))} />
        {barTable ? (
          <div className="scroll" style={{ marginTop: 10 }}>
            <table>
              <thead><tr><th>scenario</th>{POLICIES.map((p) => <th key={p.key}>{p.label}</th>)}</tr></thead>
              <tbody>{SCENARIOS.map((s) => (
                <tr key={s}><td>{SCENARIO_LABELS[s]}</td>{POLICIES.map((p) => {
                  const v = value(s, p.key);
                  return <td key={p.key}>{v ? `${v[0].toFixed(meta.digits)} ± ${((v[2] - v[1]) / 2).toFixed(meta.digits)}` : "–"}</td>;
                })}</tr>))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ width: "100%", height: 340, marginTop: 8 }}>
            <ResponsiveContainer>
              <BarChart data={barData} margin={{ top: 10, right: 12, bottom: 4, left: 0 }} barGap={2} barCategoryGap="18%">
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="scenario" {...axis} tickFormatter={(s) => SCENARIO_LABELS[s]?.split(" · ")[1] ?? s} />
                <YAxis {...axis} />
                <Tooltip {...tip} formatter={(v, name) => [typeof v === "number" ? v.toFixed(meta.digits) : "–", policyMeta(String(name)).label]} />
                {POLICIES.map((p) => (
                  <Bar key={p.key} dataKey={p.key} fill={p.color} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey={`${p.key}_err`} width={3} strokeWidth={1.2} stroke="var(--text-secondary)" />
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      <Card title="Where PPO wins and loses"
        subtitle={metric === "J_per_day"
          ? "PPO − baseline cost (relative %). ● = 95% CI excludes 0 (descriptive, not multiplicity-corrected)."
          : "PPO − baseline, effect size d_z. ● = significant after Holm correction over 100 comparisons."}
        actions={heatToggle}>
        {heatTable ? (
          <div className="scroll">
            <table>
              <thead><tr><th>PPO vs</th>{SCENARIOS.map((s) => <th key={s}>{s}</th>)}</tr></thead>
              <tbody>{BASELINES.map((b) => (
                <tr key={b}><td>{policyMeta(b).label}</td>{SCENARIOS.map((s) => {
                  const c = heat[b][s];
                  return <td key={s}>{c ? `${c.diff.toFixed(meta.digits)} [${c.lo.toFixed(meta.digits)}, ${c.hi.toFixed(meta.digits)}]${c.sig ? " *" : ""}` : "–"}</td>;
                })}</tr>))}
              </tbody>
            </table>
          </div>
        ) : (
          <>
            <div className="heat scroll" style={{ gridTemplateColumns: `170px repeat(${SCENARIOS.length}, minmax(96px, 1fr))` }}>
              <div />
              {SCENARIOS.map((s) => <div key={s} className="hdr">{SCENARIO_LABELS[s].split(" · ")[1]}</div>)}
              {BASELINES.map((b) => (
                <div key={b} style={{ display: "contents" }}>
                  <div className="rowhdr"><i className="dot" style={{ background: policyMeta(b).color, marginRight: 6 }} />{policyMeta(b).label}</div>
                  {SCENARIOS.map((s) => {
                    const c = heat[b][s];
                    return (
                      <div key={s} className="cell" style={{ background: c ? divergeColor(c.effect, 1.5) : "var(--surface-2)" }}
                        title={c ? `diff ${c.diff.toFixed(meta.digits)} [${c.lo.toFixed(meta.digits)}, ${c.hi.toFixed(meta.digits)}]` : ""}>
                        <div style={{ fontWeight: 650 }}>{c ? `${c.diff > 0 ? "+" : ""}${c.diff.toFixed(meta.digits)}` : "–"}{c?.sig ? " ●" : ""}</div>
                        <div style={{ fontSize: 11, opacity: 0.85 }}>{c?.note}</div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="legend" style={{ marginTop: 10 }}>
              <span><i className="swatch" style={{ background: divergeColor(-1.5, 1.5) }} />PPO better</span>
              <span><i className="swatch" style={{ background: "var(--diverge-mid)", border: "1px solid var(--grid)" }} />no difference</span>
              <span><i className="swatch" style={{ background: divergeColor(1.5, 1.5) }} />PPO worse</span>
              <span>● significant</span>
            </div>
          </>
        )}
      </Card>

      <div className="grid grid-2">
        <Card title="Robustness index" subtitle="mean % degradation from A to stress scenarios B–E (lower = more robust), 95% CI">
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={ri} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 8 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" unit="%" {...axis} />
                <YAxis type="category" dataKey="label" width={150} {...axis} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
                <Tooltip {...tip} formatter={(v) => [typeof v === "number" ? `${v.toFixed(0)}%` : "–", "degradation"]} />
                <Bar dataKey="value" barSize={16} radius={4} isAnimationActive={false}>
                  {ri.map((r) => <Cell key={r.policy} fill={policyMeta(r.policy).color} />)}
                  <ErrorBar dataKey="err" width={5} strokeWidth={1.2} stroke="var(--text-secondary)" direction="x" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="muted" style={{ margin: 0 }}>Relative degradation favours policies that are already poor in A; read with the absolute results above.</p>
        </Card>

        <Card title="Why common random numbers matter" subtitle="how many times narrower the CI of PPO − baseline cost is with paired seeds">
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={crn} layout="vertical" margin={{ top: 4, right: 30, bottom: 4, left: 8 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" {...axis} tickFormatter={(v) => `${v}×`} />
                <YAxis type="category" dataKey="label" width={150} {...axis} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
                <Tooltip {...tip} formatter={(v, _n, item) => [`${typeof v === "number" ? v.toFixed(1) : "–"}× narrower · variance −${(100 * (item.payload as { reduction: number }).reduction).toFixed(1)}%`, "CRN"]} />
                <Bar dataKey="ratio" barSize={16} radius={4} isAnimationActive={false} label={{ position: "right", fill: "var(--text-secondary)", fontSize: 11, formatter: (v: unknown) => `${Number(v).toFixed(1)}×` }}>
                  {crn.map((r) => <Cell key={r.policy} fill={policyMeta(r.policy).color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="muted" style={{ margin: 0 }}>Without CRN, PPO&apos;s 9% advantage over Priority would be undetectable.</p>
        </Card>
      </div>

      {structure.length > 0 && (
        <Card title="Did PPO learn to reserve beds?"
          subtitle="probability PPO holds a free bed when only low-priority patients wait, by number of free beds">
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={structure} margin={{ top: 10, right: 16, bottom: 4, left: -8 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="free" {...axis} label={{ value: "free staffed beds", position: "insideBottom", offset: -2, fill: "var(--text-muted)", fontSize: 11 }} height={36} />
                <YAxis {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip {...tip} cursor={{ stroke: "var(--axis)" }} formatter={(v, n) => [typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "–", n === "A" ? "normal load (A)" : "combined stress (E)"]} />
                <RLegend verticalAlign="top" height={28} formatter={(v) => (v === "A" ? "A · normal load (training)" : "E · combined stress (unseen)")} wrapperStyle={{ fontSize: 12 }} />
                <Line dataKey="A" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                <Line dataKey="E" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                <ReferenceLine y={0} stroke="var(--axis)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="muted" style={{ margin: 0 }}>
            Normal load: holding falls as free beds rise (Spearman ρ = {data.structure?.A.spearman_rho.toFixed(2)}),
            a soft trunk-reservation policy learned without being told. Under unseen stress the structure
            disappears (ρ = {data.structure?.E.spearman_rho.toFixed(2)}).
          </p>
        </Card>
      )}
    </div>
  );
}
