"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend as RLegend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Icon, type IconName } from "@/components/icons";
import { CountUp, Reveal, Scene, SplitWords, WhenVisible } from "@/components/motion";
import ReproduceCard from "@/components/ReproduceCard";
import { Card, Legend, SectionHeader, Segmented, Skeleton, Takeaway, TechDetails, useViewToggle } from "@/components/ui";
import { api, type Comparison, type StudyResults } from "@/lib/api";
import { ENSEMBLE, V3, METRICS, POLICIES, SCENARIO_INFO, SCENARIO_LABELS, policyMeta, scenNum, type PolicyMeta } from "@/lib/policies";

// Study results as a guided story: each chapter asks one plain question, shows one chart, states
// the takeaway, and keeps the full statistics in a collapsible "Technical detail". All values are
// read from /api/experiments/results (docs/data/stage6 and stage7); nothing is computed or
// smoothed here beyond formatting and simple ratios.

const SCENARIOS = ["A", "B", "C", "D", "E"];
const BASELINES = ["load_balancing", "priority", "trunk_reservation", "mip", "fcfs"];
const axis = { stroke: "var(--axis)", tick: { fill: "var(--text-muted)", fontSize: 11 }, tickLine: false } as const;
const tip = {
  contentStyle: { background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12, color: "var(--text-primary)" },
  labelStyle: { color: "var(--text-muted)", fontWeight: 600 },
  cursor: { fill: "var(--surface-2)" },
};
/** Single-series charts: no series colour to inherit, so use the text colour. */
const tipOne = { ...tip, itemStyle: { color: "var(--text-primary)" } };
/** Tooltip rows sorted by value: ascending (best first) where lower is better, else descending. */
const ascending = (item: { value?: unknown }) => Number(item.value ?? 0);
const descending = (item: { value?: unknown }) => -Number(item.value ?? 0);
const scenShort = (s: string) => SCENARIO_INFO[s]?.short ?? s;

const ABLATION_NAMES: Record<string, string> = {
  B_no_demand_history: "No recent-demand info",
  C_no_temporal: "No time of day / week",
  D_no_staffing: "No staffing info",
  E_no_priority_aggregates: "No urgency summary",
  F1_no_diversion_penalty: "No penalty for turning ambulances away",
  F2_no_off_service_penalty: "No penalty for wrong-ward placement",
  F3_uniform_priority_weights: "All patients weighted equally",
  G_no_icu_flow_state: "No ICU flow info",
};

const CHAPTERS = [
  { id: "overall", label: "1 · Who does best?" },
  { id: "headtohead", label: "2 · Agent vs each rule" },
  { id: "crisis", label: "3 · Coping with crises" },
  { id: "clever", label: "4 · Did it learn something?" },
  { id: "ingredients", label: "5 · What makes it work?" },
  { id: "tougher", label: "6 · Making it tougher" },
  { id: "beat", label: "7 · Beating the best rule" },
  { id: "trust", label: "8 · Why trust this?" },
  { id: "glossary", label: "Glossary" },
];

/** Diverging colour for an effect: negative (Agent better) -> blue, positive -> red, ~0 -> gray. */
function divergeColor(value: number, max: number): string {
  const t = Math.max(-1, Math.min(1, value / max));
  const pole = t < 0 ? "var(--diverge-neg)" : "var(--diverge-pos)";
  return `color-mix(in srgb, ${pole} ${Math.round(Math.abs(t) * 85)}%, var(--diverge-mid))`;
}

/** Highlights the chapter currently on screen. Uses layout offsets (not on-screen rects), so the
 *  sections' scroll animations (which move them while they fade) cannot confuse it. */
function useActiveChapter(): string {
  const [active, setActive] = useState(CHAPTERS[0].id);
  useEffect(() => {
    const top = (el: HTMLElement) => { let y = 0; let e: HTMLElement | null = el; while (e) { y += e.offsetTop; e = e.offsetParent as HTMLElement | null; } return y; };
    const onScroll = () => {
      const line = window.scrollY + window.innerHeight * 0.3;
      let cur = CHAPTERS[0].id;
      for (const c of CHAPTERS) { const el = document.getElementById(c.id); if (el && top(el) <= line) cur = c.id; }
      setActive(cur);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return active;
}

/** Horizontal bars of differences with 95% CIs (positive = worse). */
function DiffBars({ rows, unit, height, digits = 1 }: {
  rows: { label: string; diff: number; lo: number; hi: number; sig: boolean; color?: string }[];
  unit: string; height: number; digits?: number;
}) {
  const data = rows.map((r) => ({ ...r, err: [r.diff - r.lo, r.hi - r.diff] as [number, number] }));
  return (
    <WhenVisible height={height}>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 28, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" {...axis} />
            <YAxis type="category" dataKey="label" width={210} {...axis} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
            <ReferenceLine x={0} stroke="var(--text-muted)" />
            <Tooltip {...tipOne} formatter={(v, _n, item) => {
              const p = item.payload as { lo: number; hi: number; sig: boolean };
              const val = typeof v === "number" ? v.toFixed(digits) : "–";
              return [`${val} ${unit}  (95% CI ${p.lo.toFixed(digits)} to ${p.hi.toFixed(digits)})${p.sig ? " · significant" : ""}`, "difference"];
            }} />
            <Bar dataKey="diff" barSize={16} radius={4} animationDuration={900}>
              {data.map((r) => <Cell key={r.label} fill={r.color ?? (r.diff > 0 ? "var(--diverge-pos)" : "var(--diverge-neg)")}
                fillOpacity={r.sig ? 1 : 0.45} />)}
              <ErrorBar dataKey="err" width={5} strokeWidth={1.2} stroke="var(--text-secondary)" direction="x" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </WhenVisible>
  );
}

export default function ExperimentsPage() {
  const [data, setData] = useState<StudyResults | null>(null);
  const [error, setError] = useState("");
  const [metric, setMetric] = useState("J_per_day");
  const [version, setVersion] = useState<"v3" | "v2" | "v1">("v3");
  const [barTable, barToggle] = useViewToggle();
  const [heatTable, heatToggle] = useViewToggle();
  const active = useActiveChapter();

  useEffect(() => {
    api.results().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return (
    <div className="card" style={{ marginTop: 24 }}>
      <h2 className="card-title">The results server is waking up</h2>
      <p className="secondary">The free server sleeps when idle and takes about a minute to start. Refresh shortly.</p>
      <p className="muted">{error}</p>
    </div>
  );
  if (!data?.summary) return (
    <div style={{ marginTop: 24 }}>
      <p className="muted">Loading study results… (the server may take up to a minute to wake up)</p>
      <Skeleton lines={3} chart={300} />
    </div>
  );

  const meta = METRICS.find((m) => m.key === metric)!;
  // The final agent (v2) is compared with every rule on the fresh block (RESEARCH.md v2.9); the
  // development version (v1) on the original Stage 6 test block. Same analysis for both.
  const ver = version === "v3" && data.v3_summary ? "v3" : version !== "v1" && data.final_summary ? "v2" : "v1";
  const V2 = ver !== "v1"; // a final-generation agent (v2 or v3) rather than the development version
  const src = ver === "v3"
    ? { summary: data.v3_summary, J: data.v3_J_comparisons, comps: data.v3_comparisons, rob: data.v3_robustness }
    : ver === "v2"
      ? { summary: data.final_summary, J: data.final_J_comparisons, comps: data.final_comparisons, rob: data.final_robustness }
      : { summary: data.summary, J: data.J_comparisons, comps: data.comparisons, rob: data.robustness };
  const pm = (k: string): PolicyMeta => (k === "ppo"
    ? (ver === "v3" ? { ...V3, key: "ppo", label: "RL agent v3 (latest)" } : ver === "v2" ? { ...ENSEMBLE, key: "ppo", label: "RL agent v2" } : { ...policyMeta("ppo"), label: "RL agent v1 (development)" })
    : policyMeta(k));
  const STUDY = POLICIES.map((x) => pm(x.key));
  const agentName = ver === "v3" ? "the latest agent (v3)" : ver === "v2" ? "agent v2" : "the development agent (v1)";
  const value = (s: string, p: string): [number, number, number] | null => {
    const e = src.summary!.find((x) => x.scenario === s && x.policy === p)?.[metric];
    return Array.isArray(e) ? e : null;
  };
  const barData = SCENARIOS.map((s) => {
    const row: Record<string, number | string | [number, number]> = { scenario: s };
    for (const p of STUDY) {
      const v = value(s, p.key);
      if (v) { row[p.key] = v[0]; row[`${p.key}_err`] = [v[0] - v[1], v[2] - v[0]]; }
    }
    return row;
  });

  // Head-to-head cells: AI - rule.
  type HeatCell = { diff: number; lo: number; hi: number; sig: boolean; effect: number; note: string };
  const heat: Record<string, Record<string, HeatCell>> = {};
  for (const b of BASELINES) {
    heat[b] = {};
    for (const s of SCENARIOS) {
      if (metric === "J_per_day") {
        const c = src.J?.find((x) => x.scenario === s && x.baseline === b);
        if (c) heat[b][s] = { diff: c.diff, lo: c.ci_low, hi: c.ci_high, sig: c.ci_low > 0 || c.ci_high < 0,
          effect: c.rel_diff_pct / 20, note: `${c.rel_diff_pct > 0 ? "+" : ""}${c.rel_diff_pct.toFixed(1)}%` };
      } else {
        const c = src.comps?.find((x) => x.scenario === s && x.baseline === b && x.metric === metric);
        if (c) heat[b][s] = { diff: c.diff_mean, lo: c.ci_low, hi: c.ci_high, sig: c.significant_holm,
          effect: c.d_z, note: `d_z ${c.d_z.toFixed(2)}` };
      }
    }
  }
  // Keep-a-bed-free is identical to most-urgent-first (tuned reserve 0), so it is not counted twice.
  const matchups = Object.entries(heat).filter(([b]) => b !== "trunk_reservation").flatMap(([, r]) => Object.values(r));
  const wins = matchups.filter((c) => c.sig && c.diff < 0).length;
  const losses = matchups.filter((c) => c.sig && c.diff > 0).length;

  const ri = Object.entries(src.rob?.robustness_index ?? {})
    .map(([p, v]) => ({ policy: p, label: pm(p).label, value: v.estimate,
      err: [v.estimate - v.ci[0], v.ci[1] - v.estimate] as [number, number] }))
    .sort((a, b) => a.value - b.value);
  const riDiff = (src.rob as { ppo_minus_baseline_RI?: Record<string, { estimate: number; ci: [number, number] }> } | null)?.ppo_minus_baseline_RI ?? {};
  const steadier = Object.entries(riDiff).filter(([b, v]) => b !== "trunk_reservation" && v.ci[1] < 0).map(([b]) => pm(b).label.toLowerCase());
  const shakier = Object.entries(riDiff).filter(([b, v]) => b !== "trunk_reservation" && v.ci[0] > 0).map(([b]) => pm(b).label.toLowerCase());
  const robustSummary = `${agentName[0].toUpperCase()}${agentName.slice(1)} deteriorates significantly less than ${steadier.length ? steadier.join(", ") : "no rule"}${shakier.length ? `, and more than ${shakier.join(", ")}` : ""}; elsewhere the difference is within the uncertainty.`;
  const crn = (data.crn_check ?? []).filter((c) => c.metric === "J_per_day" && c.baseline !== "trunk_reservation")
    .map((c) => ({ label: pm(c.baseline).label, policy: c.baseline, ratio: c.width_ratio_independent_over_paired,
      reduction: c.variance_reduction }));
  const crnMax = Math.max(...crn.map((c) => c.ratio));
  const structure = data.structure
    ? data.structure.A.bins.map((b, i) => ({
      free: String(b.free_beds), A: +b.mean_p_hold.toFixed(3), E: +(data.structure!.E?.bins[i]?.mean_p_hold ?? NaN).toFixed(3),
    }))
    : [];

  const toRow = (c: Comparison, label: string, scale = 1) => ({
    label, diff: c.diff_mean * scale, lo: c.ci_low * scale, hi: c.ci_high * scale, sig: c.significant_holm,
  });
  const ablA = (data.ablations ?? []).filter((c) => c.scenario === "A" && c.metric === "J_per_day")
    .map((c) => {
      const key = c.baseline.replace("full_vs_", "");
      return toRow(c, ABLATION_NAMES[key] ?? key, 100 / c.baseline_mean);
    })
    .sort((a, b) => b.diff - a.diff);
  const dr = (data.ppo_dr ?? []).filter((c) => c.metric === "J_per_day")
    .map((c) => toRow(c, `${scenNum(c.scenario)} · ${scenShort(c.scenario)}`, 100 / c.baseline_mean));
  const budget = data.constrained?.budget ?? {};
  const lam = SCENARIOS.map((s) => ({
    scenario: `${scenNum(s)} · ${scenShort(s)}`,
    lambda: budget[`${s}:lambda_mean`],
    ppo: 100 * (budget[`${s}:ppo`] ?? NaN),
    constrained: 100 * (budget[`${s}:ppo_lagrangian`] ?? NaN),
  }));

  return (
    <div>
      {/* ------------------------------------------------------------------ intro */}
      <section className="hero2" style={{ marginBottom: 8 }}>
        <span className="blob b1" /><span className="blob b2" />
        <Reveal><span className="pill"><span className="badge">Results</span> {ver === "v3" ? "latest agent (v3) · 50 brand-new runs per situation" : ver === "v2" ? "agent v2 · 50 fresh runs per situation" : "development agent (v1) · 50 sealed test runs per situation"}</span></Reveal>
        <Reveal delay={80}><h1 style={{ fontSize: "clamp(30px, 4.4vw, 52px)" }}><SplitWords text="What happened when the agent" /> <SplitWords text="met the rules" delay={300} wordClass="gradient-text" /></h1></Reveal>
        <Reveal delay={160}><p className="lead">
          The study&apos;s fixed results, one question per chapter. Lower is better everywhere; full statistics sit under
          &quot;Technical detail&quot;. For a live experiment, try <Link href="/race/">the Race</Link>.</p></Reveal>
        <Reveal delay={200}>
          <div className="row" style={{ marginBottom: 4 }}>
            <Segmented options={[{ value: "v3", label: "Latest agent (v3)" }, { value: "v2", label: "v2" }, { value: "v1", label: "Development (v1)" }]}
              value={version} onChange={(v) => setVersion(v as "v3" | "v2" | "v1")} />
            <span className="muted">{ver === "v3" ? "The comparisons below show the latest agent against every rule." : ver === "v2" ? "The comparisons below show agent v2, before it practised crises." : "The comparisons below show the first agent, where it all started."}</span>
          </div>
        </Reveal>
        <Reveal delay={240}>
          <div className="ribbon">
            <div><div className="num"><CountUp value={wins} /></div><div className="lbl">confident agent wins out of {matchups.length} match-ups ({meta.label.toLowerCase()})</div></div>
            <div><div className="num"><CountUp value={losses} /></div><div className="lbl">confident agent losses</div></div>
            <div><div className="num"><CountUp value={crnMax} digits={1} suffix="×" /></div><div className="lbl">sharper comparisons (same patients)</div></div>
            <div><div className="num"><CountUp value={(data.ablations?.length ?? 0) + (data.ppo_dr?.length ?? 0) + (data.constrained?.comparisons.length ?? 0) + (data.comparisons?.length ?? 0)} /></div><div className="lbl">pre-planned statistical tests</div></div>
          </div>
        </Reveal>
      </section>

      <nav className="chapters" aria-label="Chapters">
        {CHAPTERS.map((c) => <a key={c.id} href={`#${c.id}`} className={active === c.id ? "on" : ""}>{c.label}</a>)}
      </nav>

      {/* ------------------------------------------------------------------ 1 overall */}
      <Scene id="overall">
        <Reveal><SectionHeader num="1" kicker="Overall" title="Who does best in each situation?"
          lead="One group per situation, one bar per decision-maker." /></Reveal>
        <Reveal>
          <div className="row" style={{ justifyContent: "center", marginBottom: 16 }}>
            <Segmented options={METRICS.map((m) => ({ value: m.key, label: m.label }))} value={metric} onChange={setMetric} />
          </div>
        </Reveal>
        <Reveal>
          <Card title={meta.label} subtitle={`${meta.unit} · average of 50 test runs · thin whiskers show the uncertainty range`} actions={barToggle}
            explain={meta.desc}>
            <Legend items={STUDY.map((p) => ({ label: p.label, color: p.color }))} />
            {barTable ? (
              <div className="scroll" style={{ marginTop: 10 }}>
                <table>
                  <thead><tr><th>situation</th>{STUDY.map((p) => <th key={p.key}>{p.label}</th>)}</tr></thead>
                  <tbody>{SCENARIOS.map((s) => (
                    <tr key={s}><td>{SCENARIO_LABELS[s]}</td>{STUDY.map((p) => {
                      const v = value(s, p.key);
                      return <td key={p.key}>{v ? `${v[0].toFixed(meta.digits)} ± ${((v[2] - v[1]) / 2).toFixed(meta.digits)}` : "–"}</td>;
                    })}</tr>))}
                  </tbody>
                </table>
              </div>
            ) : (
              <WhenVisible height={350}>
                <div style={{ width: "100%", height: 350, marginTop: 8 }}>
                  <ResponsiveContainer>
                    <BarChart key={metric} data={barData} margin={{ top: 10, right: 12, bottom: 4, left: 0 }} barGap={2} barCategoryGap="18%">
                      <CartesianGrid stroke="var(--grid)" vertical={false} />
                      <XAxis dataKey="scenario" {...axis} tickFormatter={(s) => scenShort(s)} />
                      <YAxis {...axis} />
                      <Tooltip {...tip} itemSorter={ascending} labelFormatter={(s) => `${SCENARIO_LABELS[String(s)] ?? String(s)} (best first)`}
                        formatter={(v, name) => [typeof v === "number" ? `${v.toFixed(meta.digits)} ${meta.unit}` : "–", pm(String(name)).label]} />
                      {STUDY.map((p, i) => (
                        <Bar key={p.key} dataKey={p.key} fill={p.color} radius={[4, 4, 0, 0]} animationDuration={900} animationBegin={i * 80}>
                          <ErrorBar dataKey={`${p.key}_err`} width={3} strokeWidth={1.2} stroke="var(--text-secondary)" />
                        </Bar>
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </WhenVisible>
            )}
            <Takeaway>
              Crises make everything worse for everyone, and the hardest one (&quot;everything at once&quot;) multiplies costs several
              times over. Within each situation, the <b>RL agent</b> and the <b>balance-the-wards rule</b> are consistently among the best,
              and <b>first-come-first-served</b> is clearly the worst.
            </Takeaway>
            <TechDetails>
              <p>Means over 50 test seeds (200000–200049); RL agent averaged over 5 independently trained agents. Whiskers: 95% t-intervals over seeds.
                Boarding is censoring-inclusive (patients still waiting at the horizon count with their wait so far), a disclosed protocol amendment.</p>
            </TechDetails>
          </Card>
        </Reveal>
      </Scene>

      {/* ------------------------------------------------------------------ 2 head to head */}
      <Scene id="headtohead">
        <Reveal><SectionHeader num="2" kicker="Head to head" title="Where does the agent beat each rule, and where does it lose?"
          lead={<>Each square compares the agent with one rule in one situation. <b style={{ color: "var(--diverge-neg)" }}>Blue</b> = Agent better,{" "}
            <b style={{ color: "var(--diverge-pos)" }}>red</b> = Agent worse, grey = no real difference. A dot ● marks a difference we are statistically confident about.</>} /></Reveal>
        <Reveal>
          <Card title={`Agent minus rule: ${meta.label.toLowerCase()}`} subtitle={meta.unit} actions={heatToggle}
            explain={`Numbers show how much higher (+) or lower (−) the agent's ${meta.label.toLowerCase()} is than each rule's. Negative is good for the agent.`}>
            {heatTable ? (
              <div className="scroll">
                <table>
                  <thead><tr><th>Agent vs</th>{SCENARIOS.map((s) => <th key={s}>{scenShort(s)}</th>)}</tr></thead>
                  <tbody>{BASELINES.map((b) => (
                    <tr key={b}><td>{pm(b).label}</td>{SCENARIOS.map((s) => {
                      const c = heat[b][s];
                      return <td key={s}>{c ? `${c.diff.toFixed(meta.digits)} [${c.lo.toFixed(meta.digits)}, ${c.hi.toFixed(meta.digits)}]${c.sig ? " *" : ""}` : "–"}</td>;
                    })}</tr>))}
                  </tbody>
                </table>
              </div>
            ) : (
              <>
                <div className="heat scroll" style={{ gridTemplateColumns: `200px repeat(${SCENARIOS.length}, minmax(96px, 1fr))` }}>
                  <div />
                  {SCENARIOS.map((s) => <div key={s} className="hdr"><Icon name={SCENARIO_INFO[s].icon} size={14} /> {scenNum(s)} · {scenShort(s)}</div>)}
                  {BASELINES.map((b, bi) => (
                    <div key={b} style={{ display: "contents" }}>
                      <div className="rowhdr"><i className="dot" style={{ background: pm(b).color, marginRight: 6 }} />{pm(b).label}</div>
                      {SCENARIOS.map((s, si) => {
                        const c = heat[b][s];
                        return (
                          <Reveal key={s} kind="zoom" delay={(bi * 5 + si) * 25}>
                            <div className="cell" style={{ background: c ? divergeColor(c.effect, 1.5) : "var(--surface-2)" }}
                              title={c ? `difference ${c.diff.toFixed(meta.digits)} (95% CI ${c.lo.toFixed(meta.digits)} to ${c.hi.toFixed(meta.digits)})` : ""}>
                              <div style={{ fontWeight: 650 }}>{c ? `${c.diff > 0 ? "+" : ""}${c.diff.toFixed(meta.digits)}` : "–"}{c?.sig ? " ●" : ""}</div>
                              <div style={{ fontSize: 11, opacity: 0.85 }}>{c?.note}</div>
                            </div>
                          </Reveal>
                        );
                      })}
                    </div>
                  ))}
                </div>
                <div className="legend" style={{ marginTop: 10 }}>
                  <span><i className="swatch" style={{ background: divergeColor(-1.5, 1.5) }} />Agent better</span>
                  <span><i className="swatch" style={{ background: "var(--diverge-mid)", border: "1px solid var(--grid)" }} />no real difference</span>
                  <span><i className="swatch" style={{ background: divergeColor(1.5, 1.5) }} />Agent worse</span>
                  <span>● confident difference</span>
                </div>
              </>
            )}
            <Takeaway tone="good">
              {V2
                ? <>The agent is blue against nearly every rule in nearly every situation, including the <b>balance-the-wards rule on normal days{ver === "v3" ? " and in a staff shortage" : ""}</b>. The exception is the combined crisis, where it is level with the two strongest rules.</>
                : <>Against the classic rules (most-urgent-first, the instant optimiser, first-come-first-served) the development agent is mostly blue. Against the <b>balance-the-wards rule</b> the squares are grey: a genuine tie, which is what version 2 set out to beat.</>}
            </Takeaway>
            <TechDetails>
              <p>{metric === "J_per_day"
                ? "Overall cost: cell = mean paired difference (agent − rule) with relative %; ● = 95% hierarchical-bootstrap CI excludes 0 (secondary, descriptive)."
                : "Cell = mean paired difference; second line = effect size d_z; ● = significant after Holm correction over all 100 pre-registered primary comparisons."}</p>
              <p>The keep-a-bed-free rule tuned itself to reserve zero beds, so it behaves exactly like most-urgent-first (identical numbers).</p>
            </TechDetails>
          </Card>
        </Reveal>
      </Scene>

      {/* ------------------------------------------------------------------ 3 crisis */}
      <Scene id="crisis">
        <Reveal><SectionHeader num="3" kicker="Robustness" title="How much worse does each one get in a crisis?"
          lead="How much worse each decision-maker gets in the four crises, relative to its own normal day." /></Reveal>
        <div className="grid grid-2">
          <Reveal kind="left">
            <Card title="Deterioration under crisis" subtitle="average % increase from normal day to crises (shorter bar = steadier)"
              explain="Each bar is one decision-maker. The whisker shows the uncertainty range.">
              <WhenVisible height={270}>
                <div style={{ width: "100%", height: 270 }}>
                  <ResponsiveContainer>
                    <BarChart data={ri} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 8 }}>
                      <CartesianGrid stroke="var(--grid)" horizontal={false} />
                      <XAxis type="number" unit="%" {...axis} />
                      <YAxis type="category" dataKey="label" width={170} {...axis} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
                      <Tooltip {...tipOne} formatter={(v) => [typeof v === "number" ? `${v.toFixed(0)}% worse` : "–", "deterioration"]} />
                      <Bar dataKey="value" barSize={16} radius={4} animationDuration={1000}>
                        {ri.map((r) => <Cell key={r.policy} fill={pm(r.policy).color} />)}
                        <ErrorBar dataKey="err" width={5} strokeWidth={1.2} stroke="var(--text-secondary)" direction="x" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </WhenVisible>
            </Card>
          </Reveal>
          <Reveal kind="right">
            <Card title="How to read this" subtitle="steady is not the same as good">
              <Takeaway tone="good">{robustSummary}</Takeaway>
              <Takeaway tone="warn">
                First-come-first-served looks &quot;steady&quot; only because it is already bad on a normal day: there is less room to get worse.
                Always read this together with the overall results in &quot;Who does best?&quot; at the top of this page.
              </Takeaway>
              <TechDetails>
                <p>Robustness index = mean relative degradation (crisis − normal day)/normal day over situations 2–5 and four metrics (cost, mean and 90th-percentile boarding,
                  off-service). Bootstrap over seeds jointly across scenarios and over agents. Diversion is excluded (≈0 on a normal day makes ratios meaningless) and
                  reported separately as an absolute change.</p>
              </TechDetails>
            </Card>
          </Reveal>
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ 4 clever */}
      {structure.length > 0 && (
        <Scene id="clever">
          <Reveal><SectionHeader num="4" kicker="Inside the agent" title="Did the agent discover something on its own?"
            lead="Hospitals sometimes keep a bed empty for the next emergency. Nobody told the agent about this idea. Did it find it anyway? (Analysed on the development version, v1, which learned purely by itself.)" /></Reveal>
          <Reveal>
            <Card title="How often the agent keeps a free bed empty"
              subtitle="when only non-urgent patients are waiting, by number of free beds"
              explain="Left = almost full, right = plenty of room. A falling line means: hold back when beds are scarce, fill freely when there is room.">
              <WhenVisible height={300}>
                <div style={{ width: "100%", height: 300 }}>
                  <ResponsiveContainer>
                    <LineChart data={structure} margin={{ top: 10, right: 16, bottom: 4, left: -8 }}>
                      <CartesianGrid stroke="var(--grid)" vertical={false} />
                      <XAxis dataKey="free" {...axis} label={{ value: "free beds in the ward", position: "insideBottom", offset: -2, fill: "var(--text-muted)", fontSize: 11 }} height={36} />
                      <YAxis {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip {...tip} itemSorter={descending} cursor={{ stroke: "var(--axis)" }} formatter={(v, n) => [typeof v === "number" ? `${(v * 100).toFixed(1)}% of the time` : "–", n === "A" ? "normal day" : "everything at once"]} />
                      <RLegend verticalAlign="top" height={28} formatter={(v) => (v === "A" ? "Normal day (trained on)" : "Everything at once (never seen)")} wrapperStyle={{ fontSize: 12 }} />
                      <Line dataKey="A" stroke="var(--series-1)" strokeWidth={2.5} dot={{ r: 4 }} animationDuration={1400} />
                      <Line dataKey="E" stroke="var(--series-2)" strokeWidth={2.5} dot={{ r: 4 }} animationDuration={1400} animationBegin={300} />
                      <ReferenceLine y={0} stroke="var(--axis)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </WhenVisible>
              <Takeaway tone="good">
                <b>Yes, on normal days.</b> The blue line falls: the agent learned a soft &quot;keep a bed free&quot; strategy, the shape queueing theory predicts.
              </Takeaway>
              <Takeaway tone="warn">
                <b>But not in the crisis it never saw.</b> The orange line is flat and high: it holds beds about a third of the time no matter what.
                This helps explain its weaker behaviour under extreme stress.
              </Takeaway>
              <TechDetails>
                <p>Normal day: Spearman ρ = {data.structure?.A.spearman_rho.toFixed(2)} ({data.structure?.A.n_states.toLocaleString()} states), isotonic fit R² = 0.98.
                  Combined stress: ρ = {data.structure?.E.spearman_rho.toFixed(2)}, R² = 0.01. All 5 agents pooled over 20 test seeds. The tuned keep-a-bed-free rule reserves 0 beds,
                  so this behaviour is not present in any human-designed baseline.</p>
              </TechDetails>
            </Card>
          </Reveal>
        </Scene>
      )}

      {/* ------------------------------------------------------------------ 5 ingredients */}
      {ablA.length > 0 && (
        <Scene id="ingredients">
          <Reveal><SectionHeader num="5" kicker="Ablation study" title="Which ingredients make the agent work?"
            lead="The development agent (v1) retrained eight times, each time without one ingredient." /></Reveal>
          <Reveal>
            <Card title="Cost increase when one ingredient is removed" subtitle="% change in overall cost vs the full agent (normal day) · faded bars = not a confident difference"
              explain="Longer red bar = that ingredient matters more. A bar near zero means the agent copes fine without it.">
              <DiffBars rows={ablA} unit="%" height={60 + ablA.length * 40} />
              <Takeaway>
                <b>The reward matters more than the inputs.</b> Telling the agent that urgent patients count more is the single most important ingredient (+11% cost without it).
                No single input group is essential, because the inputs overlap.
              </Takeaway>
              <Takeaway tone="warn">
                Removing the penalty for turning ambulances away changes <b>nothing</b>. On normal days this almost never happens, so the agent never learned from it.
                That is why the next experiment gives the agent an explicit limit on how many ambulances may be turned away.
              </Takeaway>
              <TechDetails>
                <p>8 variants × 3 agents (a disclosed reduction from 5, for compute), same seeds and hyperparameters, evaluated on the 50 test seeds in situations 1 and 5.
                  Paired hierarchical bootstrap, Holm over 80 comparisons. Only &quot;all patients weighted equally&quot; is significant after correction.
                  The other seven differences are within the uncertainty.</p>
              </TechDetails>
            </Card>
          </Reveal>
        </Scene>
      )}

      {/* ------------------------------------------------------------------ 6 tougher */}
      {(dr.length > 0 || data.constrained) && (
        <Scene id="tougher">
          <Reveal><SectionHeader num="6" kicker="Follow-up experiments" title="Three attempts to make the agent tougher"
            lead="Planned in writing before running, reported as they came out." /></Reveal>
          <div className="grid grid-2">
            {dr.length > 0 && (
              <Reveal kind="left">
                <Card title="Attempt 1: train on varied days" subtitle="% change in cost vs the original agent, per situation"
                  explain="During training, each simulated day had slightly different demand and staffing. Negative = the varied-training agent did better.">
                  <DiffBars rows={dr.map((r) => ({ ...r, color: r.sig ? undefined : "var(--series-1)" }))} unit="%" height={60 + dr.length * 40} />
                  <Takeaway>
                    A <b>small, consistent improvement</b> (3–5% lower cost in every crisis, no loss on normal days), but not large enough to be
                    statistically certain. A promising direction, not a proven fix.
                  </Takeaway>
                  <TechDetails>
                    <p>PPO-DR: arrivals ×0.95–1.04, staff and beds ×0.97–1.0, strictly milder than every test scenario (leakage guard, asserted in code).
                      5 agents, Holm over 25 comparisons: no cost difference significant; significantly fewer wrong-ward placements under combined stress.</p>
                  </TechDetails>
                </Card>
              </Reveal>
            )}
            {data.constrained && (
              <Reveal kind="right">
                <Card title="Attempt 2: an ambulance-diversion budget" subtitle="share of runs that stayed within 3 diversions per 100 arrivals"
                  explain="A controller raises the agent's diversion penalty whenever too many ambulances are turned away. Did more runs stay within budget?">
                  <WhenVisible height={260}>
                    <div style={{ width: "100%", height: 260 }}>
                      <ResponsiveContainer>
                        <BarChart data={lam} margin={{ top: 10, right: 12, bottom: 4, left: -8 }} barGap={2}>
                          <CartesianGrid stroke="var(--grid)" vertical={false} />
                          <XAxis dataKey="scenario" {...axis} tickFormatter={(s) => String(s).split(" · ")[1]} />
                          <YAxis {...axis} unit="%" domain={[0, 100]} />
                          <Tooltip {...tip} itemSorter={descending} formatter={(v, n) => [typeof v === "number" ? `${v.toFixed(0)}% of runs` : "–", n === "ppo" ? "original agent" : "agent with budget controller"]} />
                          <RLegend verticalAlign="top" height={26} formatter={(v) => (v === "ppo" ? "Original agent" : "Agent + budget controller")} wrapperStyle={{ fontSize: 12 }} />
                          <Bar dataKey="ppo" fill="var(--series-1)" radius={[4, 4, 0, 0]} animationDuration={900} />
                          <Bar dataKey="constrained" fill="var(--violet)" radius={[4, 4, 0, 0]} animationDuration={900} animationBegin={200} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </WhenVisible>
                  <Takeaway tone="warn">
                    <b>A useful negative result.</b> The controller worked (the penalty rose from about {budget["A:lambda_mean"]?.toFixed(1)} on normal days
                    to about {budget["E:lambda_mean"]?.toFixed(0)} in the hardest crisis), but the agent did not turn away fewer ambulances.
                    It never learned how to influence diversions, and in severe crises the budget is impossible for <i>every</i> method.
                  </Takeaway>
                  <TechDetails>
                    <p>λ-conditioned PPO with online projected dual ascent (λ ~ U[0,100] in training; at deployment λ ← clip(λ + 5·(d − 3), 0, 100) every 24 h),
                      planned in writing before it was run. No comparison significant after Holm correction (25 comparisons). For reference, the lowest diversion
                      of any rule: 7.9 (situation 3) and 12.0 (situation 5) per 100 emergency arrivals.</p>
                  </TechDetails>
                </Card>
              </Reveal>
            )}
          </div>
          {data.shield && (
            <Reveal>
              <Card className="shield-card" title="Attempt 3: a safety guard against the collapse"
                subtitle="test runs where the hospital collapsed (more than half of all ambulances turned away), out of 1,250"
                explain="One trained agent sometimes kept beds empty while the emergency department was full, so ambulances were turned away and the hospital spiralled. The guard forbids holding a bed in exactly that situation.">
                <div className="shield-compare">
                  {([
                    ["RL agent, no guard", data.shield.collapses.ppo, "var(--status-critical)"],
                    ["RL agent + safety guard", data.shield.collapses.ppo_shielded, "var(--status-good)"],
                    ["Balance-the-wards rule", data.shield.collapses.load_balancing, "var(--series-2)"],
                  ] as [string, number, string][]).map(([label, n, color]) => (
                    <div key={label} className="shield-tile" style={{ ["--tile" as string]: color }}>
                      <div className="shield-num"><CountUp value={n} /></div>
                      <div className="shield-lbl">{label}</div>
                    </div>
                  ))}
                </div>
                <Takeaway tone="good">
                  <b>The guard removes every collapse</b> without retraining, and the agent still ties the best rule on cost everywhere.
                </Takeaway>
                <Takeaway tone="warn">
                  It is not free: in the staff-shortage crisis average cost rises by about 4%, because a bed filled during overflow is not
                  there for a more urgent patient later. Fewer ambulances are turned away in every crisis, but the longest waits get slightly longer.
                </Takeaway>
                <TechDetails>
                  <p>Rule: if the ED is at capacity and any placement is feasible, HOLD is masked; the agent takes its best remaining action. No tunable
                    parameter. Five agents, 50 test runs × 5 situations, compared with the same agents without the guard on identical patients.
                    Cost vs unguarded agent: significant only in situation 3 (+4.4%), Holm correction over 10 comparisons. Planned in writing before running;
                    the original failure stays reported above.</p>
                </TechDetails>
              </Card>
            </Reveal>
          )}
        </Scene>
      )}

      {/* ------------------------------------------------------------------ 7 beat */}
      {data.beat_rule && (
        <Scene id="beat">
          <Reveal><SectionHeader num="7" kicker="RL agent v3" title="Can the agent beat the best rule?"
            lead="v1 tied the balance-the-wards rule. v2 learned by copying that rule and improving on it with reinforcement learning, and beat it on normal days. v3 then practised on crisis days too. Each version was tested once, on 50 runs never used before." /></Reveal>
          <div className="grid grid-2">
            <Reveal kind="left">
              <Card title={data.v3_test ? "RL agent v3 vs balance-the-wards rule" : "RL agent v2 vs balance-the-wards rule"} subtitle="% difference in overall cost, brand-new test runs · faded bars = not a confident difference"
                explain="Bars to the left (blue) mean the agent's cost is lower than the rule's. Each row is one situation.">
                <DiffBars rows={(data.v3_test ?? data.beat_rule).primary_vs_load_balancing.map((c) => ({
                  label: `${scenNum(c.scenario)} · ${scenShort(c.scenario)}`,
                  diff: (100 * c.diff_mean) / c.baseline_mean, lo: (100 * c.ci_low) / c.baseline_mean, hi: (100 * c.ci_high) / c.baseline_mean,
                  sig: c.significant_holm,
                }))} unit="%" height={260} />
                <Takeaway tone="good">
                  <b>Two clear wins, no losses:</b> 5.1% lower cost than the best hand-tuned rule on a normal day and 5.3% lower in a staff shortage,
                  both statistically certain after correcting for multiple tests. In the other three situations the gap is within the uncertainty.
                </Takeaway>
                <Takeaway tone="warn">
                  v3 practised on crisis days, so its crisis results show a crisis-trained agent, not an untrained one generalising. Version 2,
                  trained on normal days only, won on normal days (−8.5%) and tied in every crisis.
                </Takeaway>
              </Card>
            </Reveal>
            <Reveal kind="right">
              <Card title="How it was built" subtitle="validation score, normal day (lower is better)"
                explain="Every candidate was scored on validation runs only. The best one was locked in before the fresh test runs were opened.">
                <div className="stack" style={{ gap: 8 }}>
                  {([
                    ["bc", "Copying the rule only"],
                    ["load_balancing", "Balance-the-wards rule"],
                    ["ppo", "Original RL agent"],
                    ["ens_ppo", "Vote of original agents"],
                    ["bc_ppo", "Copy, then RL fine-tuning"],
                    ["ens_bc_ppo", "Vote of fine-tuned agents (v2)"],
                  ] as [string, string][]).map(([k, label]) => {
                    const v = data.beat_rule_validation?.validation_J[k]?.mean_J ?? NaN;
                    const best = k === data.beat_rule_validation?.selected;
                    return (
                      <div key={k} className={`vrow${best ? " best" : ""}`}>
                        <span>{label}</span>
                        <span className="vbar"><i style={{ width: `${Math.min(100, (100 * v) / 270)}%` }} /></span>
                        <b>{Number.isFinite(v) ? v.toFixed(1) : "–"}</b>
                      </div>
                    );
                  })}
                </div>
                {data.v3_validation && (
                  <div className="stack" style={{ gap: 8, marginTop: 12 }}>
                    <div className="eyebrow">Then: v2 or v3? Validation, all five situations (cost vs the rule)</div>
                    {(["v2", "v3"] as const).map((k) => {
                      const v = 100 * (data.v3_validation!.score_vs_lb[k] ?? NaN);
                      return (
                        <div key={k} className={`vrow${k === data.v3_validation!.selected ? " best" : ""}`}>
                          <span>{k === "v2" ? "v2 (normal days only)" : "v3 (practised crises)"}</span>
                          <span className="vbar"><i style={{ width: `${Math.min(100, 20 * Math.abs(v))}%` }} /></span>
                          <b>{Number.isFinite(v) ? `${v.toFixed(1)}%` : "–"}</b>
                        </div>
                      );
                    })}
                  </div>
                )}
                <Takeaway>
                  Copying alone fails badly: tiny mistakes add up into situations the rule never showed it. Reinforcement learning is what turns
                  the copy into an agent better than the rule it learned from, and practising crises is what made it competitive in them.
                </Takeaway>
                <TechDetails>
                  <p>Behaviour cloning of Load-Balancing (96% held-out agreement), then PPO fine-tuning (lr 1e-4, frozen observation normaliser,
                    30 critic-only warm-up updates, 3M steps), K = 5; soft-vote ensemble of the five. Planned in writing and the final choice fixed
                    before a one-time test on 50 never-used runs per situation. v3: further PPO fine-tuning with per-episode arrivals ×0.95–1.15,
                    staff and beds ×0.85–1.00 (covers situations 2–5, disclosed), selected over v2 on validation (−2.0% vs −1.2%), tested once on another
                    50 never-used runs per situation. Holm correction over 5 situations. Collapses: 0 of 1,500.</p>
                </TechDetails>
              </Card>
            </Reveal>
          </div>
        </Scene>
      )}

      {/* ------------------------------------------------------------------ 8 trust */}
      <Scene id="trust">
        <Reveal><SectionHeader num="8" kicker="Methodology" title="Why you can trust these numbers"
          lead="How the comparison was kept fair." /></Reveal>
        <div className="grid grid-2">
          <Reveal kind="left">
            <Card title="Same patients for everyone" subtitle="how many times more precise each comparison becomes"
              explain="Because every decision-maker faces exactly the same sequence of patients, luck cancels out and differences show up much more clearly.">
              <WhenVisible height={240}>
                <div style={{ width: "100%", height: 240 }}>
                  <ResponsiveContainer>
                    <BarChart data={crn} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
                      <CartesianGrid stroke="var(--grid)" horizontal={false} />
                      <XAxis type="number" {...axis} tickFormatter={(v) => `${v}×`} />
                      <YAxis type="category" dataKey="label" width={170} {...axis} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
                      <Tooltip {...tipOne} formatter={(v, _n, item) => [`${typeof v === "number" ? v.toFixed(1) : "–"}× narrower · variance −${(100 * (item.payload as { reduction: number }).reduction).toFixed(1)}%`, "Agent vs this rule"]} />
                      <Bar dataKey="ratio" barSize={16} radius={4} animationDuration={1000}
                        label={{ position: "right", fill: "var(--text-secondary)", fontSize: 11, formatter: (v: unknown) => `${Number(v).toFixed(1)}×` }}>
                        {crn.map((r) => <Cell key={r.policy} fill={pm(r.policy).color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </WhenVisible>
              <TechDetails><p>Common random numbers: separate random streams per source, patient attributes drawn at arrival. Ratio of 95% CI widths for PPO − baseline cost,
                independent seed block vs paired test seeds (situation 1).</p></TechDetails>
            </Card>
          </Reveal>
          <Reveal kind="right">
            <Card title="The ground rules we followed" subtitle="written down before looking at test results">
              <div className="stack" style={{ gap: 10 }}>
                {([
                  ["lock", "Test runs kept sealed", "50 test runs per situation were never used for training or tuning, and were opened once."],
                  ["file", "Plan first, run second", "The statistical plan was committed to git before the test run."],
                  ["target", "Fair opponents", "Every rule was tuned for its best possible score, the same way the agent was."],
                  ["repeat", "Several agents, not one", "Five independently trained agents, so one lucky (or unlucky) agent can't decide the result."],
                  ["scale", "No cherry-picking", "100 comparisons corrected together (Holm), and negative results reported in full."],
                  ["check", "Simulator verified", "Matches textbook queueing formulas in 15 of 15 checks."],
                ] as [IconName, string, string][]).map(([icon, t, d]) => (
                  <div key={t} className="guide-item"><span className="icon-badge" style={{ width: 34, height: 34, margin: 0, borderRadius: 10 }}><Icon name={icon} size={17} /></span><div><b>{t}.</b> {d}</div></div>
                ))}
              </div>
            </Card>
          </Reveal>
        </div>
        <Reveal><div style={{ marginTop: 20 }}><ReproduceCard /></div></Reveal>
      </Scene>

      {/* ------------------------------------------------------------------ glossary */}
      <Scene id="glossary">
        <Reveal><SectionHeader kicker="Glossary" title="Words used on this site" /></Reveal>
        <Reveal>
          <dl className="glossary">
            {[
              ["Overall cost score (J)", "One number combining waiting time (urgent patients count up to 4× more), turned-away ambulances and wrong-ward placements. Lower is better."],
              ["Boarding / wait for a bed", "Time an emergency patient spends waiting for a ward bed after deciding they need admission."],
              ["Diversion", "When the emergency department is full, new arrivals are sent to another hospital."],
              ["Wrong-ward (off-service)", "A patient placed in a ward of another specialty because theirs is full. Usually means a longer stay."],
              ["Urgency levels", "Critical, Very urgent, Urgent, Standard, Planned (1 to 5). Planned admissions are scheduled in advance."],
              ["RL agent (PPO)", "A neural network trained with reinforcement learning (Proximal Policy Optimisation), written from scratch for this project."],
              ["Digital twin", "A simulation that can copy the hospital's exact current state and play out possible futures from it."],
              ["Confidence interval (CI)", "The range the true value very likely lies in. If it doesn't include zero, the difference is real, not luck."],
              ["Statistically significant", "Unlikely to be luck, even after correcting for running many tests at once (Holm correction)."],
              ["Common random numbers", "Giving every decision-maker the exact same patients, so comparisons are fair and precise."],
              ["Ablation", "Removing one part of the system to see how much it mattered."],
              ["Pre-registration", "Writing down exactly how results will be analysed before seeing them, to prevent bias."],
            ].map(([t, d]) => <div key={t}><dt>{t}</dt><dd>{d}</dd></div>)}
          </dl>
        </Reveal>
        <p className="muted" style={{ marginTop: 24, textAlign: "center" }}>{data.label}</p>
      </Scene>
    </div>
  );
}
