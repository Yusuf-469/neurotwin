"use client";

import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend as RLegend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CountUp, Reveal, WhenVisible } from "@/components/motion";
import { Card, Legend, SectionHeader, Segmented, Takeaway, TechDetails, useViewToggle } from "@/components/ui";
import { api, type Comparison, type StudyResults } from "@/lib/api";
import { METRICS, POLICIES, SCENARIO_INFO, SCENARIO_LABELS, policyMeta } from "@/lib/policies";

// Study results as a guided story: each chapter asks one plain question, shows one chart, states
// the takeaway, and keeps the full statistics in a collapsible "Technical detail". All values are
// read from /api/experiments/results (docs/data/stage6 and stage7); nothing is computed or
// smoothed here beyond formatting and simple ratios.

const SCENARIOS = ["A", "B", "C", "D", "E"];
const BASELINES = ["load_balancing", "priority", "trunk_reservation", "mip", "fcfs"];
const axis = { stroke: "var(--axis)", tick: { fill: "var(--text-muted)", fontSize: 11 }, tickLine: false } as const;
const tip = {
  contentStyle: { background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12, color: "var(--text-primary)" },
  cursor: { fill: "var(--surface-2)" },
};
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
  { id: "headtohead", label: "2 · AI vs each rule" },
  { id: "crisis", label: "3 · Coping with crises" },
  { id: "clever", label: "4 · Did it learn something?" },
  { id: "ingredients", label: "5 · What makes it work?" },
  { id: "tougher", label: "6 · Making it tougher" },
  { id: "trust", label: "7 · Why trust this?" },
  { id: "glossary", label: "Glossary" },
];

/** Diverging colour for an effect: negative (AI better) -> blue, positive -> red, ~0 -> gray. */
function divergeColor(value: number, max: number): string {
  const t = Math.max(-1, Math.min(1, value / max));
  const pole = t < 0 ? "var(--diverge-neg)" : "var(--diverge-pos)";
  return `color-mix(in srgb, ${pole} ${Math.round(Math.abs(t) * 85)}%, var(--diverge-mid))`;
}

/** Highlights the chapter currently on screen. */
function useActiveChapter(): string {
  const [active, setActive] = useState(CHAPTERS[0].id);
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (vis[0]) setActive(vis[0].target.id);
    }, { rootMargin: "-20% 0px -65% 0px" });
    CHAPTERS.forEach((c) => { const el = document.getElementById(c.id); if (el) io.observe(el); });
    return () => io.disconnect();
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
            <Tooltip {...tip} formatter={(v, _n, item) => {
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
  if (!data?.summary) return <p className="muted" style={{ marginTop: 24 }}>Loading study results… (the server may take up to a minute to wake up)</p>;

  const meta = METRICS.find((m) => m.key === metric)!;
  const value = (s: string, p: string): [number, number, number] | null => {
    const e = data.summary!.find((x) => x.scenario === s && x.policy === p)?.[metric];
    return Array.isArray(e) ? e : null;
  };
  const barData = SCENARIOS.map((s) => {
    const row: Record<string, number | string | [number, number]> = { scenario: s };
    for (const p of POLICIES) {
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
  // Keep-a-bed-free is identical to most-urgent-first (tuned reserve 0), so it is not counted twice.
  const matchups = Object.entries(heat).filter(([b]) => b !== "trunk_reservation").flatMap(([, r]) => Object.values(r));
  const wins = matchups.filter((c) => c.sig && c.diff < 0).length;
  const losses = matchups.filter((c) => c.sig && c.diff > 0).length;

  const ri = Object.entries(data.robustness?.robustness_index ?? {})
    .map(([p, v]) => ({ policy: p, label: policyMeta(p).label, value: v.estimate,
      err: [v.estimate - v.ci[0], v.ci[1] - v.estimate] as [number, number] }))
    .sort((a, b) => a.value - b.value);
  const crn = (data.crn_check ?? []).filter((c) => c.metric === "J_per_day" && c.baseline !== "trunk_reservation")
    .map((c) => ({ label: policyMeta(c.baseline).label, policy: c.baseline, ratio: c.width_ratio_independent_over_paired,
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
    .map((c) => toRow(c, `${c.scenario} · ${scenShort(c.scenario)}`, 100 / c.baseline_mean));
  const budget = data.constrained?.budget ?? {};
  const lam = SCENARIOS.map((s) => ({
    scenario: `${s} · ${scenShort(s)}`,
    lambda: budget[`${s}:lambda_mean`],
    ppo: 100 * (budget[`${s}:ppo`] ?? NaN),
    constrained: 100 * (budget[`${s}:ppo_lagrangian`] ?? NaN),
  }));

  return (
    <div>
      {/* ------------------------------------------------------------------ intro */}
      <section className="hero2" style={{ marginBottom: 8 }}>
        <span className="blob b1" /><span className="blob b2" />
        <Reveal><span className="pill"><span className="badge">Results</span> 2,500 test episodes · 50 fresh runs per situation</span></Reveal>
        <Reveal delay={80}><h1 style={{ fontSize: "clamp(30px, 4.4vw, 52px)" }}>What happened when the AI <span className="gradient-text">met the rules</span></h1></Reveal>
        <Reveal delay={160}><p className="lead">
          Seven short chapters, each answering one question. Every chart says what you are looking at and what
          it means. The full statistics are under &quot;Technical detail&quot;. Lower is better everywhere.
        </p></Reveal>
        <Reveal delay={240}>
          <div className="ribbon">
            <div><div className="num"><CountUp value={wins} /></div><div className="lbl">confident AI wins out of {matchups.length} match-ups ({meta.label.toLowerCase()})</div></div>
            <div><div className="num"><CountUp value={losses} /></div><div className="lbl">confident AI losses</div></div>
            <div><div className="num"><CountUp value={crnMax} digits={1} suffix="×" /></div><div className="lbl">sharper comparisons (same patients)</div></div>
            <div><div className="num"><CountUp value={(data.ablations?.length ?? 0) + (data.ppo_dr?.length ?? 0) + (data.constrained?.comparisons.length ?? 0) + (data.comparisons?.length ?? 0)} /></div><div className="lbl">pre-planned statistical tests</div></div>
          </div>
        </Reveal>
      </section>

      <nav className="chapters" aria-label="Chapters">
        {CHAPTERS.map((c) => <a key={c.id} href={`#${c.id}`} className={active === c.id ? "on" : ""}>{c.label}</a>)}
      </nav>

      {/* ------------------------------------------------------------------ 1 overall */}
      <section className="section" id="overall">
        <Reveal><SectionHeader num="1" kicker="Overall" title="Who does best in each situation?"
          lead="Pick what to measure. Each group of bars is one situation; each bar is one decision-maker." /></Reveal>
        <Reveal>
          <div className="row" style={{ justifyContent: "center", marginBottom: 16 }}>
            <Segmented options={METRICS.map((m) => ({ value: m.key, label: m.label }))} value={metric} onChange={setMetric} />
          </div>
        </Reveal>
        <Reveal>
          <Card title={meta.label} subtitle={`${meta.unit} · average of 50 test runs · thin whiskers show the uncertainty range`} actions={barToggle}
            explain={meta.desc}>
            <Legend items={POLICIES.map((p) => ({ label: p.label, color: p.color }))} />
            {barTable ? (
              <div className="scroll" style={{ marginTop: 10 }}>
                <table>
                  <thead><tr><th>situation</th>{POLICIES.map((p) => <th key={p.key}>{p.label}</th>)}</tr></thead>
                  <tbody>{SCENARIOS.map((s) => (
                    <tr key={s}><td>{SCENARIO_LABELS[s]}</td>{POLICIES.map((p) => {
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
                      <Tooltip {...tip} labelFormatter={(s) => SCENARIO_LABELS[String(s)] ?? String(s)}
                        formatter={(v, name) => [typeof v === "number" ? `${v.toFixed(meta.digits)} ${meta.unit}` : "–", policyMeta(String(name)).label]} />
                      {POLICIES.map((p, i) => (
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
              times over. Within each situation, the <b>AI agent</b> and the <b>balance-the-wards rule</b> are consistently among the best,
              and <b>first-come-first-served</b> is clearly the worst.
            </Takeaway>
            <TechDetails>
              <p>Means over 50 test seeds (200000–200049); AI averaged over 5 independently trained agents. Whiskers: 95% t-intervals over seeds.
                Boarding is censoring-inclusive (patients still waiting at the horizon count with their wait so far), a disclosed protocol amendment.</p>
            </TechDetails>
          </Card>
        </Reveal>
      </section>

      {/* ------------------------------------------------------------------ 2 head to head */}
      <section className="section" id="headtohead">
        <Reveal><SectionHeader num="2" kicker="Head to head" title="Where does the AI beat each rule, and where does it lose?"
          lead={<>Each square compares the AI with one rule in one situation. <b style={{ color: "var(--diverge-neg)" }}>Blue</b> = AI better,{" "}
            <b style={{ color: "var(--diverge-pos)" }}>red</b> = AI worse, grey = no real difference. A dot ● marks a difference we are statistically confident about.</>} /></Reveal>
        <Reveal>
          <Card title={`AI minus rule: ${meta.label.toLowerCase()}`} subtitle={meta.unit} actions={heatToggle}
            explain={`Numbers show how much higher (+) or lower (−) the AI's ${meta.label.toLowerCase()} is than each rule's. Negative is good for the AI.`}>
            {heatTable ? (
              <div className="scroll">
                <table>
                  <thead><tr><th>AI vs</th>{SCENARIOS.map((s) => <th key={s}>{scenShort(s)}</th>)}</tr></thead>
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
                <div className="heat scroll" style={{ gridTemplateColumns: `200px repeat(${SCENARIOS.length}, minmax(96px, 1fr))` }}>
                  <div />
                  {SCENARIOS.map((s) => <div key={s} className="hdr">{SCENARIO_INFO[s].icon} {scenShort(s)}</div>)}
                  {BASELINES.map((b, bi) => (
                    <div key={b} style={{ display: "contents" }}>
                      <div className="rowhdr"><i className="dot" style={{ background: policyMeta(b).color, marginRight: 6 }} />{policyMeta(b).label}</div>
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
                  <span><i className="swatch" style={{ background: divergeColor(-1.5, 1.5) }} />AI better</span>
                  <span><i className="swatch" style={{ background: "var(--diverge-mid)", border: "1px solid var(--grid)" }} />no real difference</span>
                  <span><i className="swatch" style={{ background: divergeColor(1.5, 1.5) }} />AI worse</span>
                  <span>● confident difference</span>
                </div>
              </>
            )}
            <Takeaway tone="good">
              Against the classic rules (most-urgent-first, the instant optimiser, first-come-first-served) the AI is mostly blue.
              Against the <b>balance-the-wards rule</b> the squares are grey: it is a genuine tie.
            </Takeaway>
            <TechDetails>
              <p>{metric === "J_per_day"
                ? "Overall cost: cell = mean paired difference (AI − rule) with relative %; ● = 95% hierarchical-bootstrap CI excludes 0 (secondary, descriptive)."
                : "Cell = mean paired difference; second line = effect size d_z; ● = significant after Holm correction over all 100 pre-registered primary comparisons."}</p>
              <p>The keep-a-bed-free rule tuned itself to reserve zero beds, so it behaves exactly like most-urgent-first (identical numbers).</p>
            </TechDetails>
          </Card>
        </Reveal>
      </section>

      {/* ------------------------------------------------------------------ 3 crisis */}
      <section className="section" id="crisis">
        <Reveal><SectionHeader num="3" kicker="Robustness" title="How much worse does each one get in a crisis?"
          lead="Compared with its own normal-day performance, how much does each decision-maker deteriorate across the four crises?" /></Reveal>
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
                      <Tooltip {...tip} formatter={(v) => [typeof v === "number" ? `${v.toFixed(0)}% worse` : "–", "deterioration"]} />
                      <Bar dataKey="value" barSize={16} radius={4} animationDuration={1000}>
                        {ri.map((r) => <Cell key={r.policy} fill={policyMeta(r.policy).color} />)}
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
              <Takeaway tone="good">The AI deteriorates <b>clearly less</b> than the most-urgent-first rule, and about as much as the best rules.</Takeaway>
              <Takeaway tone="warn">
                First-come-first-served looks &quot;steady&quot; only because it is already bad on a normal day: there is less room to get worse.
                Always read this together with chapter 1.
              </Takeaway>
              <TechDetails>
                <p>Robustness index = mean relative degradation (stress − A)/A over scenarios B–E and four metrics (cost, mean and 90th-percentile boarding,
                  off-service). Bootstrap over seeds jointly across scenarios and over agents. Diversion is excluded (≈0 in A makes ratios meaningless) and
                  reported as an absolute change in the Stage 7 report.</p>
              </TechDetails>
            </Card>
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------------------------ 4 clever */}
      {structure.length > 0 && (
        <section className="section" id="clever">
          <Reveal><SectionHeader num="4" kicker="Inside the AI" title="Did the AI discover something on its own?"
            lead="Hospitals sometimes keep a bed empty for the next emergency. Nobody told the AI about this idea. Did it find it anyway?" /></Reveal>
          <Reveal>
            <Card title="How often the AI keeps a free bed empty"
              subtitle="when only non-urgent patients are waiting, by number of free beds"
              explain="Left = almost full, right = plenty of room. A falling line means: hold back when beds are scarce, fill freely when there is room.">
              <WhenVisible height={300}>
                <div style={{ width: "100%", height: 300 }}>
                  <ResponsiveContainer>
                    <LineChart data={structure} margin={{ top: 10, right: 16, bottom: 4, left: -8 }}>
                      <CartesianGrid stroke="var(--grid)" vertical={false} />
                      <XAxis dataKey="free" {...axis} label={{ value: "free beds in the ward", position: "insideBottom", offset: -2, fill: "var(--text-muted)", fontSize: 11 }} height={36} />
                      <YAxis {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip {...tip} cursor={{ stroke: "var(--axis)" }} formatter={(v, n) => [typeof v === "number" ? `${(v * 100).toFixed(1)}% of the time` : "–", n === "A" ? "normal day" : "everything at once"]} />
                      <RLegend verticalAlign="top" height={28} formatter={(v) => (v === "A" ? "Normal day (trained on)" : "Everything at once (never seen)")} wrapperStyle={{ fontSize: 12 }} />
                      <Line dataKey="A" stroke="var(--series-1)" strokeWidth={2.5} dot={{ r: 4 }} animationDuration={1400} />
                      <Line dataKey="E" stroke="var(--series-2)" strokeWidth={2.5} dot={{ r: 4 }} animationDuration={1400} animationBegin={300} />
                      <ReferenceLine y={0} stroke="var(--axis)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </WhenVisible>
              <Takeaway tone="good">
                <b>Yes, on normal days.</b> The blue line falls: the AI learned a soft &quot;keep a bed free&quot; strategy, the shape queueing theory predicts.
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
        </section>
      )}

      {/* ------------------------------------------------------------------ 5 ingredients */}
      {ablA.length > 0 && (
        <section className="section" id="ingredients">
          <Reveal><SectionHeader num="5" kicker="Ablation study" title="Which ingredients make the AI work?"
            lead="We retrained the AI eight times, each time taking one ingredient away, and measured how much worse it got on a normal day." /></Reveal>
          <Reveal>
            <Card title="Cost increase when one ingredient is removed" subtitle="% change in overall cost vs the full AI (normal day) · faded bars = not a confident difference"
              explain="Longer red bar = that ingredient matters more. A bar near zero means the AI copes fine without it.">
              <DiffBars rows={ablA} unit="%" height={60 + ablA.length * 40} />
              <Takeaway>
                <b>The reward matters more than the inputs.</b> Telling the AI that urgent patients count more is the single most important ingredient (+11% cost without it).
                No single input group is essential, because the inputs overlap.
              </Takeaway>
              <Takeaway tone="warn">
                Removing the penalty for turning ambulances away changes <b>nothing</b>. On normal days this almost never happens, so the AI never learned from it.
                This motivated chapter 6.
              </Takeaway>
              <TechDetails>
                <p>8 variants × 3 agents (a disclosed reduction from 5, for compute), same seeds and hyperparameters, evaluated on the 50 test seeds in A and E.
                  Paired hierarchical bootstrap, Holm over 80 comparisons. Only &quot;all patients weighted equally&quot; is significant after correction.
                  Full table in docs/stage7_analysis_report.md §4.</p>
              </TechDetails>
            </Card>
          </Reveal>
        </section>
      )}

      {/* ------------------------------------------------------------------ 6 tougher */}
      {(dr.length > 0 || data.constrained) && (
        <section className="section" id="tougher">
          <Reveal><SectionHeader num="6" kicker="Follow-up experiments" title="Two attempts to make the AI tougher"
            lead="Both were planned and written down before running them, and both are reported exactly as they came out." /></Reveal>
          <div className="grid grid-2">
            {dr.length > 0 && (
              <Reveal kind="left">
                <Card title="Attempt 1: train on varied days" subtitle="% change in cost vs the original AI, per situation"
                  explain="During training, each simulated day had slightly different demand and staffing. Negative = the varied-training AI did better.">
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
                  explain="A controller raises the AI's diversion penalty whenever too many ambulances are turned away. Did more runs stay within budget?">
                  <WhenVisible height={260}>
                    <div style={{ width: "100%", height: 260 }}>
                      <ResponsiveContainer>
                        <BarChart data={lam} margin={{ top: 10, right: 12, bottom: 4, left: -8 }} barGap={2}>
                          <CartesianGrid stroke="var(--grid)" vertical={false} />
                          <XAxis dataKey="scenario" {...axis} tickFormatter={(s) => String(s).split(" · ")[1]} />
                          <YAxis {...axis} unit="%" domain={[0, 100]} />
                          <Tooltip {...tip} formatter={(v, n) => [typeof v === "number" ? `${v.toFixed(0)}% of runs` : "–", n === "ppo" ? "original AI" : "AI with budget controller"]} />
                          <RLegend verticalAlign="top" height={26} formatter={(v) => (v === "ppo" ? "Original AI" : "AI + budget controller")} wrapperStyle={{ fontSize: 12 }} />
                          <Bar dataKey="ppo" fill="var(--series-1)" radius={[4, 4, 0, 0]} animationDuration={900} />
                          <Bar dataKey="constrained" fill="var(--violet)" radius={[4, 4, 0, 0]} animationDuration={900} animationBegin={200} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </WhenVisible>
                  <Takeaway tone="warn">
                    <b>A useful negative result.</b> The controller worked (the penalty rose from about {budget["A:lambda_mean"]?.toFixed(1)} on normal days
                    to about {budget["E:lambda_mean"]?.toFixed(0)} in the hardest crisis), but the AI did not turn away fewer ambulances.
                    It never learned how to influence diversions, and in severe crises the budget is impossible for <i>every</i> method.
                  </Takeaway>
                  <TechDetails>
                    <p>λ-conditioned PPO with online projected dual ascent (λ ~ U[0,100] in training; at deployment λ ← clip(λ + 5·(d − 3), 0, 100) every 24 h),
                      pre-registered as RESEARCH.md v2.6. No comparison significant after Holm (25). Lowest diversion of any policy in Stage 6: 7.9 (C) and 12.0 (E) per 100.</p>
                  </TechDetails>
                </Card>
              </Reveal>
            )}
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------ 7 trust */}
      <section className="section" id="trust">
        <Reveal><SectionHeader num="7" kicker="Methodology" title="Why you can trust these numbers"
          lead="A simulation study is only as good as its fairness. Here is how we kept it honest." /></Reveal>
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
                      <Tooltip {...tip} formatter={(v, _n, item) => [`${typeof v === "number" ? v.toFixed(1) : "–"}× narrower · variance −${(100 * (item.payload as { reduction: number }).reduction).toFixed(1)}%`, "AI vs this rule"]} />
                      <Bar dataKey="ratio" barSize={16} radius={4} animationDuration={1000}
                        label={{ position: "right", fill: "var(--text-secondary)", fontSize: 11, formatter: (v: unknown) => `${Number(v).toFixed(1)}×` }}>
                        {crn.map((r) => <Cell key={r.policy} fill={policyMeta(r.policy).color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </WhenVisible>
              <TechDetails><p>Common random numbers: separate random streams per source, patient attributes drawn at arrival. Ratio of 95% CI widths for PPO − baseline cost,
                independent seed block vs paired test seeds (scenario A).</p></TechDetails>
            </Card>
          </Reveal>
          <Reveal kind="right">
            <Card title="The ground rules we followed" subtitle="written down before looking at test results">
              <div className="stack" style={{ gap: 10 }}>
                {[
                  ["🔒", "Test runs kept sealed", "50 test runs per situation were never used for training or tuning, and were opened once."],
                  ["📝", "Plan first, run second", "The statistical plan was committed to git before the test run."],
                  ["🎯", "Fair opponents", "Every rule was tuned for its best possible score, the same way the AI was."],
                  ["🔁", "Several AIs, not one", "Five independently trained agents, so one lucky (or unlucky) agent can't decide the result."],
                  ["⚖️", "No cherry-picking", "100 comparisons corrected together (Holm), and negative results reported in full."],
                  ["✅", "Simulator verified", "Matches textbook queueing formulas in 15 of 15 checks."],
                ].map(([icon, t, d]) => (
                  <div key={t} className="guide-item"><span style={{ fontSize: 20 }}>{icon}</span><div><b>{t}.</b> {d}</div></div>
                ))}
              </div>
            </Card>
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------------------------ glossary */}
      <section className="section" id="glossary">
        <Reveal><SectionHeader kicker="Glossary" title="Words used on this site" /></Reveal>
        <Reveal>
          <dl className="glossary">
            {[
              ["Overall cost score (J)", "One number combining waiting time (urgent patients count up to 4× more), turned-away ambulances and wrong-ward placements. Lower is better."],
              ["Boarding / wait for a bed", "Time an emergency patient spends waiting for a ward bed after deciding they need admission."],
              ["Diversion", "When the emergency department is full, new arrivals are sent to another hospital."],
              ["Wrong-ward (off-service)", "A patient placed in a ward of another specialty because theirs is full. Usually means a longer stay."],
              ["Urgency levels", "Critical, Very urgent, Urgent, Standard, Planned (1 to 5). Planned admissions are scheduled in advance."],
              ["AI agent (PPO)", "A neural network trained with reinforcement learning (Proximal Policy Optimisation), written from scratch for this project."],
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
      </section>
    </div>
  );
}
