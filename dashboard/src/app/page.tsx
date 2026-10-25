"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Card, Stat } from "@/components/ui";
import { api, type StudyResults } from "@/lib/api";

function Pipeline() {
  const boxes = [
    { x: 10, title: "Event simulator", sub: "Erlang C · M/M/c/K checks" },
    { x: 222, title: "Digital twin", sub: "snapshots · what-if" },
    { x: 434, title: "Decision policies", sub: "rules · MIP · PPO" },
    { x: 646, title: "Evaluation", sub: "CRN · bootstrap · Holm" },
    { x: 858, title: "Explain & explore", sub: "log · what-if · UI" },
  ];
  return (
    <svg className="flow" viewBox="0 0 1060 120" role="img"
      aria-label="Pipeline: simulator, digital twin, policies, evaluation, explanation and dashboard">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0L10 5L0 10z" fill="var(--axis)" />
        </marker>
      </defs>
      {boxes.map((b, i) => (
        <g key={b.title}>
          <rect x={b.x} y={20} width={190} height={80} rx={12} fill="var(--surface-2)" stroke="var(--border)" />
          <rect x={b.x} y={20} width={4} height={80} rx={2} fill="var(--accent)" />
          <text x={b.x + 16} y={52} fill="var(--text-primary)" fontSize="14" fontWeight="650">{b.title}</text>
          <text x={b.x + 16} y={74} fill="var(--text-muted)" fontSize="11.5">{b.sub}</text>
          {i < boxes.length - 1 && (
            <line x1={b.x + 192} y1={60} x2={b.x + 218} y2={60} stroke="var(--text-muted)" strokeWidth={2}
              markerEnd="url(#arrow)" />
          )}
        </g>
      ))}
    </svg>
  );
}

export default function Overview() {
  const [data, setData] = useState<StudyResults | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api.health().then(() => setOnline(true)).catch(() => setOnline(false));
    api.results().then(setData).catch(() => setData(null));
  }, []);

  const lb = data?.J_comparisons?.filter((c) => c.baseline === "load_balancing") ?? [];
  const pri = data?.J_comparisons?.filter((c) => c.baseline === "priority") ?? [];
  const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const crnMax = Math.max(...(data?.crn_check?.map((c) => c.width_ratio_independent_over_paired) ?? [NaN]));
  const rhoA = data?.structure?.A?.spearman_rho;
  const fmtPct = (v: number) => (Number.isFinite(v) ? `${v > 0 ? "+" : ""}${v.toFixed(1)}%` : "–");

  return (
    <div className="stack">
      <section className="hero">
        <span className="eyebrow">Final-year research project · simulation study</span>
        <h1>Can a learned policy allocate hospital beds better than rules and optimisation?</h1>
        <p>
          NeuroTwin is a validated discrete-event digital twin of an emergency department, two
          specialty wards and an ICU. It compares an in-house PPO agent with tuned rules and an
          exactly solved optimiser, under normal load and under stress the agent never saw in
          training, using pre-registered statistics over 2,500 test episodes.
        </p>
        <div className="row">
          <Link href="/simulation/" className="btn">Open the live twin</Link>
          <Link href="/experiments/" className="btn ghost">See the study results</Link>
          <span className="chip">
            <i className="dot" style={{ background: online ? "var(--status-good)" : online === false ? "var(--status-critical)" : "var(--text-muted)" }} />
            API {online ? "online" : online === false ? "offline" : "checking"}
          </span>
        </div>
      </section>

      <div className="grid grid-4">
        <Stat label="PPO vs best tuned rule (Load-Balancing)" value={fmtPct(avg(lb.map((c) => c.rel_diff_pct)))}
          note="mean cost difference over 5 scenarios; no scenario significant" />
        <Stat label="PPO vs Priority rule" value={fmtPct(avg(pri.map((c) => c.rel_diff_pct)))}
          note="lower in all 5 scenarios; CI excludes 0 in 3 (A, C, D)" />
        <Stat label="CI narrowing from common random numbers" value={Number.isFinite(crnMax) ? `${crnMax.toFixed(1)}×` : "–"}
          note="paired vs independent seeds (scenario A)" />
        <Stat label="Learned bed reservation (scenario A)" value={rhoA !== undefined ? `ρ = ${rhoA.toFixed(2)}` : "–"}
          note="P(hold) falls as free beds rise: threshold shape" />
      </div>

      <Card title="How the system fits together" subtitle="Each stage has its own validation evidence in docs/">
        <div className="scroll"><Pipeline /></div>
      </Card>

      <div className="grid grid-3">
        <Card title="Validated simulator" subtitle="15/15 analytical checks inside the 95% CI">
          <p className="secondary" style={{ margin: 0 }}>
            Reduced configurations reproduce Erlang C and M/M/c/K results; a deliberate 5% error in
            arrivals is detected. Warm-up length set by Welch&apos;s method.
          </p>
        </Card>
        <Card title="Honest evaluation" subtitle="Protocol committed before the test run">
          <p className="secondary" style={{ margin: 0 }}>
            50 untouched test seeds, hierarchical paired bootstrap over 5 agents, Holm correction
            over 100 comparisons. One amendment, disclosed.
          </p>
        </Card>
        <Card title="A failure mode rules don&apos;t have" subtitle="1 of 5 agents collapsed in 2% of stress runs">
          <p className="secondary" style={{ margin: 0 }}>
            Out-of-distribution states made one agent hold patients while beds emptied. No rule-based
            policy ever collapsed, a key robustness finding.
          </p>
        </Card>
      </div>
    </div>
  );
}
