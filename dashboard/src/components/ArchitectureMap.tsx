"use client";

// Clickable map of the system: the parts, the data flowing between them, and for each part
// what it does, what is notable about it, and where its code and tests live. Facts match the
// README's component table; file paths are real.

import { useState } from "react";
import { Icon, type IconName } from "./icons";

interface Part {
  id: string;
  title: string;
  /** Short name shown inside the box (the full title is in the detail panel). */
  label?: string;
  icon: IconName;
  short: string;
  points: string[];
  code: string[];
  tests: string[];
  x: number;
  y: number;
}

const W = 168;
const H = 58;

const PARTS: Part[] = [
  { id: "sim", title: "Simulator", icon: "build", x: 16, y: 176,
    short: "A virtual hospital: arrivals, treatment, stays, staff shifts, minute by minute.",
    points: ["Discrete-event simulation written from scratch", "Separate random streams per source, so every decision-maker can face identical patients",
      "Matches textbook queueing formulas in 15 of 15 checks, and detects a deliberate 5% input error"],
    code: ["neurotwin/simulation/hospital.py"], tests: ["test_engine.py", "test_queueing_formulas.py", "test_validation_mmck.py"] },
  { id: "twin", title: "Digital twin", icon: "mirror", x: 214, y: 96,
    short: "Freezes the hospital at any moment and branches possible futures from that exact state.",
    points: ["Exact snapshot and restore", "Branches keep the present and re-draw the future; stays already in progress are re-drawn given the time already spent",
      "Warm-start pool of realistic busy states for training"],
    code: ["neurotwin/simulation/hospital.py (clone, branch)", "neurotwin/digital_twin/"], tests: ["test_digital_twin.py"] },
  { id: "rules", title: "Rules and optimiser", label: "Rules, optimiser", icon: "scale", x: 412, y: 26,
    short: "Five hand-designed decision-makers, including an exact optimiser.",
    points: ["First-come-first-served, most-urgent-first, balance-the-wards, keep-a-bed-free", "Exact one-step optimisation (MIP), checked against LP and Hungarian-algorithm oracles",
      "Each tuned for its best score on separate tuning runs, the same way the agent was"],
    code: ["neurotwin/policies/heuristics.py", "neurotwin/policies/optimization.py"], tests: ["test_policies.py"] },
  { id: "rl", title: "RL agent", icon: "cpu", x: 412, y: 150,
    short: "A small neural network that learned bed allocation by trial and error.",
    points: ["7 possible moves, masked so it can only choose what is physically possible", "An actor network (chooses) and a critic network (judges), 2 × 64 neurons each, 11,656 learned numbers in total",
      "Latest version: five independently trained networks vote"],
    code: ["neurotwin/rl/env.py", "neurotwin/policies/ensemble_ppo.py"], tests: ["test_rl.py", "test_stage7.py"] },
  { id: "train", title: "RL training (PPO)", label: "RL training", icon: "repeat", x: 412, y: 286,
    short: "Proximal Policy Optimisation, implemented from scratch in PyTorch.",
    points: ["Time-aware discounting by simulated hours between decisions (semi-Markov GAE)", "Checked against stable-baselines3 and MaskablePPO",
      "v2 started from a copy of the best rule; v3 practised on randomly varied crisis days"],
    code: ["neurotwin/rl/ppo/", "neurotwin/rl/bc.py"], tests: ["test_rl.py", "test_bc.py"] },
  { id: "exp", title: "Experiments and statistics", label: "Experiments", icon: "flask", x: 214, y: 286,
    short: "The fair contest: sealed test runs, paired comparisons, corrected statistics.",
    points: ["Training, tuning and test runs kept apart, enforced in code", "Hierarchical paired bootstrap; Holm correction over all comparisons",
      "Every analysis written down before it was run, negative results included"],
    code: ["neurotwin/experiments/", "neurotwin/experiments/seeds.py"], tests: ["test_statistics.py"] },
  { id: "explain", title: "Decision log and explanations", label: "Explanations", icon: "chat", x: 610, y: 88,
    short: "Every decision is logged and explained; explanations never change what happens.",
    points: ["Read-only decision log of the facts available at each decision", "Plain-language explanations built only from logged facts; an optional AI writer is number-checked",
      "Inside the agent: each network's vote and integrated-gradients attributions"],
    code: ["neurotwin/policies/decision_log.py", "neurotwin/explanation/", "neurotwin/rl/introspect.py"], tests: ["test_explanation.py", "test_insight.py"] },
  { id: "whatif", title: "What-if, race and futures", label: "What-if and race", icon: "split", x: 610, y: 286,
    short: "Compare decision-makers from any moment on the same random futures.",
    points: ["Live race: identical copies of one hospital, one per competitor", "Possible futures: percentile bands with a paired comparison",
      "Shared futures make the comparisons fair and sharp"],
    code: ["neurotwin/whatif/"], tests: ["test_whatif.py", "test_fan.py"] },
  { id: "api", title: "API", icon: "network", x: 808, y: 150,
    short: "FastAPI backend that runs simulations live and serves the study results.",
    points: ["Live runs, races, explanations, futures, the study results, and re-running any sealed test run", "Errors are reported cleanly, so the website can reconnect and recover",
      "Docker container on a free Render instance"],
    code: ["neurotwin/api/app.py", "Dockerfile"], tests: ["test_api.py"] },
  { id: "dash", title: "This website", icon: "chart", x: 808, y: 286,
    short: "Next.js and React, exported as a static site and hosted on Vercel.",
    points: ["Charts with Recharts; every animation hand-written, no animation library", "Reconnects and restores live runs if the free server restarts",
      "Every number on it comes from the simulation, labelled as such"],
    code: ["dashboard/src/"], tests: ["type-check and production build"] },
];

/** [from, to, label, optional waypoint routing the line through a gap between boxes] */
const FLOWS: [string, string, string, [number, number]?][] = [
  ["sim", "twin", "state"], ["twin", "rules", "state"], ["twin", "rl", "state"], ["train", "rl", "weights"],
  ["sim", "exp", "runs"], ["rules", "explain", "decisions"], ["rl", "explain", "decisions"], ["twin", "whatif", "branches", [540, 247]],
  ["explain", "api", ""], ["whatif", "api", ""], ["api", "dash", "JSON"],
];

const byId = Object.fromEntries(PARTS.map((p) => [p.id, p]));

function edgePath(a: Part, b: Part, via?: [number, number]): string {
  if (via) {
    // leave from the bottom of the source, run along the gap row, then into the target's side
    const x1 = a.x + W / 2, y1 = a.y + H, x2 = b.x, y2 = b.y + H / 2, [vx, vy] = via;
    return `M${x1} ${y1} C${x1} ${vy}, ${x1} ${vy}, ${x1 + 40} ${vy} L${vx} ${vy} C${vx + 50} ${vy}, ${x2 - 40} ${y2}, ${x2} ${y2}`;
  }
  // right side of a to left side of b, or vertical when stacked
  if (Math.abs(a.x - b.x) < 10) {
    const x = a.x + W / 2;
    const [y1, y2] = a.y > b.y ? [a.y, b.y + H] : [a.y + H, b.y];
    return `M${x} ${y1} L${x} ${y2}`;
  }
  const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
  const mx = (x1 + x2) / 2;
  return `M${x1} ${y1} C${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

export default function ArchitectureMap() {
  const [sel, setSel] = useState("rl");
  const p = byId[sel];
  const linked = new Set(FLOWS.filter(([a, b]) => a === sel || b === sel).flatMap(([a, b]) => [a, b]));

  return (
    <div className="arch">
      <svg className="arch-svg" viewBox="0 0 992 360" role="group" aria-label="System parts and how data flows between them">
        <defs>
          <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--axis)" />
          </marker>
        </defs>
        {FLOWS.map(([a, b, label, via], i) => {
          const on = a === sel || b === sel;
          const d = edgePath(byId[a], byId[b], via);
          return (
            <g key={i} className={`arch-edge${on ? " on" : ""}`}>
              <path d={d} className="arch-line" markerEnd="url(#arch-arrow)" />
              <path d={d} className="arch-flow" style={{ animationDelay: `${-i * 0.37}s` }} />
              {label && on && (() => {
                const A = byId[a], B = byId[b];
                if (via) return <text x={via[0] - 80} y={via[1] - 7} className="arch-label" textAnchor="middle">{label}</text>;
                const vertical = Math.abs(A.x - B.x) < 10;
                // between the facing edges of the two boxes, not their centres
                const lx = vertical ? A.x + W / 2 + 26 : (A.x + W + B.x) / 2;
                const ly = vertical ? (Math.min(A.y, B.y) + H + Math.max(A.y, B.y)) / 2 + 4 : (A.y + B.y + H) / 2 - 6;
                return <text x={lx} y={ly} className="arch-label" textAnchor="middle">{label}</text>;
              })()}
            </g>
          );
        })}
        {PARTS.map((n, i) => (
          <g key={n.id} className={`arch-node${n.id === sel ? " sel" : ""}${linked.has(n.id) && n.id !== sel ? " linked" : ""}`}
            transform={`translate(${n.x},${n.y})`} style={{ animationDelay: `${i * 60}ms` }}
            onClick={() => setSel(n.id)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSel(n.id)}
            tabIndex={0} role="button" aria-pressed={n.id === sel} aria-label={n.title}>
            <rect width={W} height={H} rx={12} className="arch-box" />
            <text x={W / 2} y={H / 2 + 5} textAnchor="middle" className="arch-title">{n.label ?? n.title}</text>
          </g>
        ))}
      </svg>

      {/* phones: the same parts as a tappable grid */}
      <div className="arch-grid">
        {PARTS.map((n) => (
          <button key={n.id} className={n.id === sel ? "on" : ""} onClick={() => setSel(n.id)}>
            <Icon name={n.icon} size={15} /> {n.title}
          </button>
        ))}
      </div>

      <div className="arch-detail card" key={p.id}>
        <div className="row" style={{ gap: 10 }}>
          <span className="icon-badge" style={{ width: 38, height: 38, margin: 0, borderRadius: 11 }}><Icon name={p.icon} size={18} /></span>
          <div><h3 style={{ margin: 0 }}>{p.title}</h3><span className="muted">{p.short}</span></div>
        </div>
        <ul className="arch-points">{p.points.map((t) => <li key={t}>{t}</li>)}</ul>
        <div className="arch-files">
          <span className="eyebrow">Code</span>
          {p.code.map((c) => <code key={c}>{c}</code>)}
          <span className="eyebrow">Tested by</span>
          {p.tests.map((t) => <code key={t}>{t}</code>)}
        </div>
      </div>
    </div>
  );
}
