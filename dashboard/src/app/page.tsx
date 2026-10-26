"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { CountUp, Reveal } from "@/components/motion";
import { SectionHeader, TechDetails } from "@/components/ui";
import { api, type StudyResults } from "@/lib/api";
import { POLICIES, PRIORITY_COLORS, SCENARIO_INFO } from "@/lib/policies";

// Story-style overview. Plain language first; each claim keeps its technical backing one click
// away. Numbers marked "live" are read from /api/experiments/results (Stage 6-7 data); the rest
// are fixed facts of the study design taken from the stage reports.

/** Decorative animated hospital: beds fill and empty, patients queue, decisions appear. */
function MiniHospital() {
  const ward = (n: number, full: number, offset: number) =>
    Array.from({ length: n }, (_, i) => (
      <i key={i} className={`mh-bed${i < full ? " full" : ""}`}
        style={{ animationDelay: `${((i * 0.73 + offset) % 9).toFixed(2)}s` }} />
    ));
  return (
    <div className="mini-hosp" aria-hidden="true">
      <div className="mh-head"><span className="live-dot" /> scaled illustration · the real one is in Live hospital</div>
      <div className="mh-ward"><div className="mh-name"><span>Medical ward</span><span>72 beds</span></div>
        <div className="mh-beds">{ward(24, 14, 0)}</div></div>
      <div className="mh-ward"><div className="mh-name"><span>Surgical ward</span><span>70 beds</span></div>
        <div className="mh-beds">{ward(24, 12, 3)}</div></div>
      <div className="mh-ward"><div className="mh-name"><span>Intensive care</span><span>12 beds</span></div>
        <div className="mh-beds">{ward(12, 7, 5)}</div></div>
      <div className="mh-name"><span>Waiting for a bed</span></div>
      <div className="mh-queue">
        {[1, 3, 3, 4, 5, 2, 4].map((p, i) => (
          <i key={i} className="mh-q" style={{ background: PRIORITY_COLORS[p], animationDelay: `${i * 0.25}s` }} />
        ))}
      </div>
      <div className="mh-decision"><span>🤖</span><span className="typing">
        AI agent: <b>Critical patient → Intensive care</b></span></div>
    </div>
  );
}

const STEPS = [
  { title: "Build a virtual hospital", icon: "🏗️",
    text: "A computer model of an emergency department, a medical ward, a surgical ward and intensive care, with patients arriving, being treated and going home, minute by minute.",
    tech: ["discrete-event simulation", "checked against queueing theory (15/15)"] },
  { title: "Make it a digital twin", icon: "🪞",
    text: "Freeze the hospital at any moment, copy it, and play out many possible futures from that exact state.",
    tech: ["exact snapshots", "conditional resampling of stays"] },
  { title: "Let different decision-makers run it", icon: "🧠",
    text: "Hand the bed decisions to simple rules, to a mathematical optimiser and to an AI agent that learned by trial and error.",
    tech: ["5 tuned baselines", "MIP", "PPO written from scratch"] },
  { title: "Test fairly, then stress it", icon: "🧪",
    text: "Every policy faces exactly the same patients. Then we test situations the AI never saw: surges, staff shortages, closed beds.",
    tech: ["2,500 test episodes", "pre-registered statistics", "Holm correction"] },
  { title: "Explain every decision", icon: "💬",
    text: "Each decision is logged and can be explained in plain words, and you can ask \"what if?\" from any moment.",
    tech: ["decision log", "fact-checked explanations", "what-if engine"] },
];

export default function Overview() {
  const [data, setData] = useState<StudyResults | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api.health().then(() => setOnline(true)).catch(() => setOnline(false));
    api.results().then(setData).catch(() => setData(null));
  }, []);

  const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const vsLB = avg(data?.J_comparisons?.filter((c) => c.baseline === "load_balancing").map((c) => c.rel_diff_pct) ?? []);
  const vsPri = avg(data?.J_comparisons?.filter((c) => c.baseline === "priority").map((c) => c.rel_diff_pct) ?? []);
  const crnMax = Math.max(...(data?.crn_check?.map((c) => c.width_ratio_independent_over_paired) ?? [NaN]));
  const rhoA = data?.structure?.A?.spearman_rho ?? NaN;
  const f3 = data?.ablations?.find((c) => c.scenario === "A" && c.metric === "J_per_day" && c.baseline.endsWith("F3_uniform_priority_weights"));
  const f3pct = f3 ? (100 * f3.diff_mean) / f3.baseline_mean : NaN;

  return (
    <div>
      {/* ------------------------------------------------------------------ hero */}
      <section className="hero2">
        <span className="blob b1" /><span className="blob b2" /><span className="blob b3" />
        <div className="hero2-grid">
          <div>
            <Reveal><span className="pill"><span className="badge">Research</span> AI × hospital operations × simulation</span></Reveal>
            <Reveal delay={80}>
              <h1>Who gets the next hospital bed?<br /><span className="gradient-text">Can an AI decide better?</span></h1>
            </Reveal>
            <Reveal delay={160}>
              <p className="lead">
                NeuroTwin is a virtual hospital where an AI agent, classic rules and a mathematical optimiser
                compete to place patients into beds. Every one of them faces the same patients, in normal
                times and in crises the AI has never seen.
              </p>
            </Reveal>
            <Reveal delay={240}>
              <div className="row">
                <Link href="/simulation/?demo=1" className="btn lg glow">Watch it live <span className="arrow">→</span></Link>
                <Link href="/experiments/" className="btn lg ghost">See the results</Link>
                <span className="pill" title="Backend status">
                  <i className={`live-dot${online === false ? " off" : online === null ? " wait" : ""}`} />
                  {online ? "Simulator online" : online === false ? "Simulator waking up / offline" : "Connecting…"}
                </span>
              </div>
            </Reveal>
          </div>
          <Reveal kind="zoom" delay={200}><MiniHospital /></Reveal>
        </div>
        <Reveal delay={300}>
          <div className="ribbon">
            <div><div className="num"><CountUp value={2500} /></div><div className="lbl">test episodes simulated</div></div>
            <div><div className="num"><CountUp value={6} /></div><div className="lbl">decision-makers compared</div></div>
            <div><div className="num"><CountUp value={5} /></div><div className="lbl">situations, 4 of them unseen crises</div></div>
            <div><div className="num"><CountUp value={49} /> days</div><div className="lbl">simulated per episode</div></div>
          </div>
        </Reveal>
      </section>

      {/* ------------------------------------------------------------------ problem */}
      <section className="section">
        <Reveal><SectionHeader kicker="The problem" title="Three decisions hospitals make every hour"
          lead="When beds are scarce, every choice has a cost somewhere else. Fixed rules handle normal days well, but crises change the trade-offs." /></Reveal>
        <div className="grid grid-3">
          {[
            { icon: "🚑", title: "Who goes first?", text: "A critical patient and a patient who has waited 6 hours both need the last bed. Urgency or fairness?" },
            { icon: "🏥", title: "Which ward?", text: "The surgical ward has room, but the patient is medical. Place them in the wrong specialty now, or make them wait?" },
            { icon: "🛏️", title: "Keep a bed free?", text: "Filling every bed helps now, but leaves nothing for the next emergency ambulance. When is holding back worth it?" },
          ].map((f, i) => (
            <Reveal key={f.title} delay={i * 120}>
              <div className="card lift feature" style={{ height: "100%" }}>
                <div className="icon-badge">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ how it works */}
      <section className="section">
        <Reveal><SectionHeader kicker="How it works" title="From a virtual hospital to a fair contest"
          lead="Five building blocks, each written and tested from scratch." /></Reveal>
        <div className="steps">
          {STEPS.map((s, i) => (
            <Reveal key={s.title} kind="left" delay={i * 90}>
              <div className="step">
                <div className="step-num">{i + 1}</div>
                <div className="card lift">
                  <h3>{s.icon} {s.title}</h3>
                  <p>{s.text}</p>
                  <div className="chips">{s.tech.map((t) => <span key={t} className="chip">{t}</span>)}</div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ contenders */}
      <section className="section">
        <Reveal><SectionHeader kicker="The contenders" title="Six ways to decide who gets a bed"
          lead="Five were designed by people and tuned for the best possible score. One learned by itself." /></Reveal>
        <div className="grid grid-3">
          {POLICIES.map((p, i) => (
            <Reveal key={p.key} delay={(i % 3) * 100}>
              <div className={`card lift policy-card${p.key === "ppo" ? " star" : ""}`} style={{ ["--pc" as string]: p.color }}>
                <div className="pc-name"><i className="dot" style={{ background: p.color }} />{p.label}{p.key === "ppo" && " ⭐"}</div>
                <div className="pc-tech">{p.tech}</div>
                <p>{p.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ findings */}
      <section className="section">
        <Reveal><SectionHeader kicker="What we found" title="The headline results"
          lead="Measured on 50 fresh test runs per situation that were never used to tune anything. All numbers are from the simulation." /></Reveal>
        <div className="grid grid-3">
          <Reveal delay={0}><div className="card lift finding">
            <span className="big" style={{ color: "var(--series-1)" }}>≈ <CountUp value={Math.abs(vsLB)} digits={1} suffix="%" /></span>
            <h3>Ties the best hand-tuned rule</h3>
            <p>The AI performs as well as the balance-the-wards rule. The difference is too small to call a winner in any situation.</p>
            <TechDetails><p>Mean relative difference in cost J over 5 scenarios; no paired 95% CI excludes 0.</p></TechDetails>
          </div></Reveal>
          <Reveal delay={120}><div className="card lift finding">
            <span className="big" style={{ color: "var(--status-good)" }}><CountUp value={Math.abs(vsPri)} digits={1} suffix="%" /> lower</span>
            <h3>Beats the most-urgent-first rule</h3>
            <p>A lower overall cost score than the classic "most urgent first" rule, in every situation, including crises it never trained on.</p>
            <TechDetails><p>Mean relative difference in J vs Priority; CI excludes 0 in scenarios A, C and D.</p></TechDetails>
          </div></Reveal>
          <Reveal delay={240}><div className="card lift finding">
            <span className="big" style={{ color: "var(--violet)" }}>🧠 self-taught</span>
            <h3>It learned to keep beds free</h3>
            <p>Nobody told it to, but the AI holds a bed back for emergencies more often when only a few beds remain, exactly what queueing theory recommends.</p>
            <TechDetails><p>P(hold) vs free beds, scenario A: Spearman ρ = {Number.isFinite(rhoA) ? rhoA.toFixed(2) : "–"}, isotonic R² = 0.98. The pattern disappears under combined stress.</p></TechDetails>
          </div></Reveal>
          <Reveal delay={0}><div className="card lift finding">
            <span className="big" style={{ color: "var(--status-critical)" }}>1 in 5</span>
            <h3>But it can fail in new ways</h3>
            <p>One of five trained AI agents occasionally froze in a crisis, holding patients while beds emptied. No human-designed rule ever did that.</p>
            <TechDetails><p>5 of 1,250 AI test episodes collapsed (diversion above 50%), all from one agent (2% of its stressed runs), against 0 of 1,250 for the rules. This is an out-of-distribution failure.</p></TechDetails>
          </div></Reveal>
          <Reveal delay={120}><div className="card lift finding">
            <span className="big" style={{ color: "var(--series-2)" }}><CountUp value={crnMax} digits={1} suffix="×" /></span>
            <h3>Sharper measurements</h3>
            <p>Giving every decision-maker the exact same patients made our comparisons this many times more precise, so small differences become visible.</p>
            <TechDetails><p>Common random numbers: ratio of 95% CI widths, independent vs paired seeds (scenario A, J).</p></TechDetails>
          </div></Reveal>
          <Reveal delay={240}><div className="card lift finding">
            <span className="big" style={{ color: "var(--series-4)" }}>+<CountUp value={f3pct} digits={0} suffix="%" /></span>
            <h3>What it is rewarded for matters most</h3>
            <p>Teaching the AI that urgent patients matter more was the most important ingredient. Without it, cost rises by this much.</p>
            <TechDetails><p>Ablation F3 (uniform priority weights in the training reward), the only ablation significant after Holm over 80 comparisons.</p></TechDetails>
          </div></Reveal>
        </div>
        <Reveal><div className="row" style={{ justifyContent: "center", marginTop: 24 }}>
          <Link href="/experiments/" className="btn lg glow">Explore every result <span className="arrow">→</span></Link>
        </div></Reveal>
      </section>

      {/* ------------------------------------------------------------------ scenarios */}
      <section className="section">
        <Reveal><SectionHeader kicker="The stress test" title="One normal day, four crises"
          lead="The AI trained only on normal days. Everything else is a surprise, which is how we test whether it really generalises." /></Reveal>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
          {Object.entries(SCENARIO_INFO).map(([k, s], i) => (
            <Reveal key={k} delay={i * 90} kind="zoom">
              <div className={`card lift scen${k === "A" ? " train" : ""}`}>
                <div className="scen-icon">{s.icon}</div>
                <div className="scen-letter">SCENARIO {k}{k === "A" ? " · TRAINING" : " · UNSEEN"}</div>
                <h3>{s.short}</h3>
                <p>{s.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ honesty */}
      <section className="section">
        <Reveal><SectionHeader kicker="Read this first" title="What this is, and what it isn't" /></Reveal>
        <Reveal>
          <div className="honest">
            <div className="card is">
              <h3 style={{ margin: 0 }}>✅ It is</h3>
              <ul>
                <li>A research study in a carefully validated simulation</li>
                <li>A fair, pre-registered comparison of AI, rules and optimisation</li>
                <li>Honest about failures and negative results</li>
                <li>Fully reproducible: every number can be regenerated from the code</li>
              </ul>
            </div>
            <div className="card isnt">
              <h3 style={{ margin: 0 }}>🚫 It is not</h3>
              <ul>
                <li>A tool for real hospitals or real patients</li>
                <li>Calibrated to any real hospital's data</li>
                <li>Evidence that AI improves real-world care</li>
                <li>A medical device or clinical decision support</li>
              </ul>
            </div>
          </div>
        </Reveal>
      </section>

      {/* ------------------------------------------------------------------ CTA */}
      <section className="section">
        <Reveal kind="zoom">
          <div className="hero2" style={{ textAlign: "center" }}>
            <span className="blob b1" /><span className="blob b2" />
            <h2 className="display" style={{ fontSize: "clamp(26px, 3.4vw, 40px)", margin: "0 0 12px", letterSpacing: "-0.03em" }}>
              See the AI make decisions, live
            </h2>
            <p className="secondary" style={{ maxWidth: 560, margin: "0 auto 22px" }}>
              Start a simulated hospital, watch beds fill, click any decision to see why it was made,
              and ask &quot;what if 30% more patients arrived right now?&quot;
            </p>
            <div className="row" style={{ justifyContent: "center" }}>
              <Link href="/simulation/?demo=1" className="btn lg glow">Launch the live hospital <span className="arrow">→</span></Link>
            </div>
          </div>
        </Reveal>
      </section>
    </div>
  );
}
