"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import HeroScene from "@/components/HeroScene";
import { Icon, type IconName } from "@/components/icons";
import { CountUp, Reveal, Scene } from "@/components/motion";
import { SectionHeader, TechDetails } from "@/components/ui";
import { api, type StudyResults } from "@/lib/api";
import { POLICIES, SCENARIO_INFO } from "@/lib/policies";

// Story-style overview. Plain language first; each claim keeps its technical backing one click
// away. Numbers marked "live" are read from /api/experiments/results (Stage 6-7 data); the rest
// are fixed facts of the study design taken from the stage reports and the code
// (config/ppo_default.json, neurotwin/rl/features.py, neurotwin/rl/actions.py).

/** Animated architecture of the RL agent: inputs -> actor network -> masked actions -> hospital
 *  -> reward -> back into training. Layer sizes and counts are the real ones. */
function AgentDiagram() {
  const inputs = ["Ward fullness ×3", "Waiting by type ×4", "Staff available ×2", "Time of day/week ×4",
    "Recent arrivals ×3", "Urgency & waits ×3", "Turned away, wrong ward ×2"];
  const actions = ["Wait (keep beds free)", "Most urgent → own ward", "Longest wait → own ward",
    "Send to another ward", "Admit to ICU", "ICU → ward (step down)", "Ward → ICU (escalate)"];
  const col = (x: number, n: number, y0: number, step: number) => Array.from({ length: n }, (_, i) => [x, y0 + i * step] as const);
  const L0 = col(330, 7, 70, 40);
  const L1 = col(450, 8, 50, 38);
  const L2 = col(570, 8, 50, 38);
  const L3 = col(690, 7, 70, 40);
  const edges = (a: readonly (readonly [number, number])[], b: readonly (readonly [number, number])[]) =>
    a.flatMap(([x1, y1], i) => b.map(([x2, y2], j) => <line key={`${x1}-${i}-${j}`} x1={x1} y1={y1} x2={x2} y2={y2} className="ad-edge" />));
  return (
    <svg className="agent-diagram" viewBox="0 0 1100 470" role="img"
      aria-label="RL agent architecture: 21 inputs, two hidden layers of 64 neurons, 7 masked actions, trained by reward from the hospital simulation">
      <rect x="10" y="30" width="250" height="310" rx="16" className="ad-box" />
      <text x="30" y="60" className="ad-title">What it sees</text>
      <text x="30" y="80" className="ad-muted">21 numbers describing the hospital</text>
      {inputs.map((t, i) => <text key={t} x="30" y={112 + i * 32} className="ad-text">{t}</text>)}
      <path d="M262 185 C 290 185, 300 185, 318 185" className="ad-flow" />

      {edges(L0, L1)}{edges(L1, L2)}{edges(L2, L3)}
      {L0.map(([x, y], i) => <circle key={`a${i}`} cx={x} cy={y} r="8" className="ad-node" style={{ animationDelay: `${i * 0.12}s` }} />)}
      {L1.map(([x, y], i) => <circle key={`b${i}`} cx={x} cy={y} r="8" className="ad-node" style={{ animationDelay: `${0.4 + i * 0.1}s` }} />)}
      {L2.map(([x, y], i) => <circle key={`c${i}`} cx={x} cy={y} r="8" className="ad-node c" style={{ animationDelay: `${0.8 + i * 0.1}s` }} />)}
      {L3.map(([x, y], i) => <circle key={`d${i}`} cx={x} cy={y} r="8" className="ad-node o" style={{ animationDelay: `${1.2 + i * 0.12}s` }} />)}
      <text x="330" y="365" textAnchor="middle" className="ad-muted">21 inputs</text>
      <text x="450" y="365" textAnchor="middle" className="ad-muted">64 neurons</text>
      <text x="570" y="365" textAnchor="middle" className="ad-muted">64 neurons</text>
      <text x="690" y="365" textAnchor="middle" className="ad-muted">7 scores</text>
      <text x="510" y="385" textAnchor="middle" className="ad-text">21 numbers in → one score per possible move → blocked moves removed → highest score wins</text>
      <text x="510" y="22" textAnchor="middle" className="ad-title">Policy network (actor)</text>

      <path d="M705 185 C 730 185, 740 185, 768 185" className="ad-flow" />
      <rect x="770" y="30" width="320" height="310" rx="16" className="ad-box hl" />
      <text x="790" y="60" className="ad-title">What it can do</text>
      <text x="790" y="80" className="ad-muted">impossible choices are blocked (masked)</text>
      {actions.map((t, i) => <text key={t} x="790" y={112 + i * 32} className="ad-text">{i + 1}. {t}</text>)}

      <rect x="310" y="395" width="480" height="62" rx="14" className="ad-box hl" />
      <text x="550" y="421" textAnchor="middle" className="ad-title">Hospital simulation</text>
      <text x="550" y="443" textAnchor="middle" className="ad-muted">reward = −(urgency-weighted waiting + turned away + wrong ward)</text>
      <path d="M930 342 C 930 420, 860 426, 792 426" className="ad-flow" />
      <path d="M308 426 C 150 426, 130 380, 140 342" className="ad-flow back" />
      <text x="40" y="462" className="ad-muted">learns from the reward</text>
      <text x="945" y="392" className="ad-muted">acts on the hospital</text>
    </svg>
  );
}

const STEPS: { title: string; icon: IconName; text: string; tech: string[] }[] = [
  { title: "Build a virtual hospital", icon: "build",
    text: "A computer model of an emergency department, a medical ward, a surgical ward and intensive care, with patients arriving, being treated and going home, minute by minute.",
    tech: ["discrete-event simulation", "checked against queueing theory (15/15)"] },
  { title: "Make it a digital twin", icon: "mirror",
    text: "Freeze the hospital at any moment, copy it, and play out many possible futures from that exact state.",
    tech: ["exact snapshots", "conditional resampling of stays"] },
  { title: "Let different decision-makers run it", icon: "brain",
    text: "Hand the bed decisions to simple rules, to a mathematical optimiser, and to a reinforcement-learning agent trained from scratch.",
    tech: ["5 tuned baselines", "MIP", "PPO written from scratch"] },
  { title: "Test fairly, then stress it", icon: "flask",
    text: "Every decision-maker faces exactly the same patients. Then we test situations the agent never saw: surges, staff shortages, closed beds.",
    tech: ["2,500 test episodes", "pre-registered statistics", "Holm correction"] },
  { title: "Explain every decision", icon: "chat",
    text: "Each decision is logged and can be explained in plain words, and you can ask \"what if?\" from any moment.",
    tech: ["decision log", "fact-checked explanations", "what-if engine"] },
];

const SPEC: { k: string; v: string; d: string; icon: IconName }[] = [
  { k: "Algorithm", v: "PPO", d: "Proximal Policy Optimisation, implemented from scratch in PyTorch (networks, losses, training loop).", icon: "cpu" },
  { k: "Network", v: "2 × 64 neurons", d: "Separate actor (chooses) and critic (judges the situation) networks with tanh activations.", icon: "network" },
  { k: "Training", v: "3 million decisions", d: "8 simulated hospitals in parallel, normal days only, trained on an ordinary CPU (no GPU needed).", icon: "repeat" },
  { k: "Time-aware", v: "γ = 0.95 / hour", d: "Rewards are discounted by real simulated time between decisions (semi-Markov GAE, λ = 0.9).", icon: "clock" },
  { k: "Safe by design", v: "Action masking", d: "It can only pick choices that are physically possible at that moment.", icon: "shield" },
  { k: "The model", v: "1 network, 5 copies", d: "One small neural network (about 11,700 learned numbers). The results use 5 copies of it trained from different random starts, so one lucky run can't decide the outcome.", icon: "layers" },
  { k: "Verified", v: "vs stable-baselines3", d: "My PPO was checked against the reference PPO (stable-baselines3) and MaskablePPO (sb3-contrib) on CartPole and on this hospital.", icon: "check" },
];

const JOURNEY: { tag: string; title: string; text: string; result: string; icon: IconName; tone: string }[] = [
  { tag: "Version 1", title: "Learning from scratch", icon: "cpu", tone: "",
    text: "A neural network trained with PPO, written from scratch, on normal days only. It beat four of the five hand-designed rules.",
    result: "Tied the best rule (balance-the-wards); no significant difference" },
  { tag: "Finding", title: "A failure no rule has", icon: "alert", tone: "bad",
    text: "Under crises one of the five trained copies sometimes froze: it kept beds empty while ambulances were turned away.",
    result: "5 collapsed runs out of 1,250; the rules had none" },
  { tag: "Fix", title: "A one-line safety guard", icon: "shield", tone: "good",
    text: "Never keep a bed empty while the emergency department is full. No retraining, no tuned number.",
    result: "Collapses 5 → 0, at a small cost in one crisis" },
  { tag: "Version 2", title: "Learn from the best, then beat it", icon: "trend", tone: "win",
    text: "The agent first copied the best rule, then improved on it with reinforcement learning; five copies vote. Tested once on 50 runs never used before.",
    result: "8.5% lower cost than the best rule on normal days; ties in crises" },
];

export default function Overview() {
  const [data, setData] = useState<StudyResults | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api.health().then(() => setOnline(true)).catch(() => setOnline(false));
    api.results().then(setData).catch(() => setData(null));
  }, []);

  const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const winA = data?.beat_rule?.primary_vs_load_balancing.find((c) => c.scenario === "A");
  const winPct = winA ? (100 * -winA.diff_mean) / winA.baseline_mean : NaN;
  const vsLB = avg(data?.J_comparisons?.filter((c) => c.baseline === "load_balancing").map((c) => c.rel_diff_pct) ?? []);
  const vsPri = avg(data?.J_comparisons?.filter((c) => c.baseline === "priority").map((c) => c.rel_diff_pct) ?? []);
  const crnMax = Math.max(...(data?.crn_check?.map((c) => c.width_ratio_independent_over_paired) ?? [NaN]));
  const rhoA = data?.structure?.A?.spearman_rho ?? NaN;
  const f3 = data?.ablations?.find((c) => c.scenario === "A" && c.metric === "J_per_day" && c.baseline.endsWith("F3_uniform_priority_weights"));
  const f3pct = f3 ? (100 * f3.diff_mean) / f3.baseline_mean : NaN;

  return (
    <div>
      {/* ------------------------------------------------------------------ hero */}
      <Scene className="hero-scene">
        <div className="hero2">
          <span className="blob b1" /><span className="blob b2" /><span className="blob b3" />
          <div className="hero2-grid">
            <div>
              <Reveal><span className="pill"><span className="badge">Research</span> Reinforcement learning × hospital operations</span></Reveal>
              <Reveal delay={80}>
                <h1>Who gets the next hospital bed?<br /><span className="gradient-text">Can a learning agent decide better?</span></h1>
              </Reveal>
              <Reveal delay={160}>
                <p className="lead">
                  NeuroTwin is a virtual hospital where a reinforcement-learning agent I built from scratch competes with
                  classic rules and a mathematical optimiser to place patients into beds. All of them face the same
                  patients, in normal times and in crises the agent has never seen.
                </p>
              </Reveal>
              <Reveal delay={240}>
                <div className="row">
                  <Link href="/simulation/?demo=1" className="btn lg glow"><Icon name="play" size={16} /> Watch it live <span className="arrow"><Icon name="arrow" size={16} /></span></Link>
                  <Link href="/experiments/" className="btn lg ghost"><Icon name="chart" size={16} /> See the results</Link>
                  <span className="pill" title="Backend status">
                    <i className={`live-dot${online === false ? " off" : online === null ? " wait" : ""}`} />
                    {online ? "Simulator online" : online === false ? "Simulator waking up / offline" : "Connecting…"}
                  </span>
                </div>
              </Reveal>
            </div>
            <Reveal kind="zoom" delay={200}><HeroScene /></Reveal>
          </div>
          <Reveal delay={300}>
            <div className="ribbon">
              <div><div className="num"><CountUp value={2500} /></div><div className="lbl">test episodes simulated</div></div>
              <div><div className="num"><CountUp value={6} /></div><div className="lbl">decision-makers compared</div></div>
              <div><div className="num"><CountUp value={5} /></div><div className="lbl">situations, 4 of them unseen crises</div></div>
              <div><div className="num"><CountUp value={3} />M</div><div className="lbl">training decisions per agent</div></div>
            </div>
          </Reveal>
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ problem */}
      <Scene>
        <Reveal><SectionHeader kicker="The problem" title="Three decisions hospitals make every hour"
          lead="When beds are scarce, every choice has a cost somewhere else. Fixed rules handle normal days well, but crises change the trade-offs." /></Reveal>
        <div className="grid grid-3">
          {([
            { icon: "ambulance", tone: "red", title: "Who goes first?", text: "A critical patient and a patient who has waited 6 hours both need the last bed. Urgency or fairness?" },
            { icon: "hospital", tone: "", title: "Which ward?", text: "The surgical ward has room, but the patient is medical. Place them in the wrong specialty now, or make them wait?" },
            { icon: "bed", tone: "green", title: "Keep a bed free?", text: "Filling every bed helps now, but leaves nothing for the next emergency ambulance. When is holding back worth it?" },
          ] as { icon: IconName; tone: string; title: string; text: string }[]).map((f, i) => (
            <Reveal key={f.title} delay={i * 120}>
              <div className="card lift feature" style={{ height: "100%" }}>
                <div className={`icon-badge ${f.tone}`}><Icon name={f.icon} size={24} /></div>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ how it works */}
      <Scene>
        <Reveal><SectionHeader kicker="How it works" title="From a virtual hospital to a fair contest"
          lead="Five building blocks, each written and tested from scratch." /></Reveal>
        <div className="steps">
          {STEPS.map((s, i) => (
            <Reveal key={s.title} kind="left" delay={i * 90}>
              <div className="step">
                <div className="step-num"><Icon name={s.icon} size={22} /></div>
                <div className="card lift">
                  <h3><span className="muted" style={{ fontSize: 13, marginRight: 8 }}>Step {i + 1}</span>{s.title}</h3>
                  <p>{s.text}</p>
                  <div className="chips">{s.tech.map((t) => <span key={t} className="chip">{t}</span>)}</div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ the agent */}
      <Scene>
        <Reveal><SectionHeader kicker="Inside the RL agent" title="What the agent is, and how I built it"
          lead="Not a chatbot and not a pre-trained model. It is a small neural network that learned bed allocation purely by trial and error inside the simulator, using reinforcement learning I implemented from scratch." /></Reveal>
        <Reveal kind="zoom">
          <div className="card lift" style={{ padding: "18px 18px 8px" }}>
            <AgentDiagram />
          </div>
        </Reveal>
        <div className="spec">
          {SPEC.map((s, i) => (
            <Reveal key={s.k} delay={(i % 3) * 100}>
              <div className="card lift" style={{ height: "100%" }}>
                <div className="k"><Icon name={s.icon} size={14} /> {s.k}</div>
                <div className="v">{s.v}</div>
                <div className="d">{s.d}</div>
              </div>
            </Reveal>
          ))}
        </div>
        <Reveal>
          <TechDetails label="How training works, step by step">
            <p>1. The 21 numbers are inputs, the 7 moves are outputs: it does not pick 7 of the 21. The network reads all 21
              numbers and outputs one score for each of the 7 possible moves. Moves that are impossible right now (for example,
              no free ICU bed) are masked out, the scores become probabilities, and in use it takes the most likely move.
              During training it samples from the probabilities instead, which is how it explores.</p>
            <p>2. The simulator advances to the next decision moment and returns a reward: minus the urgency-weighted waiting
              accumulated in between, minus a penalty per turned-away ambulance and per wrong-ward placement.</p>
            <p>3. After 2,048 decisions across 8 parallel hospitals, advantages are computed with semi-Markov GAE (discount γ^Δt per
              simulated hour), and the actor and critic are updated for 10 epochs with PPO&apos;s clipped objective.</p>
            <p>4. Repeat until 3 million decisions. Hyperparameters were chosen on validation runs only; test runs stayed sealed.</p>
          </TechDetails>
        </Reveal>
      </Scene>

      {/* ------------------------------------------------------------------ journey */}
      <Scene>
        <Reveal><SectionHeader kicker="How the agent evolved" title="One agent, improved step by step"
          lead="Each step was planned in writing before it ran, and each one is reported, including what went wrong." /></Reveal>
        <div className="journey">
          {JOURNEY.map((j, i) => (
            <Reveal key={j.title} kind={i % 2 ? "right" : "left"} delay={i * 80}>
              <div className={`journey-step ${j.tone}`}>
                <div className="journey-dot"><Icon name={j.icon} size={18} /></div>
                <div className="card lift">
                  <div className="journey-tag">{j.tag}</div>
                  <h3>{j.title}</h3>
                  <p>{j.text}</p>
                  <div className="journey-result">{j.result}</div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ contenders */}
      <Scene>
        <Reveal><SectionHeader kicker="The contenders" title="Six ways to decide who gets a bed"
          lead="Five were designed by people and tuned for the best possible score. One learned by itself." /></Reveal>
        <div className="grid grid-3">
          {POLICIES.map((p, i) => (
            <Reveal key={p.key} delay={(i % 3) * 100}>
              <div className={`card lift policy-card${p.key === "ppo" ? " star" : ""}`} style={{ ["--pc" as string]: p.color }}>
                <div className="pc-name"><i className="dot" style={{ background: p.key === "ppo" ? "var(--violet)" : p.color }} />
                  {p.key === "ppo" ? "RL agent" : p.label}
                  {p.key === "ppo" && <Icon name="sparkle" size={16} style={{ color: "var(--accent)" }} />}</div>
                <div className="pc-tech">{p.key === "ppo" ? "PPO from scratch, now in version 2" : p.tech}</div>
                <p>{p.key === "ppo" ? "A neural network that learns by trial and error. Version 2 first learned from the best rule, then improved past it." : p.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ findings */}
      <Scene>
        <Reveal><SectionHeader kicker="What we found" title="The headline results"
          lead="Measured on 50 fresh test runs per situation that were never used to tune anything. All numbers are from the simulation." /></Reveal>
        <div className="grid grid-3">
          <Reveal delay={0}><div className="card lift finding">
            <span className="big" style={{ color: "var(--series-1)" }}><CountUp value={winPct} digits={1} suffix="%" /> lower</span>
            <h3>Beats the best hand-tuned rule</h3>
            <p>RL agent v2 learned from the balance-the-wards rule, then improved on it: lower cost on normal days, tested on 50 fresh runs, and no significant loss in any crisis.</p>
            <TechDetails><p>Imitation-initialised PPO, 5-agent vote, pre-registered (RESEARCH.md v2.8) and tested once on a fresh seed block: −8.5% J in scenario A (Holm p = 0.0005); B–D tie;
              E +3.5% (not significant after Holm). The first agent tied the rule ({Number.isFinite(vsLB) ? `${vsLB > 0 ? "+" : ""}${vsLB.toFixed(1)}%` : "–"} on average).</p></TechDetails>
          </div></Reveal>
          <Reveal delay={120}><div className="card lift finding">
            <span className="big" style={{ color: "var(--status-good)" }}><CountUp value={Math.abs(vsPri)} digits={1} suffix="%" /> lower</span>
            <h3>Beats the most-urgent-first rule</h3>
            <p>A lower overall cost score than the classic "most urgent first" rule, in every situation, including crises it never trained on.</p>
            <TechDetails><p>Mean relative difference in J vs Priority; CI excludes 0 in scenarios A, C and D.</p></TechDetails>
          </div></Reveal>
          <Reveal delay={240}><div className="card lift finding">
            <span className="big" style={{ color: "var(--violet)" }}>Self-taught</span>
            <h3>It learned to keep beds free</h3>
            <p>Nobody told it to, but the agent holds a bed back for emergencies more often when only a few beds remain, exactly what queueing theory recommends.</p>
            <TechDetails><p>P(hold) vs free beds, scenario A: Spearman ρ = {Number.isFinite(rhoA) ? rhoA.toFixed(2) : "–"}, isotonic R² = 0.98. The pattern disappears under combined stress.</p></TechDetails>
          </div></Reveal>
          <Reveal delay={0}><div className="card lift finding">
            <span className="big" style={{ color: "var(--status-good)" }}>5 → 0</span>
            <h3>It found a new way to fail, then a guard fixed it</h3>
            <p>One of five trained agents occasionally froze in a crisis, keeping beds empty while ambulances were turned away. A one-line safety guard (never hold a bed while the ED is full) removed every collapse.</p>
            <TechDetails><p>Without the guard: 5 of 1,250 test episodes collapsed (diversion above 50%), all from one agent, against 0 of 1,250 for the rules.
              With it: 0 of 1,250. No retraining. Cost: +4.4% in the staff-shortage scenario (significant); elsewhere no significant change. Pre-registered as RESEARCH.md v2.7.</p></TechDetails>
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
            <p>Teaching the agent that urgent patients matter more was the most important ingredient. Without it, cost rises by this much.</p>
            <TechDetails><p>Ablation F3 (uniform priority weights in the training reward), the only ablation significant after Holm over 80 comparisons.</p></TechDetails>
          </div></Reveal>
        </div>
        <Reveal><div className="row" style={{ justifyContent: "center", marginTop: 24 }}>
          <Link href="/experiments/" className="btn lg glow">Explore every result <span className="arrow"><Icon name="arrow" size={16} /></span></Link>
        </div></Reveal>
      </Scene>

      {/* ------------------------------------------------------------------ scenarios */}
      <Scene>
        <Reveal><SectionHeader kicker="The stress test" title="One normal day, four crises"
          lead="The agent trained only on normal days. Everything else is a surprise, which is how we test whether it really generalises." /></Reveal>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
          {Object.entries(SCENARIO_INFO).map(([k, s], i) => (
            <Reveal key={k} delay={i * 90} kind="zoom">
              <div className={`card lift scen${k === "A" ? " train" : ""}`}>
                <div className={`icon-badge${k === "E" ? " red" : k === "A" ? " green" : ""}`}><Icon name={s.icon} size={24} /></div>
                <div className="scen-letter">SCENARIO {k}{k === "A" ? " · TRAINING" : " · UNSEEN"}</div>
                <h3>{s.short}</h3>
                <p>{s.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ honesty */}
      <Scene>
        <Reveal><SectionHeader kicker="Read this first" title="What this is, and what it isn't" /></Reveal>
        <div className="honest">
          <Reveal kind="left">
            <div className="card lift is" style={{ height: "100%" }}>
              <h3 style={{ margin: 0, display: "flex", gap: 8, alignItems: "center" }}><Icon name="check" style={{ color: "var(--status-good)" }} /> It is</h3>
              <ul>
                <li>A research study in a carefully validated simulation</li>
                <li>A fair, pre-registered comparison of RL, rules and optimisation</li>
                <li>Honest about failures and negative results</li>
                <li>Fully reproducible: every number can be regenerated from the code</li>
              </ul>
            </div>
          </Reveal>
          <Reveal kind="right">
            <div className="card lift isnt" style={{ height: "100%" }}>
              <h3 style={{ margin: 0, display: "flex", gap: 8, alignItems: "center" }}><Icon name="x" style={{ color: "var(--status-critical)" }} /> It is not</h3>
              <ul>
                <li>A tool for real hospitals or real patients</li>
                <li>Calibrated to any real hospital's data</li>
                <li>Evidence that RL improves real-world care</li>
                <li>A medical device or clinical decision support</li>
              </ul>
            </div>
          </Reveal>
        </div>
      </Scene>

      {/* ------------------------------------------------------------------ CTA */}
      <Scene>
        <Reveal kind="zoom">
          <div className="hero2" style={{ textAlign: "center" }}>
            <span className="blob b1" /><span className="blob b2" />
            <h2 className="display" style={{ fontSize: "clamp(26px, 3.4vw, 40px)", margin: "0 0 12px", letterSpacing: "-0.03em" }}>
              See the agent make decisions, live
            </h2>
            <p className="secondary" style={{ maxWidth: 560, margin: "0 auto 22px" }}>
              Watch ambulances arrive and patients move to their beds, click any decision to see why it was made,
              or race the decision-makers against each other on exactly the same patients.
            </p>
            <div className="row" style={{ justifyContent: "center" }}>
              <Link href="/simulation/?demo=1" className="btn lg glow"><Icon name="play" size={16} /> Launch the live hospital <span className="arrow"><Icon name="arrow" size={16} /></span></Link>
              <Link href="/race/" className="btn lg ghost"><Icon name="flag" size={16} /> Start a race</Link>
            </div>
          </div>
        </Reveal>
      </Scene>
    </div>
  );
}
