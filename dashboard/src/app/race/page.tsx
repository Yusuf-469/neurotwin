"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Icon } from "@/components/icons";
import { liveDot, Reveal, SplitWords } from "@/components/motion";
import { ConditionsSliders, SituationSelect, describe } from "@/components/setup";
import { Legend, TechDetails } from "@/components/ui";
import { RunLost, ServerUnavailable, api, type Conditions, type RaceLane, type RaceStartBody, type RaceState } from "@/lib/api";
import { LIVE_POLICIES, POLICY_GROUPS, SCENARIO_INFO, policyMeta } from "@/lib/policies";

// Live race: the server branches one hospital state into identical copies (same future
// patients), one per decision-maker, and advances them together. Every number on this page comes
// back from the server as it computes; the car positions are a visual of the running cost.

const LANE_H = 86;
const LANE_H_PHONE = 142;

/** Lane pitch in px: lanes stack their track and stats on narrow screens. */
function useLaneHeight(): number {
  const [h, setH] = useState(LANE_H);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 760px)");
    const set = () => setH(mq.matches ? LANE_H_PHONE : LANE_H);
    set();
    mq.addEventListener("change", set);
    return () => mq.removeEventListener("change", set);
  }, []);
  return h;
}
const TICK_MS = 1100;
const DEFAULT_COMPETITORS = ["ppo_v3", "load_balancing", "priority", "fcfs"];

type Phase = "setup" | "loading" | "countdown" | "racing" | "paused" | "finished";
interface Commentary { id: number; t: number; text: string; tone?: "lead" | "bad" | "info" }

const label = (k: string) => policyMeta(k).label;
const dayLabel = (h: number) => `day ${(h / 24).toFixed(h % 24 === 0 ? 0 : 1)}`;

function RaceCar({ color, n }: { color: string; n: number }) {
  return (
    <svg className="race-car" viewBox="0 0 64 28" width="64" height="28" aria-hidden="true">
      <path d="M4 20 L8 12 L22 10 L30 4 L44 4 L50 10 L60 13 L61 20 Z" fill={color} />
      <path d="M31 6 L43 6 L47 10 L28 10 Z" fill="#cfe6ff" opacity=".85" />
      <rect x="2" y="17" width="60" height="3" rx="1.5" fill="rgba(0,0,0,.25)" />
      <circle cx="16" cy="21" r="5" fill="#1d1d1d" /><circle cx="16" cy="21" r="2" fill="#9a9a9a" />
      <circle cx="49" cy="21" r="5" fill="#1d1d1d" /><circle cx="49" cy="21" r="2" fill="#9a9a9a" />
      <text x="37" y="17" textAnchor="middle" fontSize="7" fontWeight="800" fill="#fff">{n}</text>
    </svg>
  );
}

export default function RacePage() {
  const laneH = useLaneHeight();
  const [available, setAvailable] = useState<string[]>([]);
  const [scenario, setScenario] = useState("A");
  const [custom, setCustom] = useState<Conditions>({ arrival_multiplier: 1.3, staff_multiplier: 0.9, capacity_multiplier: 1.0 });
  const [seed, setSeed] = useState(3);
  const [startMode, setStartMode] = useState<"busy" | "session">("busy");
  const [fromSid, setFromSid] = useState("");
  const [days, setDays] = useState(3);
  const [picked, setPicked] = useState<string[]>(DEFAULT_COMPETITORS);
  const [phase, setPhase] = useState<Phase>("setup");
  const [count, setCount] = useState(3);
  const [rid, setRid] = useState("");
  const [race, setRace] = useState<RaceState | null>(null);
  const [meta, setMeta] = useState<{ label: string; setupMs: number; lastMs: number }>({ label: "", setupMs: 0, lastMs: 0 });
  const [notes, setNotes] = useState<Commentary[]>([]);
  const [error, setError] = useState("");
  const busy = useRef(false);
  const [reconnecting, setReconnecting] = useState("");
  const startBody = useRef<RaceStartBody | null>(null);
  const raceRef = useRef<RaceState | null>(null);
  const recovering = useRef(false);
  const leader = useRef("");
  const turned = useRef<Record<string, number>>({});
  const noteId = useRef(0);

  useEffect(() => {
    api.policies().then((p) => {
      setAvailable(p.policies);
      setPicked((cur) => cur.filter((k) => p.policies.includes(k)));
    }).catch((e) => setError(`Simulation server not reachable yet (it may be waking up): ${e}`));
    const q = new URLSearchParams(window.location.search);
    const sid = q.get("from");
    if (sid) { setFromSid(sid); setStartMode("session"); }
  }, []);

  const say = (t: number, text: string, tone?: Commentary["tone"]) =>
    setNotes((n) => [{ id: ++noteId.current, t, text, tone }, ...n].slice(0, 9));

  const absorb = useCallback((s: RaceState) => {
    const lead = s.lanes.find((l) => l.rank === 1);
    if (lead && s.elapsed_hours > 0 && lead.name !== leader.current) {
      say(s.elapsed_hours, leader.current ? `${label(lead.name)} takes the lead!` : `${label(lead.name)} leads after the first stretch`, "lead");
      leader.current = lead.name;
    }
    for (const l of s.lanes) {
      const before = turned.current[l.name] ?? 0;
      if (l.turned_away > before) say(s.elapsed_hours, `${label(l.name)} had to turn away ${l.turned_away - before} ambulance${l.turned_away - before > 1 ? "s" : ""}`, "bad");
      turned.current[l.name] = l.turned_away;
    }
    setRace(s);
  }, []);

  const start = async () => {
    setError("");
    setNotes([]);
    leader.current = "";
    turned.current = {};
    setPhase("loading");
    try {
      const body: RaceStartBody = {
        competitors: picked, scenario, seed, start: startMode, horizon_hours: days * 24,
        ...(scenario === "custom" ? { custom } : {}), ...(startMode === "session" ? { session_id: fromSid } : {}),
      };
      startBody.current = body;
      const r = await api.raceStart(body);
      setRid(r.race_id);
      setRace(r);
      setMeta({ label: r.start_label, setupMs: r.setup_ms, lastMs: 0 });
      setPhase("countdown");
      setCount(3);
    } catch (e) {
      setError(String(e));
      setPhase("setup");
    }
  };

  // 3-2-1 countdown, then go.
  useEffect(() => {
    if (phase !== "countdown") return;
    if (count === 0) {
      say(0, "Lights out! Every decision-maker gets exactly the same patients from here.", "info");
      setPhase("racing");
      return;
    }
    const t = setTimeout(() => setCount((c) => c - 1), 750);
    return () => clearTimeout(t);
  }, [phase, count]);

  const stepHours = days <= 3 ? 6 : days <= 7 ? 12 : 24;
  const tick = useCallback(async () => {
    if (!rid || busy.current) return;
    busy.current = true;
    try {
      const s = await api.raceStep(rid, stepHours);
      setMeta((m) => ({ ...m, lastMs: s.compute_ms ?? 0 }));
      absorb(s);
      if (s.finished) {
        const win = s.lanes.find((l) => l.rank === 1);
        if (win) say(s.elapsed_hours, `Chequered flag: ${label(win.name)} wins this race`, "lead");
        setPhase("finished");
      }
    } catch (e) {
      setPhase("paused");
      if (e instanceof ServerUnavailable || e instanceof RunLost) recover();
      else setError(String(e));
    } finally {
      busy.current = false;
    }
  }, [rid, stepHours, absorb]);

  useEffect(() => { raceRef.current = race; }, [race]);

  // Reconnect after the server sleeps or restarts; a lost race is recreated with the same settings
  // (identical patients) and replayed to the same point, then continues.
  const recover = async () => {
    if (recovering.current) return;
    recovering.current = true;
    setReconnecting("Lost contact with the simulation server (it is waking up or restarting). Reconnecting…");
    try {
      const deadline = Date.now() + 180_000;
      let up = false;
      while (!up && Date.now() < deadline) {
        try { await api.health(); up = true; } catch { await new Promise((r) => setTimeout(r, 3000)); }
      }
      if (!up) throw new Error("The simulation server did not come back within 3 minutes. Please refresh the page later.");
      const done = raceRef.current?.elapsed_hours ?? 0;
      try {
        await api.raceState(rid);
      } catch (e) {
        if (!(e instanceof RunLost) || !startBody.current || startBody.current.start === "session") throw e;
        setReconnecting(`The server restarted, so the race is being restored at ${dayLabel(done)}…`);
        const r = await api.raceStart(startBody.current);
        let st: RaceState = r;
        while (st.elapsed_hours < done - 1e-6) st = await api.raceStep(r.race_id, Math.min(48, done - st.elapsed_hours));
        setRid(r.race_id);
        setRace(st);
        say(done, "Server restarted: race restored at the same point", "info");
      }
      setReconnecting("");
      setPhase("racing");
    } catch (e) {
      setReconnecting("");
      setError(String(e));
    } finally {
      recovering.current = false;
    }
  };

  useEffect(() => {
    if (phase !== "racing") return;
    tick();
    const t = setInterval(tick, TICK_MS);
    return () => clearInterval(t);
  }, [phase, tick]);

  const toggle = (k: string) =>
    setPicked((p) => (p.includes(k) ? p.filter((x) => x !== k) : p.length >= 6 ? p : [...p, k]));

  const lanes: RaceLane[] = race ? [...race.lanes].sort((a, b) => a.name.localeCompare(b.name)) : [];
  const best = race ? Math.min(...race.lanes.map((l) => l.cost)) : 0;
  const progress = race ? race.elapsed_hours / race.horizon_hours : 0;
  const chart = race && race.lanes[0]
    ? race.lanes[0].trace.map((_, i) => {
      const row: Record<string, number> = { day: +(race.lanes[0].trace[i][0] / 24).toFixed(2) };
      for (const l of race.lanes) row[l.name] = Math.round(l.trace[i]?.[1] ?? 0);
      return row;
    })
    : [];
  const podium = race && phase === "finished" ? [...race.lanes].sort((a, b) => a.rank - b.rank) : [];
  const runnerUp = podium[1];
  const margin = podium[0] && runnerUp && runnerUp.cost > 0 ? (100 * (runnerUp.cost - podium[0].cost)) / runnerUp.cost : 0;
  const deadHeat = podium.length >= 2 && Math.abs(podium[podium.length - 1].cost - podium[0].cost) < 0.5;

  return (
    <div className="stack" style={{ gap: 16 }}>
      <Reveal>
        <section className="hero2" style={{ padding: "26px clamp(18px, 3vw, 36px)" }}>
          <span className="blob b1" /><span className="blob b2" />
          <span className="pill"><span className="badge">Live</span> computed on the server as you watch</span>
          <h1 style={{ fontSize: "clamp(28px, 4vw, 48px)", margin: "12px 0 8px" }}><SplitWords text="The decision-maker" /> <SplitWords text="race" delay={160} wordClass="gradient-text" /></h1>
          <p className="lead" style={{ marginBottom: 0 }}>
            Identical copies of one hospital, one per competitor, run side by side on the server. Lowest cost wins.
          </p>
        </section>
      </Reveal>

      {(phase === "setup" || phase === "loading") && (
        <Reveal>
          <section className="card">
            <h2 className="card-title" style={{ marginBottom: 14 }}>Set up the race</h2>
            <div className="control-dock">
              <label className="field">Situation<SituationSelect value={scenario} onChange={setScenario} /></label>
              <label className="field">Run number<input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
              <label className="field">Start from
                <select value={startMode} onChange={(e) => setStartMode(e.target.value as typeof startMode)}>
                  <option value="busy">Busy hospital (after 35-day warm-up)</option>
                  {fromSid && <option value="session">My live hospital run (this exact moment)</option>}
                </select>
              </label>
              <label className="field">Race length
                <select value={days} onChange={(e) => setDays(+e.target.value)}>
                  <option value={1}>1 day</option><option value={3}>3 days</option><option value={7}>1 week</option><option value={14}>2 weeks</option>
                </select>
              </label>
            </div>
            {scenario === "custom" && startMode !== "session" && <ConditionsSliders value={custom} onChange={setCustom} />}
            <div className="picker">
              {POLICY_GROUPS.map((g) => {
                const items = LIVE_POLICIES.filter((p) => g.keys.includes(p.key) && available.includes(p.key));
                return items.length ? (
                  <div key={g.label} className="picker-group">
                    <div className="eyebrow">{g.label}</div>
                    <div className="row" style={{ gap: 8 }}>
                      {items.map((p) => (
                        <button key={p.key} type="button" className={`pick${picked.includes(p.key) ? " on" : ""}`} onClick={() => toggle(p.key)}
                          style={{ ["--pc" as string]: p.color }}>
                          <i className="dot" style={{ background: p.color }} />{p.label}
                          {picked.includes(p.key) && <Icon name="check" size={14} />}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null;
              })}
            </div>
            <div className="row" style={{ marginTop: 16 }}>
              <button className="btn lg glow" onClick={start} disabled={picked.length < 2 || phase === "loading" || available.length === 0}>
                <Icon name="flag" size={16} /> {phase === "loading" ? (startMode === "busy" ? "Simulating the warm-up on the server…" : "Preparing the track…") : "Start the race"}
              </button>
              <span className="muted">{picked.length} of up to 6 competitors · {startMode === "session" ? "from your live run" :
                scenario === "custom" ? describe(custom) : SCENARIO_INFO[scenario[0]]?.short ?? scenario} · run {seed}</span>
            </div>
            {error && <p className="error">{error}</p>}
          </section>
        </Reveal>
      )}

      {race && phase !== "setup" && phase !== "loading" && (
        <>
          <section className="card race-card">
            <div className="race-head">
              <div>
                <div className="eyebrow">{meta.label === "your live run" ? "From your live hospital run" : `Situation: ${scenario === "custom" ? describe(custom) : SCENARIO_INFO[scenario[0]]?.short ?? scenario}`} · run {seed}</div>
                <div className="race-clock">{dayLabel(race.elapsed_hours)} <span className="muted">of {dayLabel(race.horizon_hours)}</span></div>
              </div>
              <span className="spacer" />
              <div className="row" style={{ gap: 8 }}>
                {phase === "racing" && <button className="btn ghost" onClick={() => setPhase("paused")}><Icon name="pause" size={15} /> Pause</button>}
                {phase === "paused" && <button className="btn glow" onClick={() => setPhase("racing")}><Icon name="play" size={15} /> Resume</button>}
                <button className="btn ghost" onClick={() => { setPhase("setup"); setRace(null); }}><Icon name="reset" size={15} /> New race</button>
              </div>
            </div>
            {reconnecting && <div className="reading warn" style={{ marginTop: 10 }}><span className="live-dot amber" style={{ marginTop: 6 }} /><span>{reconnecting}</span></div>}
            {error && <p className="error">{error}</p>}
            <div className="episode-bar" style={{ margin: "10px 0 16px" }}><div className="fill" style={{ width: `${100 * progress}%` }} /></div>

            <div className="track" style={{ height: lanes.length * laneH }}>
              {lanes.map((l, i) => {
                const m = policyMeta(l.name);
                const eff = l.cost > 0 ? best / l.cost : 1;
                const x = Math.min(1, progress * eff);
                return (
                  <div key={l.name} className={`lane${l.rank === 1 && race.elapsed_hours > 0 ? " lead" : ""}`}
                    style={{ top: (l.rank - 1) * laneH, height: laneH - 8, ["--pc" as string]: m.color }}>
                    <div className={`rank r${l.rank}`}>{l.rank}</div>
                    <div className="lane-name"><b>{m.label}</b><span>{m.tech}</span></div>
                    <div className="asphalt">
                      <div className="finish" />
                      <div className="car-wrap" style={{ left: `calc(${(100 * x).toFixed(2)}% - ${(64 * x).toFixed(1)}px)` }}>
                        {phase === "racing" && <span className="speed" />}
                        <RaceCar color={m.color} n={i + 1} />
                      </div>
                    </div>
                    <div className="lane-stats">
                      <div><span>cost so far</span><b>{Math.round(l.cost).toLocaleString()}</b></div>
                      <div><span>turned away</span><b className={l.turned_away ? "bad" : ""}>{l.turned_away}</b></div>
                      <div><span>waiting now</span><b>{l.waiting_now}</b></div>
                    </div>
                  </div>
                );
              })}
              {phase === "countdown" && <div className="countdown"><span key={count}>{count === 0 ? "GO" : count}</span></div>}
            </div>

            <div className="proof" style={{ marginTop: 14 }}>
              <Icon name="cpu" size={14} /><span>Computed live on the server:</span>
              <b>{(race.lanes[0]?.trace.length ?? 1) - 1}</b><span>steps of {stepHours} simulated hours</span>
              <span>· last step <b>{meta.lastMs.toFixed(0)} ms</b> for {race.lanes.length} hospitals · setup {meta.setupMs.toFixed(0)} ms</span>
            </div>
          </section>

          {phase === "finished" && deadHeat && (
            <section className="card" style={{ textAlign: "center" }}>
              <h2 className="card-title" style={{ fontSize: 20 }}>Dead heat</h2>
              <p className="secondary" style={{ margin: "8px auto 0", maxWidth: 620 }}>
                Every competitor finished with the same cost. That happens when the hospital has spare beds throughout, so no decision
                ever forces a patient to wait. Try a crisis situation, custom conditions with more patients, or a longer race.
              </p>
            </section>
          )}
          {phase === "finished" && !deadHeat && podium.length >= 2 && (
            <Reveal kind="zoom">
              <section className="card podium-card">
                <div className="podium">
                  {[podium[1], podium[0], podium[2]].filter(Boolean).map((l) => (
                    <div key={l.name} className={`step-${l.rank}`} style={{ ["--pc" as string]: policyMeta(l.name).color }}>
                      <div className="pod-name">{label(l.name)}</div>
                      <div className="pod-cost">{Math.round(l.cost).toLocaleString()}</div>
                      <div className="pod-block">{l.rank}</div>
                    </div>
                  ))}
                </div>
                <p className="secondary" style={{ textAlign: "center", margin: "14px 0 6px" }}>
                  <b>{label(podium[0].name)}</b> won this race with a {margin.toFixed(1)}% lower cost than {label(runnerUp.name)}.
                </p>
                <p className="muted" style={{ textAlign: "center", margin: 0 }}>
                  One race is one sample of the future. The study&apos;s verdict comes from many paired runs: see <Link href="/experiments/">Results</Link>.
                </p>
                <div className="row" style={{ justifyContent: "center", marginTop: 14 }}>
                  <button className="btn glow" onClick={() => { setSeed((s) => s + 1); setPhase("setup"); setRace(null); }}>
                    <Icon name="repeat" size={15} /> Race again with run {seed + 1}
                  </button>
                </div>
              </section>
            </Reveal>
          )}

          <div className="grid grid-2">
            <section className="card">
              <h2 className="card-title">Cost so far, live</h2>
              <p className="card-sub" style={{ marginBottom: 10 }}>overall cost accumulated since the start (lower is better)</p>
              <Legend items={race.lanes.map((l) => ({ label: label(l.name), color: policyMeta(l.name).color }))} />
              <div style={{ width: "100%", height: 260, marginTop: 8 }}>
                <ResponsiveContainer>
                  <LineChart data={chart} margin={{ top: 10, right: 12, bottom: 4, left: -6 }}>
                    <CartesianGrid stroke="var(--grid)" vertical={false} />
                    <XAxis dataKey="day" type="number" domain={[0, race.horizon_hours / 24]} unit="d" stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
                    <YAxis stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
                    <Tooltip contentStyle={{ background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12, color: "var(--text-primary)" }}
                      labelStyle={{ color: "var(--text-muted)" }}
                      itemSorter={(i) => Number(i.value ?? 0)} labelFormatter={(d) => `day ${d} (best first)`}
                      formatter={(v, n) => [typeof v === "number" ? v.toLocaleString() : "–", label(String(n))]} />
                    {race.lanes.map((l) => (
                      <Line key={l.name} dataKey={l.name} stroke={policyMeta(l.name).color} strokeWidth={2.4} isAnimationActive={false}
                        dot={liveDot(chart.length - 1, policyMeta(l.name).color)} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            <section className="card">
              <h2 className="card-title">Race commentary</h2>
              <p className="card-sub" style={{ marginBottom: 10 }}>generated from the live standings</p>
              <div className="events">
                {notes.length === 0 && <span className="muted">Waiting for the start…</span>}
                {notes.map((n) => (
                  <div key={n.id} className={`event${n.tone === "bad" ? " bad" : ""}${n.tone === "lead" ? " lead" : ""}`}>
                    <span className="time">{dayLabel(n.t)}</span>
                    <Icon name={n.tone === "bad" ? "ambulance" : n.tone === "lead" ? "flag" : "info"} size={14} />
                    <span>{n.text}</span>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </>
      )}

      <section className="card">
        <TechDetails label="How a race works">
          <p>The starting state is copied once and branched into one identical copy per competitor with the same branch seed, so all future
            arrivals, illness severities, lengths of stay and staffing are identical across lanes (common random numbers). Differences
            come only from bed decisions.</p>
          <p>&quot;Busy hospital&quot; first simulates the 35-day warm-up under the balance-the-wards rule, so every competitor inherits the same
            realistic state. Cost uses the study&apos;s formula: urgency-weighted waiting time, plus a penalty per turned-away ambulance and
            per wrong-ward placement. Car positions show progress through the race scaled by each competitor&apos;s cost relative to the leader.</p>
        </TechDetails>
      </section>
    </div>
  );
}
