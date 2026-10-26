"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DecisionLog from "@/components/DecisionLog";
import FloorPlan from "@/components/FloorPlan";
import { Icon, type IconName } from "@/components/icons";
import LiveCharts from "@/components/LiveCharts";
import { Reveal } from "@/components/motion";
import { Segmented, TechDetails } from "@/components/ui";
import WhatIfPanel from "@/components/WhatIfPanel";
import { api, type DecisionRecord, type HistoryPoint, type HospitalState, type StateResponse } from "@/lib/api";
import { LIVE_POLICIES, PRIORITY_COLORS, PRIORITY_NAMES, SCENARIO_INFO, SCENARIO_LABELS, UNIT_LABELS, policyMeta } from "@/lib/policies";

// Live "control room": the animated floor plan is driven by the real simulation state; every
// tick advances the simulator by `speed` simulated hours and animates what happened in between
// (placements, discharges, diversions). Charts, decision explanations and what-if sit in tabs.

const EXTRA_SCENARIOS = ["B_severe", "C_severe", "D_severe", "E_severe"];
type Speed = "2" | "6" | "12" | "24";
const TICK_MS = 1800;

/** Empty hospital shown before a run starts (real ward sizes; nothing occupied). */
const EMPTY: HospitalState = {
  time: 0, physical_beds: { MED: 72, SURG: 70, ICU: 12 }, staffed_capacity: { MED: 72, SURG: 70, ICU: 12 },
  occupied: { MED: 0, SURG: 0, ICU: 0 }, nurses_on_shift: { MED: 0, SURG: 0, ICU: 0 }, doctors_on_shift: 0, ed_count: 0,
  ed_waiting_for_doctor: 0, ed_in_treatment: 0, waiting_requests: [], off_service_in_beds: 0, cumulative_placements: 0,
  cumulative_diversions: 0, cumulative_off_service: 0,
};

interface FeedEvent { id: string; time: number; text: string; color: string; bad?: boolean }

function clock(hours: number) {
  const day = Math.floor(hours / 24) + 1;
  const h = Math.floor(hours % 24);
  const m = Math.floor((hours * 60) % 60);
  return { day, hh: h, label: `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}` };
}

type Tab = "trends" | "decisions" | "whatif";

export default function SimulationPage() {
  const [policies, setPolicies] = useState<string[]>([]);
  const [scenario, setScenario] = useState("A");
  const [seed, setSeed] = useState(3);
  const [policy, setPolicy] = useState("ppo");
  const [sid, setSid] = useState("");
  const [status, setStatus] = useState<StateResponse | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [decisions, setDecisions] = useState<{ total: number; decisions: DecisionRecord[] }>({ total: 0, decisions: [] });
  const [fresh, setFresh] = useState<{ decisions: DecisionRecord[]; diversions: number }>({ decisions: [], diversions: 0 });
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>("6");
  const [tab, setTab] = useState<Tab>("trends");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState("");
  const busy = useRef(false);
  const lastId = useRef(-1);
  const lastDiv = useRef(0);

  useEffect(() => {
    api.policies().then((p) => {
      setPolicies(p.policies);
      if (!p.policies.includes("ppo")) setPolicy("priority");
    }).catch((e) => setError(`Simulator not reachable yet (the free server may be waking up): ${e}`));
  }, []);

  const refresh = useCallback(async (id: string, animate: boolean) => {
    const [s, h, d] = await Promise.all([api.state(id), api.history(id), api.decisions(id, 80)]);
    const newDecisions = d.decisions.filter((x) => x.decision_id > lastId.current).sort((a, b) => a.decision_id - b.decision_id);
    const newDiv = Math.max(0, s.state.cumulative_diversions - lastDiv.current);
    if (d.decisions.length) lastId.current = Math.max(lastId.current, ...d.decisions.map((x) => x.decision_id));
    lastDiv.current = s.state.cumulative_diversions;
    setFresh(animate ? { decisions: newDecisions, diversions: newDiv } : { decisions: [], diversions: 0 });
    if (animate) {
      const ev: FeedEvent[] = newDecisions.filter((x) => x.action === "PLACE").slice(-6).map((x) => ({
        id: `d${x.decision_id}`, time: x.time, color: PRIORITY_COLORS[x.priority ?? 5],
        text: `${PRIORITY_NAMES[x.priority ?? 5]} patient → ${UNIT_LABELS[x.unit ?? ""] ?? x.unit}${x.off_service ? " (wrong ward)" : ""}`,
      }));
      if (newDiv > 0) ev.push({ id: `v${s.state.time}`, time: s.state.time, color: "var(--status-critical)", bad: true,
        text: `${newDiv} ambulance${newDiv > 1 ? "s" : ""} turned away: ED full` });
      setEvents((old) => [...ev.reverse(), ...old].slice(0, 14));
    }
    setStatus(s);
    setHistory(h.history);
    setDecisions({ total: d.total, decisions: d.decisions.slice(-40) });
    return s;
  }, []);

  const step = useCallback(async (hours: number) => {
    if (!sid || busy.current) return;
    busy.current = true;
    try {
      await api.step(sid, hours);
      const s = await refresh(sid, true);
      if (s.finished) setPlaying(false);
    } catch (e) {
      setError(String(e));
      setPlaying(false);
    } finally {
      busy.current = false;
    }
  }, [sid, refresh]);

  const begin = useCallback(async (sc: string, sd: number, pol: string, warm: boolean) => {
    setError("");
    setPlaying(false);
    setEvents([]);
    lastId.current = -1;
    lastDiv.current = 0;
    setLoading("Starting the simulator…");
    try {
      const r = await api.start(sc, sd, pol);
      setSid(r.session_id);
      if (warm) {
        // Fast-forward through the 35-day warm-up in steps within the API's 14-day limit.
        for (let h = 0; h < 24 * 35; h += 336) {
          setLoading(`Filling the hospital (warm-up day ${Math.round(h / 24)} of 35)…`);
          await api.step(r.session_id, Math.min(336, 24 * 35 - h));
        }
      }
      await refresh(r.session_id, false);
      setLoading("");
      return r.session_id;
    } catch (e) {
      setLoading("");
      setError(String(e));
      return "";
    }
  }, [refresh]);

  // One-click demo: /simulation/?demo=1 starts the RL agent in scenario A, skips warm-up and plays.
  const demoStarted = useRef(false);
  useEffect(() => {
    if (typeof window === "undefined" || !new URLSearchParams(window.location.search).has("demo")) return;
    if (policies.length === 0 || demoStarted.current) return;
    demoStarted.current = true;
    begin("A", 3, policies.includes("ppo") ? "ppo" : "priority", true).then((id) => id && setPlaying(true));
  }, [policies, begin]);

  useEffect(() => {
    if (!playing) return;
    step(Number(speed));
    const timer = setInterval(() => step(Number(speed)), TICK_MS);
    return () => clearInterval(timer);
  }, [playing, speed, step]);

  const st = status?.state ?? EMPTY;
  const now = status?.time ?? 0;
  const c = clock(now);
  const night = c.hh < 7 || c.hh >= 20;
  const horizon = status?.horizon_hours ?? 1176;
  const warmup = status?.warmup_hours ?? 840;
  const inWarmup = now < warmup;
  const totalOcc = Object.values(st.occupied).reduce((a, b) => a + b, 0);
  const totalStaffed = Object.values(st.staffed_capacity).reduce((a, b) => a + b, 0);
  const urgent = st.waiting_requests.filter((r) => r.priority <= 2).length;
  const longest = st.waiting_requests.reduce((m, r) => Math.max(m, now - r.created_at), 0);

  const kpis: { k: string; v: string; icon: IconName; color?: string }[] = [
    { k: "Beds in use", v: `${totalStaffed ? Math.round((100 * totalOcc) / totalStaffed) : 0}%`, icon: "bed" },
    { k: "Waiting for a bed", v: String(st.waiting_requests.length), icon: "hourglass" },
    { k: "Critical / very urgent", v: String(urgent), icon: "alert", color: urgent ? "var(--status-critical)" : undefined },
    { k: "Longest wait", v: `${longest.toFixed(1)} h`, icon: "clock" },
    { k: "Turned away (total)", v: String(st.cumulative_diversions), icon: "ambulance", color: st.cumulative_diversions ? "var(--status-critical)" : undefined },
    { k: "Wrong ward (total)", v: String(st.cumulative_off_service), icon: "split" },
  ];

  return (
    <div className="stack" style={{ gap: 16 }}>
      <Reveal>
        <section className="hero2" style={{ padding: "22px clamp(18px, 3vw, 32px)" }}>
          <span className="blob b1" /><span className="blob b2" />
          <div className="row" style={{ alignItems: "center", gap: 18 }}>
            <div style={{ flex: "1 1 360px" }}>
              <span className="pill"><i className={sid ? (playing ? "live-dot" : "live-dot wait") : "live-dot wait"} />
                {sid ? (playing ? "Simulation running" : "Paused") : "Ready when you are"}</span>
              <h1 style={{ fontSize: "clamp(26px, 3.4vw, 40px)", margin: "10px 0 6px" }}>The live hospital</h1>
              <p className="secondary" style={{ margin: 0, maxWidth: 640 }}>
                A real simulation running on the server. Patients arrive, wait, get a bed or are turned away, and every bed
                decision is made by the decision-maker you choose.
              </p>
            </div>
            <div className="clock">
              <div className={`clock-face${night ? " night" : ""}`}><Icon name={night ? "moon" : "sun"} size={26} /></div>
              <div>
                <div className="clock-time">Day {c.day} · {c.label}</div>
                <div className="clock-sub">{!status ? "not started" : inWarmup ? "warm-up: hospital filling (not measured)" : status.finished ? "run finished" : "measured period"}</div>
              </div>
            </div>
          </div>
          <div className="episode-bar" style={{ marginTop: 16 }} title="Run progress: hatched part = warm-up">
            <div className="warm" style={{ width: `${(100 * warmup) / horizon}%` }} />
            <div className="fill" style={{ width: `${(100 * now) / horizon}%` }} />
          </div>
        </section>
      </Reveal>

      <Reveal>
        <section className="card">
          <div className="control-dock">
            <label className="field">Situation
              <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
                {Object.keys(SCENARIO_LABELS).map((k) => <option key={k} value={k}>{k} · {SCENARIO_INFO[k].short}</option>)}
                {EXTRA_SCENARIOS.map((k) => <option key={k} value={k}>{k[0]} · {SCENARIO_INFO[k[0]].short}, extreme overload</option>)}
              </select>
            </label>
            <label className="field">Decision-maker
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                {LIVE_POLICIES.filter((p) => policies.includes(p.key)).map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
              </select>
            </label>
            <label className="field">Run number<input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
            <button className="btn" onClick={() => begin(scenario, seed, policy, true)} disabled={!!loading}>
              <Icon name={sid ? "reset" : "play"} size={15} /> {sid ? "Restart" : "Start"}
            </button>
            <span className="spacer" />
            <button className="btn glow" onClick={() => setPlaying((p) => !p)} disabled={!sid || status?.finished || !!loading}>
              <Icon name={playing ? "pause" : "play"} size={15} /> {playing ? "Pause" : "Play"}
            </button>
            <Segmented options={[{ value: "2", label: "Slow" }, { value: "6", label: "Normal" }, { value: "12", label: "Fast" }, { value: "24", label: "Turbo" }]}
              value={speed} onChange={setSpeed} />
            <button className="btn ghost" onClick={() => step(168)} disabled={!sid || playing || status?.finished}>
              <Icon name="forward" size={15} /> +1 week
            </button>
          </div>
          {error && <p className="error" style={{ marginBottom: 0 }}>{error}</p>}
          {status && (
            <p className="muted" style={{ margin: "10px 0 0" }}>
              <i className="dot" style={{ background: policyMeta(status.policy).color, marginRight: 6 }} />
              <b>{policyMeta(status.policy).label}</b> is deciding · {SCENARIO_LABELS[status.scenario] ?? status.scenario} · run {status.seed}
              {" · "}each step = {speed} simulated hours
            </p>
          )}
        </section>
      </Reveal>

      <div className="room">
        <Reveal kind="zoom">
          <section className="card plan-card">
            <FloorPlan state={st} newDecisions={fresh.decisions} newDiversions={fresh.diversions} playing={playing}
              hour={c.hh} tickMs={TICK_MS} />
            <div className="plan-legend">
              <span><i className="lg-bed" style={{ background: "color-mix(in srgb, var(--series-1) 30%, var(--surface-1))", borderColor: "var(--series-1)" }} />occupied bed</span>
              <span><i className="lg-bed" style={{ background: "var(--bed-free)" }} />free bed</span>
              <span><i className="lg-bed" style={{ background: "repeating-linear-gradient(45deg, var(--grid) 0 2px, transparent 2px 5px)", borderStyle: "dashed" }} />no nurse to staff it</span>
              <span><i className="dot" style={{ background: PRIORITY_COLORS[1] }} />critical / very urgent</span>
              <span><i className="dot" style={{ background: PRIORITY_COLORS[3] }} />urgent</span>
              <span><i className="dot" style={{ background: PRIORITY_COLORS[4] }} />standard / planned</span>
            </div>
            {(!status || loading) && (
              <div className="plan-overlay">
                <div>
                  <div className="icon-badge" style={{ margin: "0 auto 12px" }}><Icon name={loading ? "hourglass" : "play"} size={24} /></div>
                  <h2 className="card-title" style={{ fontSize: 20 }}>{loading || "Choose a situation and press Start"}</h2>
                  <p className="secondary" style={{ maxWidth: 440, margin: "8px auto 0" }}>
                    {loading ? "The free server can take up to a minute to wake up on the first visit." :
                      "The first 35 simulated days are a warm-up while the hospital fills; Start skips straight past it."}
                  </p>
                </div>
              </div>
            )}
          </section>
        </Reveal>

        <div className="stack" style={{ gap: 16 }}>
          <Reveal kind="right">
            <section className="card">
              <h2 className="card-title" style={{ marginBottom: 12 }}>Right now</h2>
              <div className="kpi">
                {kpis.map((x) => (
                  <div key={x.k}>
                    <div className="k"><Icon name={x.icon} size={13} />{x.k}</div>
                    <div key={x.v} className="v bump" style={{ color: x.color }}>{x.v}</div>
                  </div>
                ))}
              </div>
            </section>
          </Reveal>
          <Reveal kind="right" delay={100}>
            <section className="card">
              <h2 className="card-title" style={{ marginBottom: 12 }}>What just happened</h2>
              <div className="events">
                {events.length === 0 && <span className="muted">Press Play to see decisions as they happen.</span>}
                {events.map((e) => {
                  const t = clock(e.time);
                  return (
                    <div key={e.id} className={`event${e.bad ? " bad" : ""}`}>
                      <span className="time">D{t.day} {t.label}</span>
                      {e.bad ? <Icon name="ambulance" size={14} style={{ color: e.color }} /> : <i className="dot" style={{ background: e.color }} />}
                      <span>{e.text}</span>
                    </div>
                  );
                })}
              </div>
            </section>
          </Reveal>
        </div>
      </div>

      <Reveal>
        <section className="card">
          <div className="tabs" role="tablist">
            {([["trends", "chart", "Trends over time"], ["decisions", "chat", "Every decision, explained"], ["whatif", "wand", "What if…?"]] as [Tab, IconName, string][]).map(([k, icon, label]) => (
              <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
                <Icon name={icon} size={16} />{label}
              </button>
            ))}
          </div>
          <div className="tab-panel" key={tab}>
            {!status && <p className="muted">Start a run to see this.</p>}
            {status && tab === "trends" && (history.length > 1
              ? <LiveCharts history={history} warmupHours={status.warmup_hours} />
              : <p className="muted">Play for a little while to build up the charts.</p>)}
            {status && tab === "decisions" && <DecisionLog sessionId={sid} decisions={decisions.decisions} total={decisions.total} />}
            {status && tab === "whatif" && <WhatIfPanel sessionId={sid} policies={policies} disabled={!!status.finished} />}
          </div>
          <TechDetails label="How this simulation works">
            <p>Discrete-event simulation: arrivals, treatment, stays and nurse shifts are random events drawn from fixed, seeded
              distributions. The run number is the seed, so the same number gives exactly the same patients for every decision-maker.</p>
            <p>Each run lasts 49 simulated days: a 35-day warm-up while the hospital fills from empty, then 14 measured days, the same
              protocol as the study. The floor plan shows exact counts; which individual bed a patient takes is not modelled by the
              simulator, so bed positions are drawn for display only.</p>
          </TechDetails>
        </section>
      </Reveal>
    </div>
  );
}
