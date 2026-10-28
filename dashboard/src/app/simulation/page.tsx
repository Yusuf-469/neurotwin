"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DecisionLog from "@/components/DecisionLog";
import FloorPlan from "@/components/FloorPlan";
import { Icon, type IconName } from "@/components/icons";
import LiveCharts from "@/components/LiveCharts";
import RunVerdict from "@/components/RunVerdict";
import { Reveal, SplitWords } from "@/components/motion";
import { Segmented, TechDetails } from "@/components/ui";
import Link from "next/link";
import { ConditionsSliders, PolicySelect, SituationSelect, describe } from "@/components/setup";
import { RunLost, ServerUnavailable, api, type Conditions, type DecisionRecord, type HistoryPoint, type HospitalState, type StateResponse } from "@/lib/api";
import { PRIORITY_COLORS, PRIORITY_NAMES, SCENARIO_INFO, SCENARIO_LABELS, UNIT_LABELS, policyMeta } from "@/lib/policies";

// Live "control room": the animated floor plan is driven by the real simulation state; every
// tick advances the simulator by `speed` simulated hours and animates what happened in between
// (placements, discharges, diversions). Charts, decision explanations and what-if sit in tabs.

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

type Tab = "trends" | "decisions";
type Server = "checking" | "online" | "waking" | "offline";

export default function SimulationPage() {
  const [policies, setPolicies] = useState<string[]>([]);
  const [scenario, setScenario] = useState("A");
  const [seed, setSeed] = useState(3);
  const [policy, setPolicy] = useState("ppo_v3");
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
  const [server, setServer] = useState<Server>("checking");
  const [reconnecting, setReconnecting] = useState("");
  const recovering = useRef(false);
  const statusRef = useRef<StateResponse | null>(null);
  const sidRef = useRef("");
  const [ping, setPing] = useState<number | null>(null);
  const [work, setWork] = useState({ steps: 0, decisions: 0, lastMs: 0, roundTripMs: 0 });
  const [warmMode, setWarmMode] = useState<"skip" | "empty">("skip");
  const [custom, setCustom] = useState<Conditions>({ arrival_multiplier: 1.3, staff_multiplier: 0.9, capacity_multiplier: 1.0 });
  const busy = useRef(false);
  const lastId = useRef(-1);
  const lastDiv = useRef(0);

  // Server status, measured: a health request every 30 s with its round-trip time.
  useEffect(() => {
    let alive = true;
    const check = async () => {
      const t0 = performance.now();
      const slow = setTimeout(() => alive && setServer((s) => (s === "online" ? s : "waking")), 2500);
      try {
        await api.health();
        if (!alive) return;
        setServer("online");
        setPing(Math.round(performance.now() - t0));
      } catch {
        if (alive) setServer("offline");
      } finally {
        clearTimeout(slow);
      }
    };
    check();
    const t = setInterval(check, 30000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  useEffect(() => { statusRef.current = status; }, [status]);
  useEffect(() => { sidRef.current = sid; }, [sid]);

  const record = (r: Awaited<ReturnType<typeof api.step>>, t0: number) => {
    if (!r.step) return;
    setWork((w) => ({ steps: w.steps + 1, decisions: w.decisions + r.step!.decisions_made, lastMs: r.step!.compute_ms,
      roundTripMs: Math.round(performance.now() - t0) }));
  };

  useEffect(() => {
    api.policies().then((p) => {
      setPolicies(p.policies);
      setPolicy(["ppo_v3", "ppo_ensemble", "ppo", "priority"].find((k) => p.policies.includes(k)) ?? "priority");
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
      const t0 = performance.now();
      record(await api.step(sid, hours), t0);
      const s = await refresh(sid, true);
      if (s.finished) setPlaying(false);
    } catch (e) {
      setPlaying(false);
      if (e instanceof ServerUnavailable || e instanceof RunLost) recoverRef.current?.();
      else setError(String(e));
    } finally {
      busy.current = false;
    }
  }, [sid, refresh]);

  // Reconnect after the server sleeps or restarts. If the run was lost with the restart, recreate
  // it: runs are deterministic (same situation and run number = same patients and decisions), so
  // replaying to the same simulated time restores exactly the same hospital state.
  const recoverRef = useRef<(() => void) | null>(null);
  recoverRef.current = async () => {
    if (recovering.current) return;
    recovering.current = true;
    setServer("waking");
    setReconnecting("Lost contact with the simulation server (it is waking up or restarting). Reconnecting…");
    try {
      const deadline = Date.now() + 180_000;
      let up = false;
      while (!up && Date.now() < deadline) {
        try { await api.health(); up = true; } catch { await new Promise((r) => setTimeout(r, 3000)); }
      }
      if (!up) {
        setReconnecting("");
        setServer("offline");
        setError("The simulation server did not come back within 3 minutes. Please refresh the page later.");
        return;
      }
      setServer("online");
      const old = statusRef.current;
      try {
        await api.state(sidRef.current);
      } catch (e) {
        if (!(e instanceof RunLost) || !old) throw e;
        const target = old.time;
        setReconnecting(`The server restarted, so your run is being restored at day ${(target / 24).toFixed(1)}…`);
        const r = await api.start(old.scenario, old.seed, old.policy, old.scenario === "custom" ? old.conditions : undefined);
        for (let h = 0; h < target; h += 336) await api.step(r.session_id, Math.min(336, target - h));
        lastId.current = -1;
        lastDiv.current = 0;
        setSid(r.session_id);
        sidRef.current = r.session_id;
        await refresh(r.session_id, false);
        setEvents((ev) => [{ id: `restore${Date.now()}`, time: target, color: "var(--accent)", text: "Server restarted: run restored at the same moment" }, ...ev].slice(0, 14));
      }
      setReconnecting("");
      setError("");
      setPlaying(true);
    } catch (e) {
      setReconnecting("");
      setError(String(e));
    } finally {
      recovering.current = false;
    }
  };

  const begin = useCallback(async (sc: string, sd: number, pol: string, warm: boolean, cond?: Conditions) => {
    setError("");
    setPlaying(false);
    setEvents([]);
    setWork({ steps: 0, decisions: 0, lastMs: 0, roundTripMs: 0 });
    lastId.current = -1;
    lastDiv.current = 0;
    setLoading("Connecting to the simulation server…");
    try {
      const r = await api.start(sc, sd, pol, sc === "custom" ? cond : undefined);
      setSid(r.session_id);
      if (warm) {
        // Simulate the 35-day warm-up on the server one week at a time, reporting real progress.
        let decided = 0;
        for (let h = 0; h < 24 * 35; h += 168) {
          setLoading(`Simulating the warm-up on the server: day ${Math.round(h / 24)} of 35 · ${decided.toLocaleString()} bed decisions so far`);
          const t0 = performance.now();
          const res = await api.step(r.session_id, Math.min(168, 24 * 35 - h));
          record(res, t0);
          decided += res.step?.decisions_made ?? 0;
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
    const best = ["ppo_v3", "ppo_ensemble", "ppo", "priority"].find((k) => policies.includes(k)) ?? "priority";
    setPolicy(best);
    begin("A", 3, best, true).then((id) => id && setPlaying(true));
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

  const phase = reconnecting ? "reconnecting" : loading ? "busy" : !sid ? "idle" : status?.finished ? "finished" : playing ? "running" : "paused";
  const phaseLabel = { reconnecting: "Reconnecting", busy: loading.startsWith("Connecting") ? "Connecting to the server" : "Simulating warm-up on the server",
    idle: "Not started", finished: "Run finished", running: "Running live", paused: "Paused" }[phase];
  const phaseDot = phase === "running" || phase === "busy" ? "live-dot" : phase === "paused" ? "live-dot amber" : "live-dot wait";
  const serverLabel = reconnecting ? "Reconnecting to the server…" : server === "online" ? `Server online${ping !== null ? ` · ${ping} ms` : ""}`
    : server === "waking" ? "Server waking up (free tier, up to a minute)" : server === "offline" ? "Server unreachable" : "Checking server…";

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
              <div className="row" style={{ gap: 8 }}>
                <span className="pill"><i className={phaseDot} />{phaseLabel}</span>
                <span className="pill"><i className={server === "online" ? "live-dot" : server === "offline" ? "live-dot off" : "live-dot amber"} />{serverLabel}</span>
              </div>
              <h1 style={{ fontSize: "clamp(26px, 3.4vw, 40px)", margin: "10px 0 6px" }}><SplitWords text="The live hospital" /></h1>
              <p className="secondary" style={{ margin: 0, maxWidth: 640 }}>
                Computed live on the server. Every bed decision is made by the decision-maker you choose.
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
              <SituationSelect value={scenario} onChange={setScenario} />
            </label>
            <label className="field">Decision-maker
              <PolicySelect value={policy} onChange={setPolicy} available={policies} />
            </label>
            <label className="field">Run number<input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
            <label className="field">Start from
              <select value={warmMode} onChange={(e) => setWarmMode(e.target.value as "skip" | "empty")}>
                <option value="skip">Busy hospital (simulate 35-day warm-up first)</option>
                <option value="empty">Empty hospital (watch it fill from day 1)</option>
              </select>
            </label>
            <button className="btn" onClick={() => begin(scenario, seed, policy, warmMode === "skip", custom).then((id) => id && setPlaying(true))} disabled={!!loading}>
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
            {sid && !status?.finished && (
              <Link href={`/race/?from=${sid}`} className="btn ghost" onClick={() => setPlaying(false)}>
                <Icon name="flag" size={15} /> Race from this moment
              </Link>
            )}
          </div>
          {scenario === "custom" && <ConditionsSliders value={custom} onChange={setCustom} />}
          {reconnecting && <div className="reading warn" style={{ marginTop: 12 }}><span className="live-dot amber" style={{ marginTop: 6 }} /><span>{reconnecting}</span></div>}
          {error && <p className="error" style={{ marginBottom: 0 }}>{error}</p>}
          {status && (
            <p className="muted" style={{ margin: "10px 0 0" }}>
              <i className="dot" style={{ background: policyMeta(status.policy).color, marginRight: 6 }} />
              <b>{policyMeta(status.policy).label}</b> is deciding · {status.scenario === "custom" && status.conditions
                ? `Custom: ${describe(status.conditions)}` : SCENARIO_LABELS[status.scenario] ?? status.scenario} · run {status.seed}
              {" · "}each step = {speed} simulated hours
            </p>
          )}
          {work.steps > 0 && (
            <div className="proof">
              <Icon name="cpu" size={14} />
              <span>Computed live on the server:</span>
              <b>{work.steps}</b><span>steps</span>
              <b>{work.decisions.toLocaleString()}</b><span>bed decisions</span>
              <span>· last step <b>{work.lastMs.toFixed(0)} ms</b> of simulation, <b>{work.roundTripMs} ms</b> round trip</span>
            </div>
          )}
        </section>
      </Reveal>

      <div className="room">
        <Reveal kind="zoom">
          <section className="card plan-card">
            <div className="plan-scroll">
              <FloorPlan state={st} newDecisions={fresh.decisions} newDiversions={fresh.diversions} playing={playing}
                hour={c.hh} tickMs={TICK_MS} />
            </div>
            <div className="swipe-hint"><Icon name="arrow" size={13} /> Swipe the floor plan sideways</div>
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
                    {loading ? (server === "online"
                      ? "Every day is being simulated event by event on the server right now; this takes a few seconds."
                      : "The free server sleeps when unused and can take up to a minute to wake up.")
                      : "The hospital starts empty. Choose \"Busy hospital\" to simulate the 35-day warm-up first (the study measures only after it), or \"Empty hospital\" to watch it fill from day 1."}
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

      {status && sid && (
        <Reveal>
          <RunVerdict sessionId={sid} policy={status.policy} state={st} soFar={status.so_far} now={now}
            warmupHours={warmup} available={policies} onCompare={() => setPlaying(false)} />
        </Reveal>
      )}

      <Reveal>
        <section className="card">
          <div className="tabs" role="tablist">
            {([["trends", "chart", "Trends over time"], ["decisions", "chat", "Every decision, explained"]] as [Tab, IconName, string][]).map(([k, icon, label]) => (
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
