"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DecisionLog from "@/components/DecisionLog";
import HospitalMap from "@/components/HospitalMap";
import LiveCharts from "@/components/LiveCharts";
import { Reveal } from "@/components/motion";
import { Segmented, Stat, TechDetails } from "@/components/ui";
import WhatIfPanel from "@/components/WhatIfPanel";
import { api, type DecisionRecord, type HistoryPoint, type StateResponse } from "@/lib/api";
import { POLICIES, SCENARIO_INFO, SCENARIO_LABELS, policyMeta } from "@/lib/policies";

const EXTRA_SCENARIOS = ["B_severe", "C_severe", "D_severe", "E_severe"];
type Speed = "4" | "12" | "24";

export default function SimulationPage() {
  const [policies, setPolicies] = useState<string[]>([]);
  const [scenario, setScenario] = useState("A");
  const [seed, setSeed] = useState(3);
  const [policy, setPolicy] = useState("ppo");
  const [sid, setSid] = useState("");
  const [status, setStatus] = useState<StateResponse | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [decisions, setDecisions] = useState<{ total: number; decisions: DecisionRecord[] }>({ total: 0, decisions: [] });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>("12");
  const [error, setError] = useState("");
  const busy = useRef(false);

  useEffect(() => {
    api.policies().then((p) => {
      setPolicies(p.policies);
      if (!p.policies.includes("ppo")) setPolicy("priority");
    }).catch((e) => setError(`API not reachable: ${e}`));
  }, []);

  const refresh = useCallback(async (id: string) => {
    const [s, h, d] = await Promise.all([api.state(id), api.history(id), api.decisions(id, 40)]);
    setStatus(s);
    setHistory(h.history);
    setDecisions(d);
    return s;
  }, []);

  const step = useCallback(async (hours: number) => {
    if (!sid || busy.current) return;
    busy.current = true;
    try {
      await api.step(sid, hours);
      const s = await refresh(sid);
      if (s.finished) setPlaying(false);
    } catch (e) {
      setError(String(e));
      setPlaying(false);
    } finally {
      busy.current = false;
    }
  }, [sid, refresh]);

  // One-click demo: /simulation/?demo=1 starts a PPO run in scenario A and plays it.
  useEffect(() => {
    if (typeof window === "undefined" || !new URLSearchParams(window.location.search).has("demo")) return;
    if (policies.length === 0 || sid) return;
    (async () => {
      try {
        const r = await api.start("A", 3, policies.includes("ppo") ? "ppo" : "priority");
        setSid(r.session_id);
        // Fast-forward through the 35-day warm-up in steps within the API's 14-day limit.
        for (let h = 0; h < 24 * 36; h += 336) await api.step(r.session_id, Math.min(336, 24 * 36 - h));
        await refresh(r.session_id);
        setPlaying(true);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, [policies, sid, refresh]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => step(Number(speed)), 700);
    return () => clearInterval(timer);
  }, [playing, speed, step]);

  const start = async () => {
    setError("");
    setPlaying(false);
    try {
      const r = await api.start(scenario, seed, policy);
      setSid(r.session_id);
      await refresh(r.session_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const st = status?.state;
  const totalOcc = st ? Object.values(st.occupied).reduce((a, b) => a + b, 0) : 0;
  const totalStaffed = st ? Object.values(st.staffed_capacity).reduce((a, b) => a + b, 0) : 0;
  const day = status ? status.time / 24 : 0;
  const inWarmup = status ? status.time < status.warmup_hours : false;

  return (
    <div className="stack">
      <section className="hero2" style={{ padding: "28px clamp(18px, 4vw, 40px)" }}>
        <span className="blob b1" /><span className="blob b2" />
        <Reveal><span className="pill"><i className={sid ? "live-dot" : "live-dot wait"} /> {sid ? "Simulation running" : "Ready when you are"}</span></Reveal>
        <Reveal delay={80}><h1 style={{ fontSize: "clamp(26px, 3.6vw, 42px)", margin: "12px 0 8px" }}>The live hospital</h1></Reveal>
        <Reveal delay={140}><p className="lead" style={{ marginBottom: 0 }}>
          Choose a situation and a decision-maker, press <b>Start</b>, then <b>Play</b>. Watch beds fill, see every decision explained,
          and ask &quot;what if?&quot; at any moment.
        </p></Reveal>
      </section>
      <section className="card">
        <div className="row">
          <label className="field">Situation
            <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {Object.entries(SCENARIO_LABELS).map(([k]) => <option key={k} value={k}>{SCENARIO_INFO[k].icon} {k} · {SCENARIO_INFO[k].short}</option>)}
              {EXTRA_SCENARIOS.map((k) => <option key={k} value={k}>🔥 {k[0]} · {SCENARIO_INFO[k[0]].short}, extreme overload</option>)}
            </select>
          </label>
          <label className="field">Decision-maker
            <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              {POLICIES.filter((p) => policies.includes(p.key)).map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
            </select>
          </label>
          <label className="field">Run number<input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
          <button className="btn" onClick={start}>{sid ? "Restart" : "Start simulation"}</button>
          <span className="spacer" />
          <button className="btn ghost" onClick={() => setPlaying((p) => !p)} disabled={!sid || status?.finished}>
            {playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          <Segmented options={[{ value: "4", label: "4 h/tick" }, { value: "12", label: "12 h" }, { value: "24", label: "1 day" }]}
            value={speed} onChange={setSpeed} />
          <button className="btn ghost" onClick={() => step(168)} disabled={!sid || playing || status?.finished}>+1 week</button>
        </div>
        {error && <p className="error">{error}</p>}
        {status && (
          <p className="muted" style={{ margin: "10px 0 0" }}>
            <i className="dot" style={{ background: policyMeta(status.policy).color, marginRight: 6 }} />
            {policyMeta(status.policy).label} · {SCENARIO_LABELS[status.scenario] ?? status.scenario} · run {status.seed}
            {" · "}{inWarmup ? "warm-up: the hospital is still filling up (not counted in the study)" : "measured period"}
            {status.finished ? " · episode finished" : ""}
          </p>
        )}
      </section>

      {!st && (
        <Reveal>
          <section className="card">
            <h2 className="card-title" style={{ marginBottom: 14 }}>How to use it</h2>
            <div className="guide">
              {[
                ["Pick a situation", "Start with a normal day, then try a crisis such as staff shortage."],
                ["Pick a decision-maker", "The AI agent, or one of the hand-designed rules, to compare."],
                ["Start, then Play", "The first 5 weeks are warm-up while the hospital fills. Use +1 week to skip ahead."],
                ["Click decisions & ask what-if", "See why each bed was given, and test what happens with more patients or fewer staff."],
              ].map(([t, d], i) => (
                <div key={t} className="guide-item"><span className="guide-num">{i + 1}</span><div><b>{t}.</b> {d}</div></div>
              ))}
            </div>
            <TechDetails>
              <p>Each run is a full 49-day episode (35-day warm-up, then 14 measured days), the same protocol as the study. The run number is the
                random seed: the same number gives exactly the same patients for every decision-maker.</p>
            </TechDetails>
          </section>
        </Reveal>
      )}

      {st && status && (
        <>
          <div className="grid grid-4">
            <Stat icon="🕒" label="Simulated time" value={`day ${day.toFixed(1)}`} note={`of ${status.horizon_hours / 24} days`} />
            <Stat icon="🛏️" accent="var(--series-1)" label="Beds in use" value={`${totalOcc} / ${totalStaffed}`}
              note={`${totalStaffed ? Math.round((100 * totalOcc) / totalStaffed) : 0}% of beds with staff`} />
            <Stat icon="⏳" accent="var(--status-serious)" label="Waiting for a bed" value={String(st.waiting_requests.length)}
              note={`${st.waiting_requests.filter((r) => r.priority <= 2).length} critical or very urgent`} />
            <Stat icon="🚑" accent="var(--status-critical)" label="Turned away · wrong ward" value={`${st.cumulative_diversions} · ${st.cumulative_off_service}`}
              note={`${st.cumulative_diversions} turned away · ${st.cumulative_off_service} placed in a wrong ward (total)`} />
          </div>
          <HospitalMap state={st} />
          {history.length > 1 && <LiveCharts history={history} warmupHours={status.warmup_hours} />}
          <DecisionLog sessionId={sid} decisions={decisions.decisions} total={decisions.total} />
          <WhatIfPanel sessionId={sid} policies={policies} disabled={!!status.finished} />
        </>
      )}
    </div>
  );
}
