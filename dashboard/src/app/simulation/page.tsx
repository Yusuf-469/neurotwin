"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DecisionLog from "@/components/DecisionLog";
import HospitalMap from "@/components/HospitalMap";
import LiveCharts from "@/components/LiveCharts";
import { Segmented, Stat } from "@/components/ui";
import WhatIfPanel from "@/components/WhatIfPanel";
import { api, type DecisionRecord, type HistoryPoint, type StateResponse } from "@/lib/api";
import { POLICIES, SCENARIO_LABELS, policyMeta } from "@/lib/policies";

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
      <section className="card">
        <div className="row">
          <label className="field">Scenario
            <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {Object.entries(SCENARIO_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              {EXTRA_SCENARIOS.map((k) => <option key={k} value={k}>{k.replace("_", " · ")} (overload)</option>)}
            </select>
          </label>
          <label className="field">Policy
            <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              {POLICIES.filter((p) => policies.includes(p.key)).map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
            </select>
          </label>
          <label className="field">Random seed<input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
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
            {policyMeta(status.policy).label} · {SCENARIO_LABELS[status.scenario] ?? status.scenario} · seed {status.seed}
            {" · "}{inWarmup ? "warm-up period (not measured in the study protocol)" : "measurement period"}
            {status.finished ? " · episode finished" : ""}
          </p>
        )}
      </section>

      {!st && (
        <section className="card">
          <p className="secondary" style={{ margin: 0 }}>
            Start a simulation, then press <strong>Play</strong>. The hospital fills during the 35-day warm-up; PPO,
            rules and the optimiser then make every bed decision live. Try a stress scenario and compare policies
            with the what-if tool.
          </p>
        </section>
      )}

      {st && status && (
        <>
          <div className="grid grid-4">
            <Stat label="Simulated time" value={`day ${day.toFixed(1)}`} note={`of ${status.horizon_hours / 24} days`} />
            <Stat label="Beds occupied" value={`${totalOcc} / ${totalStaffed}`}
              note={`${totalStaffed ? Math.round((100 * totalOcc) / totalStaffed) : 0}% of staffed capacity`} />
            <Stat label="Waiting for a bed" value={String(st.waiting_requests.length)}
              note={`${st.waiting_requests.filter((r) => r.priority <= 2).length} urgent (P1–2)`} />
            <Stat label="Diversions · off-service" value={`${st.cumulative_diversions} · ${st.cumulative_off_service}`}
              note="cumulative since start" />
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
