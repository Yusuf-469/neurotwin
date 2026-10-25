"use client";

import { useCallback, useEffect, useState } from "react";
import DecisionLog from "@/components/DecisionLog";
import HistoryChart from "@/components/HistoryChart";
import StatePanel from "@/components/StatePanel";
import WhatIfPanel from "@/components/WhatIfPanel";
import { api, type DecisionRecord, type HistoryPoint, type StateResponse } from "@/lib/api";

const SCENARIOS = ["A", "B", "C", "D", "E", "B_severe", "C_severe", "D_severe", "E_severe"];

export default function SimulationPage() {
  const [policies, setPolicies] = useState<string[]>([]);
  const [scenario, setScenario] = useState("A");
  const [seed, setSeed] = useState(0);
  const [policy, setPolicy] = useState("priority");
  const [sid, setSid] = useState<string>("");
  const [status, setStatus] = useState<StateResponse | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [decisions, setDecisions] = useState<{ total: number; decisions: DecisionRecord[] }>({ total: 0, decisions: [] });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.policies().then((p) => setPolicies(p.policies)).catch((e) => setError(String(e)));
  }, []);

  const refresh = useCallback(async (id: string) => {
    const [s, h, d] = await Promise.all([api.state(id), api.history(id), api.decisions(id, 25)]);
    setStatus(s);
    setHistory(h.history);
    setDecisions(d);
  }, []);

  const guard = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const start = () => guard(async () => {
    const r = await api.start(scenario, seed, policy);
    setSid(r.session_id);
    await refresh(r.session_id);
  });
  const step = (hours: number) => guard(async () => {
    await api.step(sid, hours);
    await refresh(sid);
  });

  return (
    <>
      <div className="panel">
        <h2>Run a simulation</h2>
        <div className="row">
          <label>scenario
            <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {SCENARIOS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label>seed<input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
          <label>policy
            <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              {policies.map((p) => <option key={p}>{p}</option>)}
            </select>
          </label>
          <button onClick={start} disabled={busy}>start</button>
          <button className="secondary" onClick={() => step(4)} disabled={!sid || busy || status?.finished}>+4 h</button>
          <button className="secondary" onClick={() => step(24)} disabled={!sid || busy || status?.finished}>+1 day</button>
          <button className="secondary" onClick={() => step(168)} disabled={!sid || busy || status?.finished}>+1 week</button>
        </div>
        {status && (
          <p className="muted">
            {status.policy} · scenario {status.scenario} · seed {status.seed} · day{" "}
            {(status.time / 24).toFixed(1)} of {status.horizon_hours / 24}
            {status.finished ? " · finished" : ""}
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </div>

      {status && <StatePanel state={status.state} />}
      {history.length > 1 && status && <HistoryChart history={history} warmupHours={status.warmup_hours} />}
      {sid && <DecisionLog sessionId={sid} decisions={decisions.decisions} total={decisions.total} />}
      {sid && <WhatIfPanel sessionId={sid} policies={policies} disabled={!!status?.finished} />}
    </>
  );
}
