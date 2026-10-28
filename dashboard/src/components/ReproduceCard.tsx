"use client";

// "Re-run it yourself": the server re-simulates one run of the sealed final test with the
// study's own episode function and compares it with the stored result, digit by digit.
// The comparison is computed server-side; this component only displays it.

import { useEffect, useState } from "react";
import { api, type ReproduceResult } from "@/lib/api";
import { SCENARIO_LABELS, policyMeta } from "@/lib/policies";
import { Icon } from "./icons";
import { CountUp } from "./motion";
import { Card, TechDetails } from "./ui";

const CHOICES = ["ppo_v3", "load_balancing", "priority", "trunk_reservation", "mip", "fcfs"];
const FIRST_SEED = 500_000;
const RUNS = 50;

const METRIC_LABELS: Record<string, string> = {
  J_per_day: "Overall cost per day",
  mean_boarding_h: "Average wait for a bed (hours)",
  diversion_per_100: "Ambulances turned away (per 100)",
  off_service_per_100: "Placed in the wrong ward (per 100)",
};

/** Every significant digit, so "identical" can be checked by eye. */
const full = (v: number) => v.toPrecision(15).replace(/\.?0+$/, "");

export default function ReproduceCard() {
  const [policy, setPolicy] = useState("ppo_v3");
  const [scenario, setScenario] = useState("A");
  const [run, setRun] = useState(1);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<ReproduceResult | null>(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  // seconds counter while the server works (the free server needs up to ~40 s for agent runs)
  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const t0 = Date.now();
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 250);
    return () => window.clearInterval(id);
  }, [busy]);

  const go = async (p = policy, s = scenario, r = run) => {
    setBusy(true); setError(""); setRes(null);
    try {
      setRes(await api.reproduce(p, s, FIRST_SEED + r - 1));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const random = () => {
    const p = CHOICES[Math.floor(Math.random() * CHOICES.length)];
    const s = "ABCDE"[Math.floor(Math.random() * 5)];
    const r = 1 + Math.floor(Math.random() * RUNS);
    setPolicy(p); setScenario(s); setRun(r);
    void go(p, s, r);
  };

  const cost = res?.metrics.find((m) => m.metric === "J_per_day");

  return (
    <Card title="Re-run any test result yourself" subtitle="the server repeats one sealed test run from scratch, live"
      explain="Pick any of the 1,500 runs behind these results. The server simulates it again from an empty hospital and compares its answer with the published number.">
      <div className="repro-controls">
        <label className="field">
          <span>Decision-maker</span>
          <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
            {CHOICES.map((k) => <option key={k} value={k}>{policyMeta(k).label}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Situation</span>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {"ABCDE".split("").map((k) => <option key={k} value={k}>{SCENARIO_LABELS[k]}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Test run (1 to {RUNS})</span>
          <input type="number" min={1} max={RUNS} value={run}
            onChange={(e) => setRun(Math.min(RUNS, Math.max(1, Math.round(Number(e.target.value) || 1))))} />
        </label>
        <div className="row" style={{ gap: 8, alignSelf: "end" }}>
          <button className="btn lg glow" onClick={() => go()} disabled={busy}><Icon name="repeat" size={16} /> Re-run it</button>
          <button className="btn lg ghost" onClick={random} disabled={busy}>Surprise me</button>
        </div>
      </div>

      {busy && (
        <div className="repro-busy">
          <span className="live-dot wait" /> Simulating 49 days of this hospital on the server, decision by decision
          <span className="mono"> · {elapsed} s</span>
          <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
            On the free server this takes a few seconds for a rule, and up to about 40 seconds for the RL agent or the optimiser.
          </div>
          <div className="skel skel-line" style={{ marginTop: 10 }} />
        </div>
      )}
      {error && <p className="error">{error}</p>}

      {res && cost && (
        <div className="repro-result">
          <div className="repro-pair">
            <div className="repro-num">
              <span className="eyebrow">Published result</span>
              <span className="repro-big">{cost.stored.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              <span className="repro-digits">{full(cost.stored)}</span>
            </div>
            <div className={`repro-eq ${res.all_exact ? "ok" : "diff"}`}>
              <Icon name={res.all_exact ? "check" : "alert"} size={22} stroke={2.4} />
            </div>
            <div className="repro-num">
              <span className="eyebrow">Just recomputed ({(res.compute_ms / 1000).toFixed(1)} s)</span>
              <span className="repro-big"><CountUp value={cost.reproduced} digits={2} duration={900} /></span>
              <span className="repro-digits">{full(cost.reproduced)}</span>
            </div>
          </div>
          <p className={`repro-verdict ${res.all_exact ? "ok" : "diff"}`}>
            {res.all_exact
              ? `Identical to every digit. ${policyMeta(res.policy).label}, ${SCENARIO_LABELS[res.scenario]}, test run ${res.run_number}: the simulation is fully determined by its run number, so anyone can check any published figure.`
              : "Not identical: the recomputed run differs from the stored one by the amounts below. This is shown as it is, not hidden."}
          </p>
          <table className="repro-table">
            <thead><tr><th>Measure</th><th>Published</th><th>Recomputed</th><th /></tr></thead>
            <tbody>
              {res.metrics.map((m) => (
                <tr key={m.metric}>
                  <td>{METRIC_LABELS[m.metric] ?? m.metric}</td>
                  <td className="mono">{m.stored.toFixed(4)}</td>
                  <td className="mono">{m.reproduced.toFixed(4)}</td>
                  <td>{m.exact ? <span className="repro-ok"><Icon name="check" size={13} /> same</span>
                    : <span className="repro-bad">{m.difference > 0 ? "+" : ""}{m.difference.toExponential(2)}</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <TechDetails>
            <p>Server call: the same episode function the study used for its one-time final test, with the same arguments; seed {res.seed}
              (test runs are seeds {FIRST_SEED.toLocaleString("en-GB")} to {(FIRST_SEED + RUNS - 1).toLocaleString("en-GB")}). Stored values come from
              the published per-run file of that test. Comparison is exact floating-point equality.</p>
          </TechDetails>
        </div>
      )}
    </Card>
  );
}
