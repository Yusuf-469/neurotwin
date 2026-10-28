"use client";

// What a live run means: a plain-language reading of the measured results so far, and a verdict
// comparing the chosen decision-maker with the rules on exactly the same patients (the server
// re-runs this run's full 49 days for each one; the simulator is deterministic, so the chosen
// decision-maker's own re-run reproduces this run).

import { useState } from "react";
import { api, type HospitalState, type SoFar } from "@/lib/api";
import { policyMeta } from "@/lib/policies";
import { Icon } from "./icons";
import { TechDetails } from "./ui";

interface Props {
  sessionId: string;
  policy: string;
  state: HospitalState;
  soFar: SoFar | null | undefined;
  now: number;
  warmupHours: number;
  available: string[];
  /** Called when a comparison starts (the page pauses playback so the server can run it). */
  onCompare?: () => void;
}

function reading(state: HospitalState, soFar: SoFar | null | undefined, now: number): { tone: string; headline: string } {
  const staffed = Object.values(state.staffed_capacity).reduce((a, b) => a + b, 0);
  const occ = Object.values(state.occupied).reduce((a, b) => a + b, 0);
  const pct = staffed ? (100 * occ) / staffed : 0;
  const urgent = state.waiting_requests.filter((r) => r.priority <= 2);
  const longestUrgent = urgent.reduce((m, r) => Math.max(m, now - r.created_at), 0);
  if (soFar && soFar.turned_away_per_100 >= 5)
    return { tone: "bad", headline: `Overwhelmed: about ${soFar.turned_away_per_100.toFixed(0)} of every 100 arriving patients are being turned away.` };
  if (urgent.length && longestUrgent > 6)
    return { tone: "warn", headline: `Under strain: ${urgent.length} urgent patient${urgent.length > 1 ? "s are" : " is"} waiting, the longest for ${longestUrgent.toFixed(1)} hours.` };
  if (pct >= 95)
    return { tone: "info", headline: `Full but coping: ${pct.toFixed(0)}% of staffed beds are in use, and patients are still being placed.` };
  return { tone: "good", headline: `Coping comfortably: ${pct.toFixed(0)}% of staffed beds are in use and waits are short.` };
}

export default function RunVerdict({ sessionId, policy, state, soFar, now, warmupHours, available, onCompare }: Props) {
  const rivals = ["load_balancing", "priority", "fcfs"].filter((p) => p !== policy && available.includes(p));
  const field = [policy, ...rivals, ...(policy !== "ppo_ensemble" && available.includes("ppo_ensemble") ? ["ppo_ensemble"] : [])];
  const [results, setResults] = useState<Record<string, number>>({});
  const [running, setRunning] = useState<string>("");
  const [error, setError] = useState("");
  const read = reading(state, soFar, now);

  const compare = async () => {
    onCompare?.();
    setResults({});
    setError("");
    try {
      for (const p of field) {
        setRunning(p);
        const r = await api.benchmark(sessionId, p);
        setResults((x) => ({ ...x, [p]: r.metrics.J_per_day }));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning("");
    }
  };

  const done = field.every((p) => results[p] !== undefined);
  const max = Math.max(1, ...Object.values(results));
  const ranked = done ? [...field].sort((a, b) => results[a] - results[b]) : [];
  const mine = results[policy];
  const lb = results.load_balancing;

  return (
    <div className="grid grid-2">
      <section className="card">
        <h2 className="card-title" style={{ marginBottom: 10 }}>What this run shows</h2>
        {now < warmupHours ? (
          <p className="secondary" style={{ margin: 0 }}>
            Still warming up: day {Math.floor(now / 24) + 1} of 35. The hospital started empty and is filling to realistic levels;
            like the study, results are only counted after day 35.
          </p>
        ) : (
          <>
            {!soFar && <p className="muted" style={{ margin: "0 0 10px" }}>The measured period has just started; figures appear after the next step.</p>}
            <div className={`reading ${read.tone}`}><Icon name={read.tone === "bad" ? "alert" : read.tone === "good" ? "check" : "pulse"} size={18} /><span>{read.headline}</span></div>
            {soFar && (
              <div className="kpi" style={{ marginTop: 12 }}>
                <div><div className="k">Measured so far</div><div className="v">{(soFar.measured_hours / 24).toFixed(1)} days</div></div>
                <div><div className="k">Overall cost per day</div><div className="v">{soFar.cost_per_day.toFixed(0)}</div></div>
                <div><div className="k">Turned away per 100</div><div className="v">{soFar.turned_away_per_100.toFixed(1)}</div></div>
                <div><div className="k">Wrong-ward placements</div><div className="v">{soFar.wrong_ward}</div></div>
              </div>
            )}
            <p className="muted" style={{ margin: "10px 0 0" }}>
              Cost per day is the study&apos;s score: urgency-weighted waiting plus penalties for turned-away ambulances and wrong-ward
              placements. On its own it has no meaning; compare it with the others on the right.
            </p>
          </>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Verdict: how would the others have done?</h2>
        <p className="card-sub" style={{ marginBottom: 12 }}>
          The server re-runs this exact run, all 49 days with identical patients, once for each decision-maker
        </p>
        <div className="verdict-bars">
          {field.map((p) => {
            const m = policyMeta(p);
            const v = results[p];
            return (
              <div key={p} className={`vb${p === policy ? " mine" : ""}`} style={{ ["--pc" as string]: m.color }}>
                <span className="vb-name">{m.label}{p === policy ? " (this run)" : ""}</span>
                <span className="vb-track"><i style={{ width: v !== undefined ? `${(100 * v) / max}%` : "0%" }} /></span>
                <b>{v !== undefined ? v.toFixed(0) : running === p ? "running…" : "–"}</b>
              </div>
            );
          })}
        </div>
        {done && mine !== undefined && (
          <div className={`reading ${ranked[0] === policy ? "good" : "info"}`} style={{ marginTop: 12 }}>
            <Icon name="flag" size={18} />
            <span>
              On these patients, <b>{policyMeta(ranked[0]).label}</b> had the lowest cost.
              {lb !== undefined && policy !== "load_balancing" && ` ${policyMeta(policy).label} was ${Math.abs((100 * (mine - lb)) / lb).toFixed(1)}% ${mine <= lb ? "lower" : "higher"} than the balance-the-wards rule.`}
            </span>
          </div>
        )}
        {error && <p className="error">{error}</p>}
        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn glow" onClick={compare} disabled={!!running}>
            <Icon name="scale" size={15} /> {running ? `Simulating ${policyMeta(running).label}…` : done ? "Compare again" : "Compare on these exact patients"}
          </button>
        </div>
        <TechDetails>
          <p>Each bar is a full 49-day run from an empty hospital under the study protocol, with this run&apos;s situation and run number,
            so every decision-maker faces identical patients. Lower is better. One run is one sample; the study&apos;s verdict over
            50 runs per situation is on the Results page.</p>
        </TechDetails>
      </section>
    </div>
  );
}
