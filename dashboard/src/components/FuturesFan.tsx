"use client";

// "Possible futures": from this moment, the server plays the same N random futures for two
// decision-makers and returns percentile bands (neurotwin/whatif/fan.py). The chart shows the
// spread of outcomes, so a single lucky or unlucky path cannot mislead.

import { useEffect, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type FuturesResult } from "@/lib/api";
import { policyMeta } from "@/lib/policies";
import { Icon } from "./icons";
import { Legend, Segmented, TechDetails, useViewToggle } from "./ui";

const axisProps = { stroke: "var(--axis)", tick: { fill: "var(--text-muted)", fontSize: 11 }, tickLine: false } as const;
/** Futures per request: sized so 24 h ahead stays under a minute on the free server (about
 *  60 s measured for 20 futures there; requests are cut off at about 100 s). */
const N_FUTURES = 16;
const RIVALS = ["load_balancing", "priority", "trunk_reservation", "mip", "fcfs"];
type Metric = "cost" | "waiting";

interface Props {
  sessionId: string;
  /** Who is deciding in this run; compared with a rule (or, for a rule, with the latest agent). */
  policy: string;
  /** Pause live playback so the futures start from the moment shown on screen. */
  onStart?: () => void;
}

/** Hover card: for each decision-maker, the middle future and the 8-in-10 range at that hour. */
function FanTip({ active, payload, label, names, metric }: {
  active?: boolean; payload?: { payload: Record<string, number | [number, number]> }[]; label?: number;
  names: string[]; metric: Metric;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  const d = metric === "cost" ? 0 : 1;
  return (
    <div className="fan-tip">
      <div className="muted">+{Number(label)} h from now</div>
      {names.map((n) => {
        const r80 = row[`${n}_80`] as [number, number];
        return (
          <div key={n} className="fan-tip-row">
            <i className="dot" style={{ background: policyMeta(n).color }} />
            <span>{policyMeta(n).label}</span>
            <b>{Number(row[`${n}_med`]).toFixed(d)}</b>
            <span className="muted">({r80[0].toFixed(d)} to {r80[1].toFixed(d)})</span>
          </div>
        );
      })}
    </div>
  );
}

const fmtDay = (h: number) => `day ${Math.floor(h / 24) + 1}, ${String(Math.floor(h % 24)).padStart(2, "0")}:00`;

export default function FuturesFan({ sessionId, policy, onStart }: Props) {
  const own = policy;
  const [rival, setRival] = useState(policy === "load_balancing" ? "ppo_v3" : "load_balancing");
  const [hours, setHours] = useState("24");
  const [metric, setMetric] = useState<Metric>("cost");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [res, setRes] = useState<FuturesResult | null>(null);
  const [error, setError] = useState("");
  const [table, toggle] = useViewToggle();

  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const t0 = Date.now();
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 250);
    return () => window.clearInterval(id);
  }, [busy]);

  const run = async () => {
    onStart?.();
    setBusy(true); setError(""); setRes(null);
    try {
      setRes(await api.futures(sessionId, [own, rival], Number(hours), N_FUTURES));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const names = res ? Object.keys(res.series) : [];
  const rows = res ? res.hours.map((h, i) => {
    const r: Record<string, number | [number, number]> = { h };
    for (const n of names) {
      const b = res.series[n][metric];
      r[`${n}_80`] = [b.p10[i], b.p90[i]];
      r[`${n}_50`] = [b.p25[i], b.p75[i]];
      r[`${n}_med`] = b.p50[i];
    }
    return r;
  }) : [];
  const c = res?.comparison;
  const nCheaper = c ? Math.round(c.share_policy_cheaper * (res?.n_futures ?? 0)) : 0;
  const nTied = c ? Math.round(c.share_tied * (res?.n_futures ?? 0)) : 0;
  const sig = c ? (c.ci95[1] < 0 || c.ci95[0] > 0) : false;

  return (
    <div className="stack" style={{ gap: 14 }}>
      <p className="explain-line"><Icon name="split" size={16} /><span>
        Freeze this moment and play out {N_FUTURES} different futures (new arrivals, new lengths of stay) twice: once with each
        decision-maker, on exactly the same futures. The bands show the spread of what could happen.</span></p>
      <div className="fan-controls">
        <div className="fan-vs">
          <span className="chip"><i className="dot" style={{ background: policyMeta(own).color }} />{policyMeta(own).label}</span>
          <span className="muted">versus</span>
          <select value={rival} onChange={(e) => setRival(e.target.value)} aria-label="Compare with">
            {[...(own === "ppo_v3" ? [] : ["ppo_v3"]), ...RIVALS].filter((k) => k !== own).map((k) =>
              <option key={k} value={k}>{policyMeta(k).label}</option>)}
          </select>
        </div>
        <Segmented options={[{ value: "12", label: "Next 12 h" }, { value: "24", label: "24 h" }]}
          value={hours} onChange={setHours} />
        <button className="btn lg glow" onClick={run} disabled={busy}><Icon name="split" size={16} /> Play out {N_FUTURES} futures</button>
      </div>

      {busy && (
        <div className="repro-busy">
          <span className="live-dot wait" /> Simulating {Number(hours) * 2 * N_FUTURES / 24} days of hospital time on the server
          <span className="mono"> · {elapsed} s</span>
          <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>Playback is paused. On the free server this takes up to about a minute.</div>
          <div className="skel skel-chart" style={{ height: 220, marginTop: 10 }} />
        </div>
      )}
      {error && <p className="error">{error}</p>}

      {res && (
        <div className="fan-result">
          <div className="row" style={{ justifyContent: "space-between", gap: 10 }}>
            <Segmented options={[{ value: "cost", label: "Cost so far" }, { value: "waiting", label: "Patients waiting" }]}
              value={metric} onChange={(v) => setMetric(v as Metric)} />
            {toggle}
          </div>
          <Legend items={names.map((n) => ({ label: policyMeta(n).label, color: policyMeta(n).color }))} />
          {table ? (
            <div className="scroll" style={{ maxHeight: 300 }}>
              <table>
                <thead><tr><th>hours ahead</th>{names.map((n) => <th key={n}>{policyMeta(n).label}: middle (10th to 90th percentile)</th>)}</tr></thead>
                <tbody>{res.hours.map((h, i) => (
                  <tr key={h}><td>+{h} h</td>{names.map((n) => {
                    const b = res.series[n][metric];
                    return <td key={n}>{b.p50[i].toFixed(metric === "cost" ? 0 : 1)} ({b.p10[i].toFixed(0)} to {b.p90[i].toFixed(0)})</td>;
                  })}</tr>
                ))}</tbody>
              </table>
            </div>
          ) : (
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <ComposedChart data={rows} margin={{ top: 10, right: 14, bottom: 4, left: -6 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="h" type="number" domain={[0, "dataMax"]} tickFormatter={(h) => `+${h}h`} {...axisProps} />
                  <YAxis allowDecimals={false} {...axisProps} />
                  <Tooltip content={<FanTip names={names} metric={metric} />} />
                  {names.map((n, k) => (
                    <Area key={`${n}80`} dataKey={`${n}_80`} stroke="none" fill={policyMeta(n).color} fillOpacity={0.12}
                      animationDuration={900} animationBegin={k * 200} activeDot={false} legendType="none" />
                  ))}
                  {names.map((n, k) => (
                    <Area key={`${n}50`} dataKey={`${n}_50`} stroke="none" fill={policyMeta(n).color} fillOpacity={0.24}
                      animationDuration={900} animationBegin={150 + k * 200} activeDot={false} legendType="none" />
                  ))}
                  {names.map((n, k) => (
                    <Line key={`${n}m`} dataKey={`${n}_med`} stroke={policyMeta(n).color} strokeWidth={2.4} dot={false}
                      animationDuration={1100} animationBegin={300 + k * 200} />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
          <p className="ai-note">Line: the middle future. Dark band: the middle half of futures. Light band: 8 in 10 futures.
            Starting point: {fmtDay(res.start_time)}.</p>

          {c && (
            <div className={`fan-verdict ${sig ? (c.diff_mean < 0 ? "win" : "lose") : "tie"}`}>
              <div className="fan-count"><b>{nCheaper}</b><span>of {res.n_futures}</span></div>
              <div>
                <b>{policyMeta(c.policy).label}</b> ended with a lower cost than <b>{policyMeta(c.reference).label}</b> in {nCheaper} of
                {" "}{res.n_futures} futures{nTied ? ` (${nTied} tied)` : ""}. Average difference {c.diff_mean > 0 ? "+" : ""}{c.diff_mean.toFixed(0)}
                {" "}(95% interval {c.ci95[0].toFixed(0)} to {c.ci95[1].toFixed(0)}){sig ? "" : ", which is within the uncertainty"}.
                <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
                  This is one moment of one run. The study&apos;s verdict comes from 50 complete runs per situation, on the Results page.
                </div>
              </div>
            </div>
          )}
          <TechDetails>
            <p>The current state is copied; each future re-draws arrivals, staffing and remaining durations (conditional on time already elapsed)
              from its own random stream, futures 0 to {res.n_futures - 1}. Both decision-makers get the same futures (common random numbers), so the
              end-of-window costs are paired; interval = paired bootstrap. Cost = the study&apos;s cost score accumulated from this moment.
              Computed in {(res.compute_ms / 1000).toFixed(1)} s.</p>
          </TechDetails>
        </div>
      )}
    </div>
  );
}
