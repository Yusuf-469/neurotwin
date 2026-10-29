"use client";

// Training replay: the logged learning curves of each agent generation, played back over the
// 3 million training steps (neurotwin/experiments/training_curves.py). Curves only: network
// snapshots from mid-training were not kept, so this shows how learning progressed.

import { useEffect, useRef, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from "recharts";
import type { TrainingCurves } from "@/lib/api";
import { Icon } from "./icons";
import { Segmented, TechDetails } from "./ui";

type Metric = "episode_reward" | "entropy" | "explained_variance";

const METRICS: Record<Metric, { label: string; what: string; digits: number }> = {
  episode_reward: { label: "Reward", what: "reward per training day (higher = lower cost)", digits: 0 },
  entropy: { label: "Decisiveness", what: "uncertainty of its choices (lower = more decisive)", digits: 2 },
  explained_variance: { label: "Foresight", what: "how well its critic predicts what comes next (1 = perfect)", digits: 2 },
};

// Captions describe the plotted (smoothed) medians; numbers checked against training_curves.json.
const STORY: Record<string, Partial<Record<Metric, string>>> = {
  v1: {
    episode_reward: "From random choices to a working policy: the reward improves from about -480 to about -360, most of it within the first 800,000 steps; after that it levels off.",
    entropy: "It starts close to random (1.15) and becomes steadily more decisive (0.38) as it learns.",
    explained_variance: "Its critic learns fast, from 0.10 to above 0.9 within about 220,000 steps. That is what turns raw rewards into a useful learning signal.",
  },
  v2: {
    episode_reward: "It starts higher (about -420) because it began as a copy of the best rule, and reinforcement learning keeps improving it, to about -354.",
    entropy: "A copy of a fixed rule starts very decisive (0.17); reinforcement learning makes it explore a little more (0.2 to 0.35) while it improves.",
    explained_variance: "The critic starts untrained, because copying the rule teaches only the choices, and learns within about 400,000 steps.",
  },
  v3: {
    episode_reward: "Hard to read by design: every training day had randomly harder conditions, so the reward mostly reflects how hard each day was. Its improvement was checked on separate validation runs instead.",
    entropy: "Practising varied days makes it a little more decisive (0.50 to 0.39).",
    explained_variance: "Its critic was already accurate, inherited from v2, and stays so as conditions vary (0.95 to 0.98).",
  },
};

const VERSION_COLOR: Record<string, string> = { v1: "var(--series-1)", v2: "var(--violet)", v3: "var(--accent)" };
const PLAY_MS = 6000;

export default function TrainingReplay({ data }: { data: TrainingCurves }) {
  const versions = Object.keys(data.versions);
  const [version, setVersion] = useState(versions[0] ?? "v1");
  const [metric, setMetric] = useState<Metric>("episode_reward");
  const [pos, setPos] = useState(data.bins);
  const [playing, setPlaying] = useState(false);
  const frame = useRef(0);

  const v = data.versions[version];
  const c = v.curves[metric];
  const color = VERSION_COLOR[version] ?? "var(--accent)";

  // play: sweep the playhead from 0 to the end over PLAY_MS
  useEffect(() => {
    if (!playing) return;
    const t0 = performance.now();
    const start = pos >= data.bins ? 0 : pos;
    const tick = (t: number) => {
      const p = Math.min(data.bins, start + Math.round(((t - t0) / PLAY_MS) * data.bins));
      setPos(p);
      if (p < data.bins) frame.current = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // replay automatically when the version or the curve changes
  useEffect(() => { setPos(0); setPlaying(true); }, [version, metric]);

  const rows = v.step.map((s, i) => ({
    s,
    median: i < pos ? c.median[i] : null,
    range: i < pos && c.low[i] !== null && c.high[i] !== null ? [c.low[i], c.high[i]] : null,
    ghost: c.median[i],
  }));
  const at = Math.max(0, Math.min(pos, data.bins) - 1);
  const now = c.median[at];
  const meta = METRICS[metric];

  return (
    <div className="replay">
      <div className="row" style={{ justifyContent: "space-between", gap: 10 }}>
        <Segmented options={versions.map((k) => ({ value: k, label: `Agent ${k}` }))} value={version} onChange={setVersion} />
        <Segmented options={(Object.keys(METRICS) as Metric[]).map((k) => ({ value: k, label: METRICS[k].label }))}
          value={metric} onChange={(m) => setMetric(m as Metric)} />
      </div>
      <p className="replay-story"><b>Agent {version}</b>: {v.story}.</p>

      <div className="replay-readout">
        <div><span className="eyebrow">Training step</span><b className="mono">{(at >= 0 ? v.step[at] : 0).toLocaleString("en-GB")}</b></div>
        <div><span className="eyebrow">{meta.label}</span><b className="mono">{now === null || now === undefined ? "–" : now.toFixed(meta.digits)}</b></div>
        <div className="muted" style={{ fontSize: 13 }}>{meta.what}</div>
      </div>

      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="s" type="number" domain={[0, v.total_steps]} stroke="var(--axis)" tickLine={false}
              tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickFormatter={(s) => `${(s / 1e6).toFixed(1)}M`} />
            <YAxis stroke="var(--axis)" tickLine={false} tick={{ fill: "var(--text-muted)", fontSize: 11 }} width={52}
              domain={["auto", "auto"]} tickFormatter={(x) => Number(x).toFixed(meta.digits)} />
            <Line dataKey="ghost" stroke="var(--axis)" strokeDasharray="3 4" strokeWidth={1} dot={false} isAnimationActive={false} />
            <Area dataKey="range" stroke="none" fill={color} fillOpacity={0.16} isAnimationActive={false} connectNulls={false} />
            <Line dataKey="median" stroke={color} strokeWidth={2.6} dot={false} isAnimationActive={false} connectNulls={false} />
            {at >= 0 && <ReferenceLine x={v.step[at]} stroke={color} strokeOpacity={0.5} />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="replay-controls">
        <button className="btn ghost" onClick={() => { if (pos >= data.bins) setPos(0); setPlaying((p) => !p); }}
          aria-label={playing ? "Pause" : "Play"}>
          <Icon name={playing ? "pause" : "play"} size={15} /> {playing ? "Pause" : pos >= data.bins ? "Replay" : "Play"}
        </button>
        <input type="range" min={0} max={data.bins} value={pos} aria-label="Training progress"
          onChange={(e) => { setPlaying(false); setPos(Number(e.target.value)); }} />
      </div>

      <p className="ai-note">Line: middle of the {v.n_agents} independently trained agents. Band: the range across them. Dashed: the full curve.{metric === "episode_reward" ? " Reward is averaged over about 200,000 steps, because training days differ a lot in difficulty." : ""}</p>
      {STORY[version]?.[metric] && <div key={version + metric} className="replay-caption">{STORY[version][metric]}</div>}
      <TechDetails>
        <p>{data.note} Logged over {v.total_steps.toLocaleString("en-GB")} environment steps per agent, condensed to {data.bins} points
          (median and min-max over agents per bin). Training reward uses sampled (not greedy) actions.</p>
      </TechDetails>
    </div>
  );
}
