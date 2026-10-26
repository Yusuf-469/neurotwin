"use client";

import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { HistoryPoint } from "@/lib/api";
import { UNIT_COLORS, UNIT_LABELS } from "@/lib/policies";
import { Card, Legend, useViewToggle } from "./ui";

interface Props {
  history: HistoryPoint[];
  warmupHours: number;
}

const axisProps = {
  stroke: "var(--axis)",
  tick: { fill: "var(--text-muted)", fontSize: 11 },
  tickLine: false,
} as const;

const tooltipStyle = {
  contentStyle: {
    background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 10,
    fontSize: 12, color: "var(--text-primary)", boxShadow: "var(--shadow)",
  },
  labelStyle: { color: "var(--text-muted)" },
  cursor: { stroke: "var(--axis)", strokeWidth: 1 },
};

export default function LiveCharts({ history, warmupHours }: Props) {
  const [occTable, occToggle] = useViewToggle();
  const [qTable, qToggle] = useViewToggle();
  const warmDay = warmupHours / 24;
  const data = history.map((h) => {
    const pct = (u: string) => {
      const s = h.staffed_capacity[u];
      return s > 0 ? +((100 * h.occupied[u]) / s).toFixed(1) : 0;
    };
    return {
      day: +(h.time / 24).toFixed(2), MED: pct("MED"), SURG: pct("SURG"), ICU: pct("ICU"),
      waiting: h.waiting, diversions: h.cumulative_diversions, offService: h.cumulative_off_service,
    };
  });
  const last = data[data.length - 1];
  const warm = last && last.day > 0 ? (
    <ReferenceArea x1={0} x2={Math.min(warmDay, last.day)} fill="var(--surface-2)" fillOpacity={0.9}
      ifOverflow="hidden" label={{ value: "warm-up", position: "insideTopLeft", fill: "var(--text-muted)", fontSize: 11 }} />
  ) : null;

  return (
    <div className="grid grid-2">
      <Card title="How full each ward is" subtitle="% of beds that have nurses to staff them" actions={occToggle}
        explain="The shaded strip is the warm-up: the hospital starts empty and fills up before measurement begins.">
        <Legend items={["MED", "SURG", "ICU"].map((u) => ({ label: UNIT_LABELS[u], color: UNIT_COLORS[u] }))} />
        {occTable ? (
          <div className="scroll" style={{ maxHeight: 260, marginTop: 8 }}>
            <table><thead><tr><th>day</th><th>MED %</th><th>SURG %</th><th>ICU %</th></tr></thead>
              <tbody>{[...data].reverse().map((d) => (
                <tr key={d.day}><td>{d.day}</td><td>{d.MED}</td><td>{d.SURG}</td><td>{d.ICU}</td></tr>))}
              </tbody></table>
          </div>
        ) : (
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer>
              <LineChart data={data} margin={{ top: 12, right: 12, bottom: 4, left: -12 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                {warm}
                <XAxis dataKey="day" type="number" domain={[0, "dataMax"]} unit="d" {...axisProps} />
                {/* Can exceed 100%: a shift change may remove nurses while beds stay occupied. */}
                <YAxis domain={[0, (max: number) => Math.max(100, Math.ceil(max / 25) * 25)]}
                  allowDecimals={false} unit="%" {...axisProps} />
                <Tooltip {...tooltipStyle} formatter={(v) => (v === undefined ? "–" : `${v}%`)} labelFormatter={(d) => `day ${d}`} />
                {(["MED", "SURG", "ICU"] as const).map((u) => (
                  <Line key={u} type="monotone" dataKey={u} stroke={UNIT_COLORS[u]} strokeWidth={2} dot={false}
                    isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
      <Card title="Patients waiting for a bed" subtitle="queue length over time" actions={qToggle}
        explain="Spikes mean demand briefly outran discharges. A steadily rising line means the hospital is overwhelmed.">
        {qTable ? (
          <div className="scroll" style={{ maxHeight: 260 }}>
            <table><thead><tr><th>day</th><th>waiting</th><th>diversions (cum.)</th><th>off-service (cum.)</th></tr></thead>
              <tbody>{[...data].reverse().map((d) => (
                <tr key={d.day}><td>{d.day}</td><td>{d.waiting}</td><td>{d.diversions}</td><td>{d.offService}</td></tr>))}
              </tbody></table>
          </div>
        ) : (
          <div style={{ width: "100%", height: 282 }}>
            <ResponsiveContainer>
              <AreaChart data={data} margin={{ top: 12, right: 12, bottom: 4, left: -12 }}>
                <defs>
                  <linearGradient id="qfill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--series-1)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--series-1)" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                {warm}
                <XAxis dataKey="day" type="number" domain={[0, "dataMax"]} unit="d" {...axisProps} />
                <YAxis allowDecimals={false} {...axisProps} />
                <Tooltip {...tooltipStyle} labelFormatter={(d) => `day ${d}`} />
                <Area type="monotone" dataKey="waiting" name="waiting" stroke="var(--series-1)" strokeWidth={2}
                  fill="url(#qfill)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
    </div>
  );
}
