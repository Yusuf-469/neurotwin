"use client";

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HistoryPoint } from "@/lib/api";

interface Props {
  history: HistoryPoint[];
  warmupHours: number;
}

export default function HistoryChart({ history, warmupHours }: Props) {
  const data = history.map((h) => ({
    day: +(h.time / 24).toFixed(2),
    occupied: Object.values(h.occupied).reduce((a, b) => a + b, 0),
    staffed: Object.values(h.staffed_capacity).reduce((a, b) => a + b, 0),
    waiting: h.waiting,
  }));
  return (
    <div className="panel">
      <h2>Occupancy and queue over time</h2>
      <div className="muted">
        Days 0–{warmupHours / 24} are warm-up (not measured in the study protocol).
      </div>
      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="day" type="number" domain={["dataMin", "dataMax"]} unit=" d" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="occupied" name="beds occupied" stroke="#2563eb" dot={false} />
            <Line type="monotone" dataKey="staffed" name="staffed capacity" stroke="#94a3b8" dot={false} strokeDasharray="4 3" />
            <Line type="monotone" dataKey="waiting" name="waiting for a bed" stroke="#dc2626" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
