"use client";

// Shared run-setup controls: situation picker, custom-conditions sliders and a decision-maker
// picker grouped so the RL agent reads as one agent with versions, not several unrelated models.

import type { Conditions } from "@/lib/api";
import { LIVE_POLICIES, POLICY_GROUPS, SCENARIO_INFO, SCENARIO_LABELS, scenNum } from "@/lib/policies";
import { Icon, type IconName } from "./icons";

export const EXTRA_SCENARIOS = ["B_severe", "C_severe", "D_severe", "E_severe"];

/** "30% more patients, 10% fewer staff" summary of custom conditions. */
export function describe(c: Conditions): string {
  const part = (m: number, more: string, fewer: string) =>
    Math.abs(m - 1) < 0.005 ? "" : m > 1 ? `${Math.round((m - 1) * 100)}% ${more}` : `${Math.round((1 - m) * 100)}% ${fewer}`;
  const parts = [part(c.arrival_multiplier, "more patients", "fewer patients"),
    part(c.staff_multiplier, "more staff", "fewer staff"), part(c.capacity_multiplier, "more beds", "fewer beds")].filter(Boolean);
  return parts.length ? parts.join(", ") : "same as a normal day";
}

export function SituationSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {Object.keys(SCENARIO_LABELS).map((k) => <option key={k} value={k}>{scenNum(k)} · {SCENARIO_INFO[k].short}</option>)}
      {EXTRA_SCENARIOS.map((k) => <option key={k} value={k}>{scenNum(k)} · {SCENARIO_INFO[k[0]].short}, extreme overload</option>)}
      <option value="custom">Custom: set your own conditions</option>
    </select>
  );
}

export function ConditionsSliders({ value, onChange, note }: {
  value: Conditions; onChange: (c: Conditions) => void; note?: string;
}) {
  return (
    <div className="custom-panel">
      {([
        ["arrival_multiplier", "Patients arriving", 0.5, 1.6, "users"],
        ["staff_multiplier", "Nurses and doctors", 0.6, 1.4, "staff"],
        ["capacity_multiplier", "Beds available", 0.6, 1.4, "bed"],
      ] as [keyof Conditions, string, number, number, IconName][]).map(([key, label, lo, hi, icon]) => (
        <label key={key} className="slider">
          <span className="slider-head"><Icon name={icon} size={15} />{label}
            <b>{value[key] === 1 ? "normal" : `${value[key] > 1 ? "+" : "−"}${Math.round(Math.abs(value[key] - 1) * 100)}%`}</b></span>
          <input type="range" min={lo} max={hi} step={0.05} value={value[key]}
            onChange={(e) => onChange({ ...value, [key]: +e.target.value })} />
          <span className="slider-scale"><span>{Math.round(lo * 100)}%</span><span>normal</span><span>{Math.round(hi * 100)}%</span></span>
        </label>
      ))}
      <p className="muted" style={{ margin: 0 }}>
        Your situation: <b>{describe(value)}</b>. {note ?? "The agent never trained on this, so it is a genuine test."}
      </p>
    </div>
  );
}

/** Decision-maker <select> with the RL agent's versions grouped together. */
export function PolicySelect({ value, onChange, available }: {
  value: string; onChange: (v: string) => void; available: string[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {POLICY_GROUPS.map((g) => {
        const items = LIVE_POLICIES.filter((p) => g.keys.includes(p.key) && available.includes(p.key));
        return items.length ? (
          <optgroup key={g.label} label={g.label}>
            {items.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
          </optgroup>
        ) : null;
      })}
    </select>
  );
}
