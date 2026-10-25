import type { HospitalState } from "@/lib/api";
import { PRIORITY_COLORS } from "@/lib/policies";

const WARDS: { key: string; name: string; note: string }[] = [
  { key: "MED", name: "Medical ward", note: "general beds" },
  { key: "SURG", name: "Surgical ward", note: "general beds" },
  { key: "ICU", name: "Intensive care", note: "critical-care beds" },
];

const TYPE_LABELS: Record<string, string> = {
  admit_general: "awaiting ward bed",
  admit_icu: "awaiting ICU bed",
  step_down: "ICU step-down",
  escalation: "deteriorated → ICU",
};

function Ward({ state, unit, name, note }: { state: HospitalState; unit: string; name: string; note: string }) {
  const physical = state.physical_beds[unit];
  const staffed = state.staffed_capacity[unit];
  const occupied = state.occupied[unit];
  const pct = staffed > 0 ? Math.min(100, (100 * occupied) / staffed) : 100;
  const beds = Array.from({ length: Math.max(physical, occupied) }, (_, i) =>
    i < occupied ? "bed occ" : i < staffed ? "bed" : "bed unstaffed");
  return (
    <div className="ward">
      <div className="ward-head">
        <span className="ward-name">{name}</span>
        <span className="muted">{note}</span>
        <span className="spacer" />
        <strong>{occupied}</strong><span className="muted">/ {staffed} staffed</span>
      </div>
      <div className="meter" aria-hidden="true"><div style={{ width: `${pct}%` }} /></div>
      <div className="beds" style={{ marginTop: 10 }} role="img"
        aria-label={`${name}: ${occupied} of ${staffed} staffed beds occupied, ${physical} physical beds`}>
        {beds.map((cls, i) => <div key={i} className={cls} />)}
      </div>
      <div className="muted" style={{ marginTop: 8 }}>
        {state.nurses_on_shift[unit]} nurses on shift · {Math.round(pct)}% of staffed capacity
      </div>
    </div>
  );
}

export default function HospitalMap({ state }: { state: HospitalState }) {
  const byType: Record<string, number> = {};
  for (const r of state.waiting_requests) byType[r.type] = (byType[r.type] ?? 0) + 1;
  const sorted = [...state.waiting_requests].sort((a, b) => a.priority - b.priority || a.created_at - b.created_at);
  const urgent = state.waiting_requests.filter((r) => r.priority <= 2).length;
  const edPct = Math.min(100, (100 * state.ed_count) / 30);

  return (
    <div className="stack">
      <div className="ward">
        <div className="ward-head">
          <span className="ward-name">Emergency department & bed queue</span>
          <span className="spacer" />
          <span className="muted">ED occupancy</span><strong>{state.ed_count}</strong><span className="muted">/ 30</span>
        </div>
        <div className="meter" aria-hidden="true">
          <div style={{ width: `${edPct}%`, background: edPct >= 100 ? "var(--status-critical)" : "var(--series-1)" }} />
        </div>
        <div className="row" style={{ margin: "10px 0 8px" }}>
          <span className="chip">doctors {state.ed_in_treatment}/{state.doctors_on_shift} busy</span>
          <span className="chip">{state.ed_waiting_for_doctor} waiting for a doctor</span>
          {Object.entries(byType).map(([t, n]) => (
            <span className="chip" key={t}>{n} {TYPE_LABELS[t] ?? t}</span>
          ))}
          {edPct >= 100 && <span className="chip" style={{ color: "var(--status-critical)" }}>⚠ ED full: arrivals diverted</span>}
        </div>
        <div className="queue" role="img" aria-label={`${sorted.length} patients waiting for a bed, ${urgent} urgent`}>
          {sorted.slice(0, 160).map((r) => (
            <span key={r.rid} className="qdot" title={`priority ${r.priority} · ${TYPE_LABELS[r.type] ?? r.type}`}
              style={{ background: PRIORITY_COLORS[r.priority] }} />
          ))}
          {sorted.length === 0 && <span className="muted">No one waiting for a bed.</span>}
          {sorted.length > 160 && <span className="muted">+{sorted.length - 160} more</span>}
        </div>
        <div className="legend" style={{ marginTop: 8 }}>
          <span><i className="dot" style={{ background: PRIORITY_COLORS[1] }} />urgent (P1–2)</span>
          <span><i className="dot" style={{ background: PRIORITY_COLORS[3] }} />P3</span>
          <span><i className="dot" style={{ background: PRIORITY_COLORS[4] }} />P4–5</span>
        </div>
      </div>
      <div className="grid grid-3">
        {WARDS.map((w) => <Ward key={w.key} state={state} unit={w.key} name={w.name} note={w.note} />)}
      </div>
      <div className="legend">
        <span><i className="swatch" style={{ background: "var(--series-1)" }} />occupied</span>
        <span><i className="swatch" style={{ background: "var(--bed-free)", border: "1px solid var(--grid)" }} />free (staffed)</span>
        <span><i className="swatch" style={{ background: "repeating-linear-gradient(45deg, var(--bed-unstaffed) 0 3px, transparent 3px 6px)", border: "1px solid var(--grid)" }} />no nurse to staff it</span>
      </div>
    </div>
  );
}
