import type { HospitalState } from "@/lib/api";

interface Props {
  state: HospitalState;
}

const UNITS = ["MED", "SURG", "ICU"] as const;

export default function StatePanel({ state }: Props) {
  const byType: Record<string, number> = {};
  for (const r of state.waiting_requests) byType[r.type] = (byType[r.type] ?? 0) + 1;
  const urgent = state.waiting_requests.filter((r) => r.priority <= 2).length;

  return (
    <div className="grid">
      {UNITS.map((u) => {
        const staffed = state.staffed_capacity[u];
        const occ = state.occupied[u];
        const pct = staffed > 0 ? Math.min(100, (100 * occ) / staffed) : 100;
        return (
          <div className="panel" key={u}>
            <h2>{u}</h2>
            <div className="stat">
              {occ} / {staffed}
            </div>
            <div className="muted">occupied / staffed beds (physical {state.physical_beds[u]})</div>
            <div className="bar" style={{ marginTop: 8 }}>
              <div style={{ width: `${pct}%` }} />
            </div>
            <div className="muted">nurses on shift: {state.nurses_on_shift[u]}</div>
          </div>
        );
      })}
      <div className="panel">
        <h2>Waiting for a bed</h2>
        <div className="stat">{state.waiting_requests.length}</div>
        <div className="muted">
          {urgent} urgent (P1–2) · {Object.entries(byType).map(([t, n]) => `${t.replace("_", " ")}: ${n}`).join(" · ") || "none"}
        </div>
      </div>
      <div className="panel">
        <h2>Emergency department</h2>
        <div className="stat">{state.ed_count}</div>
        <div className="muted">
          waiting for doctor {state.ed_waiting_for_doctor} · in treatment {state.ed_in_treatment} ·
          doctors on shift {state.doctors_on_shift}
        </div>
      </div>
      <div className="panel">
        <h2>Cumulative</h2>
        <div className="muted">
          placements {state.cumulative_placements} · diversions {state.cumulative_diversions} ·
          off-service {state.cumulative_off_service} (in beds now: {state.off_service_in_beds})
        </div>
      </div>
    </div>
  );
}
