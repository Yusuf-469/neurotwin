"use client";

// Animated floor plan of the simulated hospital. Everything shown is driven by the real
// simulation state from the API on each tick:
//   * bed fill per ward = occupied / staffed / physical beds;
//   * patients walking from the ED to a ward = the PLACE decisions logged since the last tick;
//   * ambulances driving past the entrance = diversions since the last tick;
//   * the ED and the "waiting for a bed" bay = current ED count and waiting requests.
// Which *individual* bed a patient goes to is not modelled by the simulator (it tracks counts),
// so bed positions are assigned visually with a fixed-seed shuffle; counts are always exact.

import { useEffect, useMemo, useRef, useState } from "react";
import type { DecisionRecord, HospitalState } from "@/lib/api";
import { PRIORITY_COLORS, PRIORITY_NAMES, UNIT_COLORS, UNIT_LABELS } from "@/lib/policies";

const W = 1200;
const H = 660;
const ED_CAPACITY = 30;

interface WardGeom { unit: string; y0: number; y1: number; cols: number; bw: number; bh: number; px: number; py: number }
const WARDS: WardGeom[] = [
  { unit: "MED", y0: 30, y1: 236, cols: 18, bw: 27, bh: 31, px: 36, py: 39 },
  { unit: "SURG", y0: 250, y1: 456, cols: 18, bw: 27, bh: 31, px: 36, py: 39 },
  { unit: "ICU", y0: 470, y1: 630, cols: 12, bw: 40, bh: 78, px: 54, py: 0 },
];
const WX0 = 500;
const WX1 = 1185;
const CORR_X = 470;
const BAY = { x: 146, y: 438, cols: 8, px: 34, py: 34 };

type BedState = "occ" | "free";
interface BedVisual { state: BedState; arrive?: number; leave?: boolean }

const wardOf = (u: string) => WARDS.find((w) => w.unit === u)!;
const doorY = (w: WardGeom) => (w.y0 + w.y1) / 2;
function bedXY(w: WardGeom, i: number): [number, number] {
  const c = i % w.cols;
  const r = Math.floor(i / w.cols);
  return [520 + c * w.px, w.y0 + 44 + r * w.py];
}

/** Deterministic PRNG so bed choices are stable between renders. */
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Mover { id: number; pts: [number, number][]; color: string; start: number; dur: number; label: string }
interface Ambulance { id: number; start: number; kind: "divert" | "arrive" }

const MOVE_MS = 1500;

function along(pts: [number, number][], k: number): [number, number] {
  const segs = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
  const total = segs.reduce((a, b) => a + b, 0) || 1;
  let d = k * total;
  for (let i = 0; i < segs.length; i++) {
    if (d <= segs[i] || i === segs.length - 1) {
      const t = segs[i] ? Math.min(1, d / segs[i]) : 1;
      return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t];
    }
    d -= segs[i];
  }
  return pts[pts.length - 1];
}
const ease = (k: number) => (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);

interface Props {
  state: HospitalState;
  /** Decisions made since the previous frame (oldest first). */
  newDecisions: DecisionRecord[];
  newDiversions: number;
  playing: boolean;
  hour: number;
  tickMs: number;
}

export default function FloorPlan({ state, newDecisions, newDiversions, playing, hour, tickMs }: Props) {
  const beds = useRef<Record<string, BedVisual[]>>({});
  const rng = useRef(mulberry(20261027));
  const [movers, setMovers] = useState<Mover[]>([]);
  const [ambulances, setAmbulances] = useState<Ambulance[]>([]);
  const [, setFrame] = useState(0);
  const nextId = useRef(1);
  const lastStateTime = useRef<number>(-1);

  // ---- update bed visuals and spawn movers when a new state arrives -----------------------
  const now = typeof performance !== "undefined" ? performance.now() : 0;
  if (state.time !== lastStateTime.current) {
    const first = lastStateTime.current < 0;
    lastStateTime.current = state.time;
    const spawned: Mover[] = [];
    const spread = Math.max(400, tickMs - MOVE_MS);
    const placements = newDecisions.filter((d) => d.action === "PLACE" && d.unit).slice(-28);
    for (const w of WARDS) {
      const n = Math.max(state.physical_beds[w.unit] ?? 0, state.occupied[w.unit] ?? 0);
      let arr = beds.current[w.unit] ?? [];
      if (arr.length !== n) arr = Array.from({ length: n }, (_, i) => arr[i] ?? { state: "free" as BedState });
      arr = arr.map((b) => ({ state: b.state }));
      const target = state.occupied[w.unit] ?? 0;
      const occIdx = () => arr.map((b, i) => (b.state === "occ" ? i : -1)).filter((i) => i >= 0);
      const freeIdx = () => arr.map((b, i) => (b.state === "free" ? i : -1)).filter((i) => i >= 0);
      let occ = occIdx();
      while (occ.length > target) { // discharges
        const i = occ[Math.floor(rng.current() * occ.length)];
        arr[i] = { state: "free", leave: !first };
        occ = occIdx();
      }
      const arrivedBeds: number[] = [];
      while (occ.length < target) { // admissions
        const free = freeIdx();
        const i = free[Math.floor(rng.current() * free.length)];
        arr[i] = { state: "occ" };
        arrivedBeds.push(i);
        occ = occIdx();
      }
      if (!first) {
        const mine = placements.filter((d) => d.unit === w.unit);
        mine.forEach((d, j) => {
          const bed = arrivedBeds[j] ?? occ[Math.floor(rng.current() * Math.max(1, occ.length))] ?? 0;
          const [bx, by] = bedXY(w, bed);
          const dest: [number, number] = [bx + w.bw / 2, by + w.bh / 2];
          let start: [number, number] = [BAY.x + 120, BAY.y + 60];
          if (d.request_type === "step_down") start = [WX0 + 30, doorY(wardOf("ICU"))];
          if (d.request_type === "escalation") start = [WX0 + 30, doorY(wardOf(d.primary_unit === "SURG" ? "SURG" : "MED"))];
          const delay = (placements.indexOf(d) / Math.max(1, placements.length)) * spread;
          const pts: [number, number][] = [start, [CORR_X, start[1]], [CORR_X, doorY(w)], [WX0 + 14, doorY(w)], dest];
          spawned.push({ id: nextId.current++, pts, color: PRIORITY_COLORS[d.priority ?? 5], start: now + delay, dur: MOVE_MS,
            label: `${PRIORITY_NAMES[d.priority ?? 5]} → ${UNIT_LABELS[w.unit]}` });
          if (j < arrivedBeds.length) arr[arrivedBeds[j]] = { state: "occ", arrive: delay + MOVE_MS };
        });
      }
      beds.current[w.unit] = arr;
    }
    if (spawned.length) setTimeout(() => setMovers((m) => [...m, ...spawned]), 0);
    if (!first) {
      const amb: Ambulance[] = [];
      for (let i = 0; i < Math.min(5, newDiversions); i++) amb.push({ id: nextId.current++, start: now + i * 420, kind: "divert" });
      if (playing && state.ed_count < ED_CAPACITY) amb.push({ id: nextId.current++, start: now + 200, kind: "arrive" });
      if (amb.length) setTimeout(() => setAmbulances((a) => [...a, ...amb]), 0);
    }
  }

  // ---- animation loop ---------------------------------------------------------------------
  const active = movers.length > 0 || ambulances.length > 0;
  useEffect(() => {
    if (!active) return;
    let raf = 0;
    const loop = () => {
      const t = performance.now();
      setMovers((m) => (m.some((x) => t > x.start + x.dur + 50) ? m.filter((x) => t <= x.start + x.dur + 50) : m));
      setAmbulances((a) => (a.some((x) => t > x.start + 2600) ? a.filter((x) => t <= x.start + 2600) : a));
      setFrame((f) => (f + 1) % 1_000_000);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  const sortedQueue = useMemo(
    () => [...state.waiting_requests].sort((a, b) => a.priority - b.priority || a.created_at - b.created_at),
    [state.waiting_requests],
  );
  const night = hour < 7 || hour >= 20;
  const edFull = state.ed_count >= ED_CAPACITY;
  const t = typeof performance !== "undefined" ? performance.now() : 0;

  return (
    <svg className={`floorplan${night ? " night" : ""}`} viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Hospital floor plan: ${state.ed_count} patients in the emergency department, ${state.waiting_requests.length} waiting for a bed`}>
      <defs>
        <pattern id="unstaffed" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--surface-2)" /><line x1="0" y1="0" x2="0" y2="6" stroke="var(--grid)" strokeWidth="3" />
        </pattern>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        <linearGradient id="road" x1="0" x2="1"><stop offset="0" stopColor="var(--surface-2)" /><stop offset="1" stopColor="var(--grid)" /></linearGradient>
      </defs>

      {/* road + driveway */}
      <rect x="0" y="0" width="82" height={H} fill="url(#road)" />
      <line x1="41" y1="0" x2="41" y2={H} className="fp-lane" />
      <rect x="82" y="300" width="30" height="60" fill="var(--grid)" />
      <text x="41" y="20" className="fp-small" textAnchor="middle">ROAD</text>

      {/* emergency department */}
      <rect x="110" y="30" width="330" height="600" rx="18" className="fp-room" />
      <text x="132" y="62" className="fp-title">Emergency department</text>
      <text x="420" y="62" className="fp-sub" textAnchor="end">{state.ed_count} / {ED_CAPACITY} patients</text>
      <rect x="132" y="72" width="288" height="5" rx="2.5" fill="var(--grid)" />
      <rect x="132" y="72" width={288 * Math.min(1, state.ed_count / ED_CAPACITY)} height="5" rx="2.5"
        fill={edFull ? "var(--status-critical)" : "var(--accent)"} className="fp-bar" />
      {Array.from({ length: ED_CAPACITY }, (_, i) => {
        const x = 150 + (i % 6) * 46;
        const y = 106 + Math.floor(i / 6) * 38;
        const on = i < state.ed_count;
        return <circle key={i} cx={x} cy={y} r={on ? 10 : 9} className={on ? "fp-ed on" : "fp-ed"} />;
      })}
      <text x="132" y="316" className="fp-sub">Doctors on shift</text>
      {Array.from({ length: state.doctors_on_shift }, (_, i) => {
        const busy = i < state.ed_in_treatment;
        return (
          <g key={i} transform={`translate(${142 + i * 34}, 332)`} className={busy ? "fp-doc busy" : "fp-doc"}>
            <circle cx="10" cy="6" r="6" /><path d="M0 26c0-7 4.5-11 10-11s10 4 10 11z" />
          </g>
        );
      })}
      <text x="420" y="316" className="fp-sub" textAnchor="end">{state.ed_waiting_for_doctor} waiting for a doctor</text>

      <rect x="128" y="384" width="294" height="230" rx="12" className="fp-bay" />
      <text x="142" y="410" className="fp-sub strong">Waiting for a ward bed · {state.waiting_requests.length}</text>
      {sortedQueue.slice(0, 40).map((r, i) => (
        <circle key={r.rid} cx={BAY.x + (i % BAY.cols) * BAY.px} cy={BAY.y + Math.floor(i / BAY.cols) * BAY.py} r="10"
          fill={PRIORITY_COLORS[r.priority]} className="fp-wait">
          <title>{`${PRIORITY_NAMES[r.priority]} patient, waiting for ${UNIT_LABELS[r.primary_unit] ?? r.primary_unit}`}</title>
        </circle>
      ))}
      {sortedQueue.length > 40 && <text x="410" y="604" className="fp-sub" textAnchor="end">+{sortedQueue.length - 40} more</text>}
      {sortedQueue.length === 0 && <text x="275" y="510" className="fp-sub" textAnchor="middle">Nobody waiting</text>}

      {edFull && (
        <g className="fp-full">
          <rect x="118" y="268" width="150" height="26" rx="6" fill="var(--status-critical)" />
          <text x="193" y="286" textAnchor="middle" className="fp-full-text">ED FULL · DIVERTING</text>
        </g>
      )}

      {/* corridor */}
      <rect x="440" y="30" width="60" height="600" className="fp-corridor" />
      <line x1={CORR_X} y1="40" x2={CORR_X} y2="620" className="fp-lane" />

      {/* wards */}
      {WARDS.map((w) => {
        const arr = beds.current[w.unit] ?? [];
        const staffed = state.staffed_capacity[w.unit] ?? 0;
        const occupied = state.occupied[w.unit] ?? 0;
        // Free beds beyond the staffed capacity have no nurse: draw the highest-index free beds as unstaffed.
        const freeIdx = arr.map((b, i) => (b.state === "free" ? i : -1)).filter((i) => i >= 0);
        const nUnstaffed = Math.max(0, arr.length - Math.max(staffed, occupied));
        const unstaffed = new Set(freeIdx.slice(freeIdx.length - nUnstaffed));
        const color = UNIT_COLORS[w.unit];
        return (
          <g key={w.unit}>
            <rect x={WX0} y={w.y0} width={WX1 - WX0} height={w.y1 - w.y0} rx="16" className="fp-room" />
            <rect x={WX0 - 2} y={doorY(w) - 16} width="8" height="32" rx="3" fill="var(--page)" />
            <text x={WX0 + 20} y={w.y0 + 28} className="fp-title">{UNIT_LABELS[w.unit]}</text>
            <text x={WX1 - 16} y={w.y0 + 28} className="fp-sub" textAnchor="end">
              {occupied} occupied · {staffed} staffed · {state.nurses_on_shift[w.unit]} nurses
            </text>
            {arr.map((b, i) => {
              const [x, y] = bedXY(w, i);
              const cls = unstaffed.has(i) ? "fp-bed unstaffed" : b.state === "occ" ? `fp-bed occ${b.arrive ? " arrive" : ""}` : `fp-bed free${b.leave ? " leave" : ""}`;
              return (
                <g key={i} className={cls} style={{ ["--c" as string]: color, ["--delay" as string]: `${b.arrive ?? 0}ms` }}>
                  <rect x={x} y={y} width={w.bw} height={w.bh} rx="5" className="body" />
                  <rect x={x + 4} y={y + 3} width={w.bw - 8} height={w.unit === "ICU" ? 10 : 6} rx="2.5" className="pillow" />
                  <circle cx={x + w.bw / 2} cy={y + (w.unit === "ICU" ? 26 : 16)} r={w.unit === "ICU" ? 6 : 4.2} className="head" />
                </g>
              );
            })}
          </g>
        );
      })}

      {/* night tint */}
      <rect x="0" y="0" width={W} height={H} className="fp-night" pointerEvents="none" />

      {/* moving patients */}
      {movers.map((m) => {
        const k = Math.max(0, Math.min(1, (t - m.start) / m.dur));
        if (t < m.start) return null;
        const [x, y] = along(m.pts, ease(k));
        return (
          <g key={m.id} transform={`translate(${x}, ${y})`} className="fp-mover">
            <circle r="13" fill={m.color} opacity="0.25" filter="url(#glow)" />
            <circle r="7.5" fill={m.color} stroke="var(--surface-1)" strokeWidth="2" />
          </g>
        );
      })}

      {/* ambulances */}
      {ambulances.map((a) => {
        const k = (t - a.start) / (a.kind === "divert" ? 2400 : 2000);
        if (k < 0 || k > 1) return null;
        let x = 41, y = -40, op = 1;
        if (a.kind === "divert") { y = -40 + ease(k) * (H + 80); }
        else if (k < 0.6) { y = -40 + ease(k / 0.6) * 370; }
        else { y = 330; x = 41 + ((k - 0.6) / 0.4) * 60; op = 1 - (k - 0.6) / 0.4; }
        return (
          <g key={a.id} transform={`translate(${x}, ${y})`} opacity={op}>
            <g transform={a.kind === "arrive" && k >= 0.6 ? "rotate(-90)" : undefined}>
              <rect x="-13" y="-22" width="26" height="44" rx="6" fill="#f4f4f2" stroke="var(--axis)" />
              <rect x="-10" y="10" width="20" height="9" rx="2" fill="#9cc3ef" />
              <path d="M-2 -10h4v4h4v4h-4v4h-4v-4h-4v-4h4z" fill="#d03b3b" />
              <rect x="-11" y="-21" width="22" height="3" fill={a.kind === "divert" ? "#d03b3b" : "#3987e5"} className="fp-siren" />
            </g>
            {a.kind === "divert" && (
              <g transform="translate(20, 0)">
                <rect x="0" y="-11" width="96" height="22" rx="6" fill="var(--status-critical)" />
                <text x="48" y="4" textAnchor="middle" className="fp-full-text">TURNED AWAY</text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
