"use client";

import { useEffect, useState, type ReactNode } from "react";

interface CardProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}

export function Card({ title, subtitle, actions, children }: CardProps) {
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2 className="card-title">{title}</h2>
          {subtitle && <p className="card-sub">{subtitle}</p>}
        </div>
        <div className="spacer" />
        {actions}
      </div>
      {children}
    </section>
  );
}

interface StatProps {
  label: string;
  value: string;
  note?: string;
}

export function Stat({ label, value, note }: StatProps) {
  return (
    <div className="card stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

interface SegmentedProps<T extends string> {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}

export function Segmented<T extends string>({ options, value, onChange }: SegmentedProps<T>) {
  return (
    <div className="seg" role="tablist">
      {options.map((o) => (
        <button key={o.value} className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)}
          role="tab" aria-selected={o.value === value}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** "Chart | Table" switch: every chart has a table twin (accessibility, exact values). */
export function useViewToggle(): [boolean, ReactNode] {
  const [table, setTable] = useState(false);
  const toggle = (
    <Segmented options={[{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }]}
      value={table ? "table" : "chart"} onChange={(v) => setTable(v === "table")} />
  );
  return [table, toggle];
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}><i className="swatch" style={{ background: i.color }} />{i.label}</span>
      ))}
    </div>
  );
}

/** Theme toggle; stored per viewer (convenience only, page works without storage). */
export function ThemeToggle() {
  const [theme, setTheme] = useState<"system" | "light" | "dark">("system");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("nt-theme");
      if (saved === "light" || saved === "dark") setTheme(saved);
    } catch { /* storage unavailable */ }
  }, []);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("nt-theme", theme); } catch { /* ignore */ }
  }, [theme]);
  return (
    <Segmented options={[{ value: "system", label: "Auto" }, { value: "light", label: "Light" },
      { value: "dark", label: "Dark" }]} value={theme} onChange={setTheme} />
  );
}
