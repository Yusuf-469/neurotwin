"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Icon, type IconName } from "./icons";
import { Scramble, SplitWords } from "./motion";

interface CardProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  /** Plain-language "what am I looking at" line shown under the title. */
  explain?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Card({ title, subtitle, actions, explain, children, className = "" }: CardProps) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <div>
          <h2 className="card-title">{title}</h2>
          {subtitle && <p className="card-sub">{subtitle}</p>}
        </div>
        <div className="spacer" />
        {actions}
      </div>
      {explain && <p className="explain-line"><Icon name="eye" size={16} /><span>{explain}</span></p>}
      {children}
    </section>
  );
}

interface StatProps {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  icon?: IconName;
  accent?: string;
}

export function Stat({ label, value, note, icon, accent }: StatProps) {
  return (
    <div className="card stat" style={accent ? { ["--stat-accent" as string]: accent } : undefined}>
      <span className="stat-label">{icon && <span className="stat-icon"><Icon name={icon} size={15} /></span>}{label}</span>
      <span className="stat-value">{value}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

/** Highlighted one-sentence conclusion for a chart or section. */
export function Takeaway({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "good" | "warn" }) {
  return (
    <div className={`takeaway ${tone}`}>
      <span className="takeaway-icon"><Icon name={tone === "good" ? "check" : tone === "warn" ? "alert" : "arrow"} size={14} stroke={2.4} /></span>
      <div>{children}</div>
    </div>
  );
}

/** Shimmering placeholder blocks shown while data loads (keeps the layout from jumping). */
export function Skeleton({ lines = 3, chart = 0 }: { lines?: number; chart?: number }) {
  return (
    <div className="skel-wrap" aria-hidden="true">
      {Array.from({ length: lines }, (_, i) => <div key={i} className="skel skel-line" style={{ width: `${92 - i * 14}%` }} />)}
      {chart > 0 && <div className="skel skel-chart" style={{ height: chart }} />}
    </div>
  );
}

/** Collapsible technical detail: the full statistics stay one click away. */
export function TechDetails({ children, label = "Technical detail" }: { children: ReactNode; label?: string }) {
  return (
    <details className="tech">
      <summary>{label}</summary>
      <div className="tech-body">{children}</div>
    </details>
  );
}

/** Numbered chapter heading for long, story-like pages. */
export function SectionHeader({ id, num, kicker, title, lead }: {
  id?: string; num?: string; kicker?: string; title: string; lead?: ReactNode;
}) {
  return (
    <header className="section-head" id={id}>
      {(num || kicker) && (
        <span className="kicker">{num && <span className="kicker-num">{num}</span>}{kicker && <Scramble text={kicker} />}</span>
      )}
      <h2><SplitWords text={title} /></h2>
      {lead && <p>{lead}</p>}
    </header>
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
