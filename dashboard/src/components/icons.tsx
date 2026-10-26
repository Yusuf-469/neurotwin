// Hand-drawn line icon set (24x24, stroke = currentColor) used instead of emoji, so icons follow
// the theme colours and render identically on every platform.

import type { CSSProperties, ReactNode } from "react";

export type IconName =
  | "ambulance" | "hospital" | "bed" | "users" | "brain" | "clock" | "alert" | "check" | "x"
  | "chart" | "shield" | "flask" | "layers" | "eye" | "arrow" | "sparkle" | "pulse" | "sun"
  | "moon" | "play" | "pause" | "forward" | "lock" | "repeat" | "scale" | "target" | "file"
  | "cpu" | "trend" | "staff" | "door" | "fire" | "split" | "chat" | "wand" | "mirror" | "build"
  | "info" | "hourglass" | "bolt" | "network" | "reset";

const P: Record<IconName, ReactNode> = {
  ambulance: <><path d="M3 7h10v9H3z" /><path d="M13 10h4l3 3v3h-7" /><circle cx="7" cy="17.5" r="1.8" /><circle cx="16.5" cy="17.5" r="1.8" /><path d="M8 9.5v3M6.5 11h3" /></>,
  hospital: <><path d="M4 21V7l8-4 8 4v14" /><path d="M9 21v-5h6v5" /><path d="M12 7.5v4M10 9.5h4" /></>,
  bed: <><path d="M3 18V7" /><path d="M3 14h18v4" /><path d="M21 14v-2a3 3 0 0 0-3-3h-7v5" /><circle cx="7" cy="11" r="1.8" /></>,
  users: <><circle cx="9" cy="8" r="3" /><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" /><circle cx="17" cy="9" r="2.4" /><path d="M16 14.2c2.8.3 5 2.6 5 5.8" /></>,
  brain: <><path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h0V4z" /><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h0V4z" /><path d="M9 9h2M13 9h2M9 14h2M13 14h2" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  alert: <><path d="M12 3l9.5 17h-19z" /><path d="M12 10v4M12 17.2v.1" /></>,
  check: <path d="M4.5 12.5l5 5L19.5 7" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  chart: <><path d="M4 20V4M4 20h16" /><path d="M8 16v-4M12 16V8M16 16v-6" /></>,
  shield: <><path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z" /><path d="M8.5 12l2.5 2.5 4.5-4.5" /></>,
  flask: <><path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3" /><path d="M7 15h10" /></>,
  layers: <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /></>,
  eye: <><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  sparkle: <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
  pulse: <path d="M3 12h4l2-5 4 10 2-5h6" />,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  play: <path d="M7 5l12 7-12 7z" />,
  pause: <path d="M8 5v14M16 5v14" />,
  forward: <path d="M4 6l8 6-8 6zM12 6l8 6-8 6z" />,
  lock: <><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></>,
  repeat: <><path d="M17 2l3 3-3 3" /><path d="M4 11V9a4 4 0 0 1 4-4h12" /><path d="M7 22l-3-3 3-3" /><path d="M20 13v2a4 4 0 0 1-4 4H4" /></>,
  scale: <><path d="M12 3v18M7 21h10M5 7h14" /><path d="M5 7l-3 6a3 3 0 0 0 6 0zM19 7l-3 6a3 3 0 0 0 6 0z" /></>,
  target: <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></>,
  file: <><path d="M14 3H6v18h12V7z" /><path d="M14 3v4h4M9 12h6M9 16h6" /></>,
  cpu: <><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></>,
  trend: <><path d="M3 17l6-6 4 4 8-8" /><path d="M15 7h6v6" /></>,
  staff: <><circle cx="12" cy="7" r="3.5" /><path d="M5 21c0-4 3-7 7-7s7 3 7 7" /><path d="M12 14v4M10 16h4" /></>,
  door: <><path d="M5 21V4h11v17" /><path d="M3 21h18M13 12h.1" /></>,
  fire: <path d="M12 3c1 4 5 5.5 5 10a5 5 0 0 1-10 0c0-2.5 1.5-3.5 2-5 .8 1.5 1.5 2 2.5 2C11 7.5 11 5 12 3z" />,
  split: <><path d="M4 12h6l4-6h6M14 18h6M10 12l4 6" /><path d="M17 3l3 3-3 3M17 15l3 3-3 3" /></>,
  chat: <><path d="M4 5h16v11H9l-5 4z" /><path d="M8 9h8M8 12h5" /></>,
  wand: <><path d="M4 20L16 8" /><path d="M14 4l1 2 2 1-2 1-1 2-1-2-2-1 2-1zM19 11l.6 1.4L21 13l-1.4.6L19 15l-.6-1.4L17 13l1.4-.6z" /></>,
  mirror: <><rect x="3" y="4" width="8" height="16" rx="2" /><rect x="13" y="4" width="8" height="16" rx="2" /><path d="M7 9v6M17 9v6" /></>,
  build: <><path d="M3 21h18M5 21V10l7-5 7 5v11" /><path d="M9 21v-6h6v6M9 11h6" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.1" /></>,
  hourglass: <><path d="M6 3h12M6 21h12" /><path d="M7 3c0 5 10 6 10 9s-10 4-10 9M17 3c0 5-10 6-10 9s10 4 10 9" /></>,
  bolt: <path d="M13 2L4 14h7l-1 8 9-12h-7z" />,
  network: <><circle cx="5" cy="6" r="2" /><circle cx="5" cy="18" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="6" r="2" /><circle cx="19" cy="18" r="2" /><path d="M7 6.8l3.2 4M7 17.2l3.2-4M13.8 11.2L17 6.8M13.8 12.8L17 17.2" /></>,
  reset: <><path d="M4 4v6h6" /><path d="M4.5 15a8 8 0 1 0 1.9-8.4L4 10" /></>,
};

export function Icon({ name, size = 18, stroke = 1.8, className = "", style, title }: {
  name: IconName; size?: number; stroke?: number; className?: string; style?: CSSProperties; title?: string;
}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
      strokeLinecap="round" strokeLinejoin="round" className={`icon ${className}`} style={style}
      aria-hidden={title ? undefined : true} role={title ? "img" : undefined}>
      {title && <title>{title}</title>}
      {P[name]}
    </svg>
  );
}

/** NeuroTwin mark: two mirrored rounded halves (the twin) joined by a heartbeat that ends in a
 *  neural node. The gradient uses the theme accent tokens. */
export function Logo({ size = 30, animated = true }: { size?: number; animated?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true" className={`logo${animated ? " logo-anim" : ""}`}>
      <defs>
        <linearGradient id="ntg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="var(--accent)" />
          <stop offset="60%" stopColor="var(--violet)" />
          <stop offset="100%" stopColor="var(--series-3)" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="36" height="36" rx="11" fill="url(#ntg)" />
      <path d="M12 9.5a6 6 0 0 0-5 6v9a6 6 0 0 0 5 6" fill="none" stroke="#fff" strokeOpacity=".55" strokeWidth="2" strokeLinecap="round" />
      <path d="M28 9.5a6 6 0 0 1 5 6v9a6 6 0 0 1-5 6" fill="none" stroke="#fff" strokeOpacity=".55" strokeWidth="2" strokeLinecap="round" />
      <path d="M8 21h6l2.5-6 4 12 2.5-6h4" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <path className="logo-beat" d="M8 21h6l2.5-6 4 12 2.5-6h4" fill="none" stroke="#bfe3ff" strokeWidth="3.2"
        strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="29.5" cy="21" r="2.6" fill="#fff" className="logo-node" />
    </svg>
  );
}
