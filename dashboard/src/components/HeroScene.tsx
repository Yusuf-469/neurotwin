// Animated hero illustration of a hospital: ambulances arriving (one turned away), a helicopter
// landing on the roof, patients moving from the emergency entrance into wards whose windows
// light up as beds fill. Purely decorative and labelled as an illustration; the real,
// data-driven view is the Live hospital page. SVG + SMIL/CSS only, no scripts.

import { Icon } from "./icons";

function Ambulance({ tag }: { tag?: boolean }) {
  return (
    <g>
      <rect x="0" y="-26" width="44" height="22" rx="4" fill="#f5f6f8" stroke="#c9ccd3" />
      <path d="M44 -20h10l8 8v8H44z" fill="#e9ecf1" stroke="#c9ccd3" />
      <rect x="47" y="-18" width="8" height="7" rx="1.5" fill="#9cc3ef" />
      <path d="M19 -21h6v4h4v6h-4v4h-6v-4h-4v-6h4z" fill="#d03b3b" />
      <rect x="8" y="-30" width="10" height="4" rx="1.5" className="hs-siren" />
      <circle cx="12" cy="-3" r="5" fill="#2b2b2b" /><circle cx="50" cy="-3" r="5" fill="#2b2b2b" />
      {tag && (
        <g transform="translate(-6,-52)">
          <rect width="78" height="17" rx="5" fill="var(--status-critical)" />
          <text x="39" y="12" textAnchor="middle" className="hs-tag">TURNED AWAY</text>
        </g>
      )}
    </g>
  );
}

export default function HeroScene() {
  const windows = Array.from({ length: 24 }, (_, i) => ({ c: i % 6, r: Math.floor(i / 6) }));
  return (
    <div className="hero-scene-wrap lift">
      <svg className="hero-scene" viewBox="0 0 600 470" role="img" aria-label="Illustration of an animated hospital with ambulances, a helicopter and patients moving to wards">
        <defs>
          <linearGradient id="hs-bld" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--surface-1)" /><stop offset="1" stopColor="var(--surface-2)" />
          </linearGradient>
          <filter id="hs-glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>

        {/* clouds */}
        <g className="hs-cloud" style={{ animationDuration: "38s" }}>
          <ellipse cx="90" cy="60" rx="38" ry="12" /><ellipse cx="115" cy="52" rx="24" ry="12" />
        </g>
        <g className="hs-cloud" style={{ animationDuration: "52s", animationDelay: "-20s" }}>
          <ellipse cx="300" cy="36" rx="30" ry="9" /><ellipse cx="318" cy="30" rx="18" ry="9" />
        </g>

        {/* building */}
        <rect x="160" y="120" width="390" height="268" rx="10" fill="url(#hs-bld)" stroke="var(--border)" strokeWidth="1.5" />
        <rect x="160" y="120" width="390" height="14" rx="6" fill="var(--grid)" />
        {windows.map(({ c, r }, i) => (
          <rect key={i} x={262 + c * 46} y={150 + r * 50} width="32" height="28" rx="4" className="hs-win"
            style={{ animationDelay: `${((i * 1.37) % 11).toFixed(2)}s` }} />
        ))}
        <text x="380" y="368" textAnchor="middle" className="hs-label">WARDS</text>

        {/* roof: sign + helipad */}
        <g transform="translate(355,96)">
          <circle r="20" className="hs-ring" />
          <circle r="16" fill="#d03b3b" />
          <path d="M-4 -10h8v6h6v8h-6v6h-8v-6h-6v-8h6z" fill="#fff" />
        </g>
        <g transform="translate(470,112)">
          <ellipse rx="34" ry="8" fill="var(--grid)" />
          <text x="0" y="4" textAnchor="middle" className="hs-h">H</text>
        </g>

        {/* emergency entrance */}
        <rect x="160" y="282" width="92" height="16" rx="3" fill="#d03b3b" />
        <text x="206" y="294" textAnchor="middle" className="hs-tag">EMERGENCY</text>
        <rect x="176" y="298" width="60" height="90" rx="3" fill="var(--surface-2)" stroke="var(--border)" />
        <line x1="206" y1="300" x2="206" y2="388" stroke="var(--border)" />
        <rect x="176" y="298" width="60" height="90" rx="3" className="hs-door" />

        {/* corridor from ED into the wards: patients move along it */}
        {[
          { color: "var(--status-critical)", path: "M206 360 L 248 360 L 248 214 L 324 214", begin: "0s" },
          { color: "var(--status-serious)", path: "M206 360 L 248 360 L 248 264 L 416 264", begin: "1.1s" },
          { color: "var(--text-muted)", path: "M206 360 L 248 360 L 248 164 L 462 164", begin: "2.2s" },
          { color: "var(--status-serious)", path: "M206 360 L 248 360 L 248 314 L 370 314", begin: "3.3s" },
        ].map((p, i) => (
          <g key={i}>
            <circle r="9" fill={p.color} opacity="0.3" filter="url(#hs-glow)">
              <animateMotion dur="4.4s" begin={p.begin} repeatCount="indefinite" path={p.path} keyPoints="0;1;1" keyTimes="0;0.7;1" calcMode="linear" />
            </circle>
            <circle r="5.5" fill={p.color} stroke="var(--surface-1)" strokeWidth="1.5">
              <animateMotion dur="4.4s" begin={p.begin} repeatCount="indefinite" path={p.path} keyPoints="0;1;1" keyTimes="0;0.7;1" calcMode="linear" />
            </circle>
          </g>
        ))}

        {/* road */}
        <rect x="0" y="392" width="600" height="56" fill="var(--surface-2)" />
        <line x1="0" y1="420" x2="600" y2="420" className="hs-lane" />
        <rect x="160" y="388" width="92" height="6" fill="var(--grid)" />

        {/* ambulance arriving and stopping at the entrance */}
        <g>
          <animateMotion dur="8s" repeatCount="indefinite" path="M-90 410 L 160 410" keyPoints="0;1;1" keyTimes="0;0.45;1" calcMode="spline" keySplines="0.3 0 0.2 1;0 0 1 1" />
          <animate attributeName="opacity" dur="8s" repeatCount="indefinite" values="1;1;1;0" keyTimes="0;0.45;0.92;1" />
          <Ambulance />
        </g>
        {/* ambulance turned away: drives straight past */}
        <g opacity="0">
          <animateMotion dur="9s" begin="3s" repeatCount="indefinite" path="M-110 440 L 700 440" />
          <animate attributeName="opacity" dur="9s" begin="3s" repeatCount="indefinite" values="1;1" />
          <Ambulance tag />
        </g>

        {/* helicopter */}
        <g>
          <animateMotion dur="14s" repeatCount="indefinite" calcMode="spline"
            values="-90,40; 150,24; 330,48; 440,88; 470,101; 470,101; 500,86; 600,40; 700,6"
            keyTimes="0; 0.14; 0.28; 0.38; 0.44; 0.66; 0.74; 0.88; 1"
            keySplines="0 0 1 1; 0 0 1 1; 0 0 1 1; 0 0 0.4 1; 0 0 1 1; 0.6 0 1 1; 0 0 1 1; 0 0 1 1" />
          <g transform="translate(0,-8)">
            <ellipse cx="0" cy="0" rx="22" ry="9" fill="var(--accent)" />
            <ellipse cx="9" cy="-1" rx="8" ry="5" fill="#bfe0ff" />
            <rect x="-44" y="-3" width="26" height="4" rx="2" fill="var(--accent)" />
            <rect x="-48" y="-9" width="4" height="12" rx="2" fill="var(--accent)" />
            <line x1="-14" y1="11" x2="14" y2="11" stroke="var(--axis)" strokeWidth="2" />
            <rect x="-30" y="-14" width="60" height="3" rx="1.5" fill="var(--text-muted)" className="hs-rotor" />
          </g>
        </g>
      </svg>

      {/* floating readouts */}
      <div className="hs-chip c1"><Icon name="bed" size={14} /> Beds in use <b>94%</b><span className="hs-bar"><i /></span></div>
      <div className="hs-chip c2"><Icon name="cpu" size={14} /> RL agent: <b>Critical patient → ICU</b></div>
      <div className="hs-chip c3 bad"><Icon name="ambulance" size={14} /> ED full: ambulance turned away</div>
      <div className="hs-caption">Illustration · the real simulation runs in <b>Live hospital</b></div>
    </div>
  );
}
