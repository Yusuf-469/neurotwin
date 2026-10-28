"use client";

// Scroll-driven motion without an animation library: IntersectionObserver + CSS transitions.
// Kept dependency-free on purpose (AGENTS.md §5). Every effect is cosmetic: content is always in
// the DOM. Motion is intentionally *not* disabled by the OS "reduce motion" setting, at the
// project owner's request; it is limited to opacity/transform so it stays cheap.

import { Fragment, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

type Place = "below" | "in" | "above";

/** Where an element is relative to a band in the middle of the viewport (updates both ways). */
export function usePlace<T extends Element>(margin = "-8% 0px -8% 0px"): [React.RefObject<T | null>, Place] {
  const ref = useRef<T>(null);
  const [place, setPlace] = useState<Place>("below");
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") { setPlace("in"); return; }
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) setPlace("in");
      else setPlace(e.boundingClientRect.top < 0 ? "above" : "below");
    }, { rootMargin: margin, threshold: 0 });
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);
  return [ref, place];
}

/** True once the element has been in view (one-way; used to mount charts lazily). */
export function useInView<T extends Element>(margin = "0px 0px -5% 0px"): [React.RefObject<T | null>, boolean] {
  const [ref, place] = usePlace<T>(margin);
  const [seen, setSeen] = useState(false);
  useEffect(() => { if (place === "in") setSeen(true); }, [place]);
  return [ref, seen];
}

type RevealKind = "up" | "left" | "right" | "zoom";

/** Animates children in when they enter the viewport and out again when they leave it, in the
 *  direction of travel (scrolling back reverses it). */
export function Reveal({ children, delay = 0, kind = "up", className = "", style }: {
  children: ReactNode; delay?: number; kind?: RevealKind; className?: string; style?: CSSProperties;
}) {
  const [ref, place] = usePlace<HTMLDivElement>("-4% 0px -4% 0px");
  return (
    <div ref={ref} className={`reveal reveal-${kind} is-${place} ${className}`}
      style={{ transitionDelay: place === "in" ? `${delay}ms` : "0ms", ...style }}>
      {children}
    </div>
  );
}

/** A full page section that fades/scales away once scrolled past and returns when scrolled back:
 *  "scroll to reveal the next section". */
export function Scene({ children, id, className = "" }: { children: ReactNode; id?: string; className?: string }) {
  const [ref, place] = usePlace<HTMLElement>("-18% 0px -18% 0px");
  return (
    <section ref={ref} id={id} className={`section scene scene-${place} ${className}`}>
      {children}
    </section>
  );
}

/** Renders children only after they scroll into view, so chart animations play on arrival. */
export function WhenVisible({ children, height }: { children: ReactNode; height: number }) {
  const [ref, seen] = useInView<HTMLDivElement>();
  return (
    <div ref={ref} className={`when-visible${seen ? " seen" : ""}`} style={{ minHeight: height }}>
      {seen ? children : <div className="skel skel-chart" style={{ height }} aria-hidden="true" />}
    </div>
  );
}

/** Number that counts up from 0 each time it scrolls into view. */
export function CountUp({ value, digits = 0, prefix = "", suffix = "", duration = 1400 }: {
  value: number; digits?: number; prefix?: string; suffix?: string; duration?: number;
}) {
  const [ref, place] = usePlace<HTMLSpanElement>("0px");
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (place !== "in" || !Number.isFinite(value)) return;
    let frame = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / duration);
      setShown(value * (1 - Math.pow(1 - k, 3))); // ease-out cubic
      if (k < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [place, value, duration]);
  const text = Number.isFinite(value)
    ? shown.toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : "–";
  return <span ref={ref}>{prefix}{text}{suffix}</span>;
}

/** Thin bar at the top of the page showing how far the reader has scrolled. */
export function ScrollProgress() {
  // Writes the transform directly, once per frame: no React re-render and no page-wide CSS
  // variable per scroll event (that forced a style recalculation of the whole document).
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let frame = 0;
    const paint = () => {
      frame = 0;
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      if (ref.current) ref.current.style.transform = `scaleX(${max > 0 ? h.scrollTop / max : 0})`;
    };
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(paint); };
    paint();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll);
    };
  }, []);
  return <div ref={ref} className="scroll-progress" style={{ transform: "scaleX(0)" }} aria-hidden="true" />;
}

/** Pointer effects for every `.lift` card: 3D tilt toward the cursor plus a spotlight glow that
 *  follows it. One delegated listener for the whole page. */
export function PointerFX() {
  useEffect(() => {
    let last: HTMLElement | null = null;
    const reset = (el: HTMLElement | null) => {
      if (!el) return;
      el.style.removeProperty("--rx"); el.style.removeProperty("--ry");
      el.classList.remove("hovering");
    };
    const onMove = (e: PointerEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.(".lift") as HTMLElement | null;
      if (el !== last) { reset(last); last = el; }
      if (!el) return;
      const r = el.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width;
      const y = (e.clientY - r.top) / r.height;
      el.classList.add("hovering");
      el.style.setProperty("--mx", `${(x * 100).toFixed(1)}%`);
      el.style.setProperty("--my", `${(y * 100).toFixed(1)}%`);
      el.style.setProperty("--rx", `${((0.5 - y) * 7).toFixed(2)}deg`);
      el.style.setProperty("--ry", `${((x - 0.5) * 9).toFixed(2)}deg`);
    };
    const onLeave = () => { reset(last); last = null; };
    // Magnetic large buttons: the button drifts a few px toward the cursor (uses `translate`, so it
    // composes with the existing hover `transform`).
    let mag: HTMLElement | null = null;
    const onMag = (e: PointerEvent) => {
      const b = (e.target as HTMLElement | null)?.closest?.(".btn.lg") as HTMLElement | null;
      if (mag && mag !== b) { mag.style.removeProperty("--bx"); mag.style.removeProperty("--by"); }
      mag = b;
      if (!b) return;
      const r = b.getBoundingClientRect();
      b.style.setProperty("--bx", `${((e.clientX - r.left - r.width / 2) * 0.18).toFixed(1)}px`);
      b.style.setProperty("--by", `${((e.clientY - r.top - r.height / 2) * 0.3).toFixed(1)}px`);
    };
    const onDown = (e: PointerEvent) => {
      const b = (e.target as HTMLElement | null)?.closest?.(".btn, .seg button") as HTMLElement | null;
      if (!b) return;
      const r = b.getBoundingClientRect();
      const d = Math.max(r.width, r.height) * 2;
      const dot = document.createElement("span");
      dot.className = "ripple";
      dot.style.cssText = `width:${d}px;height:${d}px;left:${e.clientX - r.left - d / 2}px;top:${e.clientY - r.top - d / 2}px`;
      b.appendChild(dot);
      window.setTimeout(() => dot.remove(), 650);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointermove", onMag, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    document.addEventListener("pointerleave", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove); window.removeEventListener("pointermove", onMag);
      window.removeEventListener("pointerdown", onDown); document.removeEventListener("pointerleave", onLeave);
    };
  }, []);
  return <CursorGlow />;
}

/** Cursor light: a soft halo and a brighter core that ease toward the pointer at different
 *  speeds, so the light trails smoothly. Moved with translate3d from one requestAnimationFrame
 *  loop (compositor-only: no layout or style recalculation) that stops once it has caught up.
 *  Hidden for touch input, where there is no hovering cursor. */
function CursorGlow() {
  const halo = useRef<HTMLDivElement>(null);
  const core = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const target = { x: -600, y: -600 };
    const h = { x: -600, y: -600 };
    const c = { x: -600, y: -600 };
    let frame = 0;
    let visible = false;
    const show = (on: boolean) => {
      if (on === visible) return;
      visible = on;
      halo.current?.classList.toggle("on", on);
      core.current?.classList.toggle("on", on);
    };
    const tick = () => {
      h.x += (target.x - h.x) * 0.08; h.y += (target.y - h.y) * 0.08;
      c.x += (target.x - c.x) * 0.2; c.y += (target.y - c.y) * 0.2;
      if (halo.current) halo.current.style.transform = `translate3d(${h.x}px, ${h.y}px, 0)`;
      if (core.current) core.current.style.transform = `translate3d(${c.x}px, ${c.y}px, 0)`;
      const moving = Math.abs(target.x - h.x) + Math.abs(target.y - h.y) > 0.5;
      frame = moving ? requestAnimationFrame(tick) : 0;
    };
    const onMove = (e: PointerEvent) => {
      if (e.pointerType === "touch") { show(false); return; }
      if (!visible) { h.x = c.x = e.clientX; h.y = c.y = e.clientY; }
      target.x = e.clientX; target.y = e.clientY;
      show(true);
      if (!frame) frame = requestAnimationFrame(tick);
    };
    const onLeave = () => show(false);
    window.addEventListener("pointermove", onMove, { passive: true });
    document.documentElement.addEventListener("pointerleave", onLeave);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove);
      document.documentElement.removeEventListener("pointerleave", onLeave);
    };
  }, []);
  return (
    <>
      <div ref={halo} className="cursor-glow" aria-hidden="true" />
      <div ref={core} className="cursor-core" aria-hidden="true" />
    </>
  );
}

/** Fixed, decorative hospital-themed background: blueprint grid, drifting medical crosses and
 *  ECG traces sweeping across the page. Purely visual; sits behind all content. */
export function HospitalBackdrop() {
  const grid = useRef<HTMLDivElement>(null);
  const depth = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Scroll parallax for the blueprint grid and a gentle pointer-driven depth shift for the
    // light layers. One eased rAF loop, transforms only; it idles when nothing is changing.
    const want = { x: 0, y: 0 };
    const cur = { x: 0, y: 0 };
    let frame = 0;
    const place = () => {
      if (depth.current) depth.current.style.transform = `translate3d(${cur.x * -34}px, ${cur.y * -24}px, 0)`;
      if (grid.current) grid.current.style.transform =
        `translate3d(${cur.x * -10}px, ${-window.scrollY * 0.08 + cur.y * -8}px, 0)`;
    };
    const tick = () => {
      cur.x += (want.x - cur.x) * 0.05; cur.y += (want.y - cur.y) * 0.05;
      place();
      frame = Math.abs(want.x - cur.x) + Math.abs(want.y - cur.y) > 0.001 ? requestAnimationFrame(tick) : 0;
    };
    const onMove = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      want.x = e.clientX / window.innerWidth - 0.5;
      want.y = e.clientY / window.innerHeight - 0.5;
      if (!frame) frame = requestAnimationFrame(tick);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("scroll", place, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove); window.removeEventListener("scroll", place);
    };
  }, []);
  // Fixed (not random) particle layout so the server and the browser render the same markup.
  const motes = Array.from({ length: 22 }, (_, i) => ({
    x: (i * 37 + 11) % 100, s: 2 + (i % 3), d: 14 + (i * 7) % 12, delay: -((i * 5.3) % 20),
  }));
  const crosses = [
    { x: 6, y: 18, s: 22, d: 0 }, { x: 88, y: 12, s: 16, d: -6 }, { x: 72, y: 64, s: 26, d: -12 },
    { x: 14, y: 78, s: 18, d: -3 }, { x: 46, y: 40, s: 12, d: -9 }, { x: 93, y: 88, s: 20, d: -15 },
    { x: 30, y: 6, s: 14, d: -18 }, { x: 58, y: 92, s: 15, d: -7 },
  ];
  const ecg = "M0 50 L180 50 L200 50 L210 30 L222 72 L236 10 L250 88 L262 50 L300 50 L420 50 L432 42 L446 58 L460 50 L600 50";
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="depth" ref={depth}>
        <div className="aurora a1" /><div className="aurora a2" /><div className="aurora a3" />
      </div>
      <div className="floor"><div className="floor-plane" /></div>
      <div className="bp-grid" ref={grid} />
      <div className="motes">
        {motes.map((m, i) => (
          <span key={i} style={{ left: `${m.x}%`, width: m.s, height: m.s, animationDuration: `${m.d}s`, animationDelay: `${m.delay}s` }} />
        ))}
      </div>
      <div className="scanline" />
      <div className="vignette" />
      {crosses.map((c, i) => (
        <svg key={i} className="bp-cross" viewBox="0 0 24 24" width={c.s} height={c.s}
          style={{ left: `${c.x}%`, top: `${c.y}%`, animationDelay: `${c.d}s` }}>
          <path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z" />
        </svg>
      ))}
      <svg className="bp-ecg e1" viewBox="0 0 600 100" preserveAspectRatio="none"><path d={ecg} /></svg>
      <svg className="bp-ecg e2" viewBox="0 0 600 100" preserveAspectRatio="none"><path d={ecg} /></svg>
    </div>
  );
}

/** Headline that builds word by word (rise + un-blur, staggered) each time it scrolls into view.
 *  The real sentence sits in a screen-reader-only span; the animated words are aria-hidden. */
export function SplitWords({ text, delay = 0, wordClass = "" }: { text: string; delay?: number; wordClass?: string }) {
  const [ref, place] = usePlace<HTMLSpanElement>("-6% 0px -6% 0px");
  return (
    <span ref={ref} className={`split is-${place}`}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {text.split(" ").map((w, i) => (
          // the space sits outside the inline-block word, otherwise it collapses
          <Fragment key={i}>
            {i > 0 && " "}
            <span className="split-w" style={{ transitionDelay: place === "in" ? `${delay + i * 55}ms` : "0ms" }}>
              <span className={wordClass}>{w}</span>
            </span>
          </Fragment>
        ))}
      </span>
    </span>
  );
}

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%+/<>";

/** Monospace label that "decodes" from random glyphs into its text, left to right, on arrival. */
export function Scramble({ text, duration = 650 }: { text: string; duration?: number }) {
  const [ref, place] = usePlace<HTMLSpanElement>("0px");
  const [shown, setShown] = useState(text);
  useEffect(() => {
    if (place !== "in") return;
    let frame = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / duration);
      const fixed = Math.floor(k * text.length);
      setShown(text.split("").map((ch, i) =>
        i < fixed || ch === " " ? ch : GLYPHS[Math.floor(Math.random() * GLYPHS.length)]).join(""));
      if (k < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [place, text, duration]);
  return (
    <span ref={ref}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">{shown}</span>
    </span>
  );
}

/** Cycles through short phrases with a vertical slide (e.g. the five test situations). */
export function Rotator({ items, interval = 2200 }: { items: string[]; interval?: number }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setI((n) => (n + 1) % items.length), interval);
    return () => window.clearInterval(id);
  }, [items.length, interval]);
  return (
    <span className="rotator" aria-live="off">
      <span key={i} className="rotator-item">{items[i]}</span>
    </span>
  );
}

/** Pulsing "now" marker for the newest point of a live line chart (Recharts `dot` render prop).
 *  Every other point renders an empty group so the line itself stays clean. */
export function liveDot(lastIndex: number, color: string) {
  // eslint-disable-next-line react/display-name
  return (p: { cx?: number; cy?: number; index?: number }) => {
    if (p.index !== lastIndex || p.cx == null || p.cy == null) return <g key={`d${p.index}`} />;
    return (
      <g key={`d${p.index}`} className="live-now" style={{ ["--c" as string]: color }}>
        <circle cx={p.cx} cy={p.cy} r={9} className="live-now-ring" />
        <circle cx={p.cx} cy={p.cy} r={4} fill={color} stroke="var(--surface-1)" strokeWidth={2} />
      </g>
    );
  };
}

/** Floating button that appears after scrolling down a screen and returns to the top. */
export function BackToTop() {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const on = () => setShow(window.scrollY > window.innerHeight);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  return (
    <button className={`to-top${show ? " show" : ""}`} aria-label="Back to top" tabIndex={show ? 0 : -1}
      onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2"
        strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
    </button>
  );
}

/** Pauses CSS animations inside page sections, cards and illustrations while they are off
 *  screen, and resumes them on the way back in. Nothing visible changes, but the browser stops
 *  doing per-frame work for things nobody can see, which keeps scrolling and the cursor light
 *  smooth. New content (route changes, live updates) is picked up by a MutationObserver. */
export function AnimationGovernor() {
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const seen = new WeakSet<Element>();
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) e.target.classList.toggle("anim-off", !e.isIntersecting);
    }, { rootMargin: "120px 0px" });
    const scan = () => {
      document.querySelectorAll("main .scene, main .section, main .card, main .hero-scene-wrap").forEach((el) => {
        if (!seen.has(el)) { seen.add(el); io.observe(el); }
      });
    };
    let timer = 0;
    const mo = new MutationObserver(() => {
      if (!timer) timer = window.setTimeout(() => { timer = 0; scan(); }, 400);
    });
    scan();
    mo.observe(document.body, { childList: true, subtree: true });
    return () => { io.disconnect(); mo.disconnect(); window.clearTimeout(timer); };
  }, []);
  return null;
}
