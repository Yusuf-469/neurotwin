"use client";

// Scroll-driven motion without an animation library: IntersectionObserver + CSS transitions.
// Kept dependency-free on purpose (AGENTS.md §5). Every effect is cosmetic: content is always in
// the DOM. Motion is intentionally *not* disabled by the OS "reduce motion" setting, at the
// project owner's request; it is limited to opacity/transform so it stays cheap.

import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

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
  return <div ref={ref} style={{ minHeight: height }}>{seen ? children : null}</div>;
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
  const [p, setP] = useState(0);
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      setP(max > 0 ? h.scrollTop / max : 0);
      h.style.setProperty("--scroll", String(h.scrollTop));
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);
  return <div className="scroll-progress" style={{ transform: `scaleX(${p})` }} aria-hidden="true" />;
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
      document.documentElement.style.setProperty("--mx-page", `${e.clientX}px`);
      document.documentElement.style.setProperty("--my-page", `${e.clientY}px`);
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
    window.addEventListener("pointermove", onMove, { passive: true });
    document.addEventListener("pointerleave", onLeave);
    return () => { window.removeEventListener("pointermove", onMove); document.removeEventListener("pointerleave", onLeave); };
  }, []);
  return <div className="cursor-glow" aria-hidden="true" />;
}

/** Fixed, decorative hospital-themed background: blueprint grid, drifting medical crosses and
 *  ECG traces sweeping across the page. Purely visual; sits behind all content. */
export function HospitalBackdrop() {
  const crosses = [
    { x: 6, y: 18, s: 22, d: 0 }, { x: 88, y: 12, s: 16, d: -6 }, { x: 72, y: 64, s: 26, d: -12 },
    { x: 14, y: 78, s: 18, d: -3 }, { x: 46, y: 40, s: 12, d: -9 }, { x: 93, y: 88, s: 20, d: -15 },
    { x: 30, y: 6, s: 14, d: -18 }, { x: 58, y: 92, s: 15, d: -7 },
  ];
  const ecg = "M0 50 L180 50 L200 50 L210 30 L222 72 L236 10 L250 88 L262 50 L300 50 L420 50 L432 42 L446 58 L460 50 L600 50";
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="aurora a1" /><div className="aurora a2" /><div className="aurora a3" />
      <div className="floor" />
      <div className="bp-grid" />
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
