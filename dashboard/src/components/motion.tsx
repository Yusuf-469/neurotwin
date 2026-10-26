"use client";

// Scroll-driven motion without an animation library: IntersectionObserver + CSS transitions.
// Kept dependency-free on purpose (AGENTS.md §5). Every effect is cosmetic: content is rendered
// in the DOM from the start, and prefers-reduced-motion disables all of it (globals.css).

import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

/** True once the element has scrolled into view (stays true). */
export function useInView<T extends Element>(margin = "0px 0px -10% 0px"): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return;
    if (typeof IntersectionObserver === "undefined") { setInView(true); return; }
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setInView(true); io.disconnect(); }
    }, { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [inView, margin]);
  return [ref, inView];
}

type RevealKind = "up" | "left" | "right" | "zoom";

/** Fades/slides its children in when they enter the viewport. */
export function Reveal({ children, delay = 0, kind = "up", className = "", style }: {
  children: ReactNode; delay?: number; kind?: RevealKind; className?: string; style?: CSSProperties;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <div ref={ref} className={`reveal reveal-${kind}${inView ? " in" : ""} ${className}`}
      style={{ transitionDelay: `${delay}ms`, ...style }}>
      {children}
    </div>
  );
}

/** Renders children only after they scroll into view, so chart animations play on arrival. */
export function WhenVisible({ children, height }: { children: ReactNode; height: number }) {
  const [ref, inView] = useInView<HTMLDivElement>("0px 0px -5% 0px");
  return <div ref={ref} style={{ minHeight: height }}>{inView ? children : null}</div>;
}

/** Number that counts up from 0 when it becomes visible. */
export function CountUp({ value, digits = 0, prefix = "", suffix = "", duration = 1400 }: {
  value: number; digits?: number; prefix?: string; suffix?: string; duration?: number;
}) {
  const [ref, inView] = useInView<HTMLSpanElement>();
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!inView || !Number.isFinite(value)) return;
    const reduce = typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setShown(value); return; }
    let frame = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / duration);
      setShown(value * (1 - Math.pow(1 - k, 3))); // ease-out cubic
      if (k < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [inView, value, duration]);
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
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);
  return <div className="scroll-progress" style={{ transform: `scaleX(${p})` }} aria-hidden="true" />;
}
