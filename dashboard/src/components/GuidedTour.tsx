"use client";

// Guided tour for presentations: a fixed sequence of steps across the pages. Each step opens
// its page, scrolls to one element, dims everything else and shows a short caption. Mounted in
// the root layout, so it survives navigation between pages. Started by a "nt-tour-start" window
// event (header button, hero link). Keys: right arrow / Enter = next, left arrow = back, Esc = end.

import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "./icons";

interface Step { page: string; query?: string; selector: string; title: string; text: string }

// Captions state only results shown elsewhere on the site.
const STEPS: Step[] = [
  { page: "/", selector: ".hero2 h1", title: "The question",
    text: "A virtual hospital decides who gets the next bed. A reinforcement-learning agent, built from scratch, competes with hand-tuned rules and an optimiser on identical patients." },
  { page: "/", selector: ".arch", title: "Ten parts, built and tested",
    text: "Simulator, digital twin, decision-makers, training, statistics, explanations, API and this website. Click any part for its code and tests." },
  { page: "/simulation/", query: "?demo=1", selector: ".floorplan", title: "A live run",
    text: "This hospital is being simulated on the server right now. Every dot is a patient, and every placement is decided by the RL agent." },
  { page: "/simulation/", query: "?demo=1", selector: ".tabs", title: "Look inside the agent",
    text: "Open any decision to see how the five networks voted, what the best rule would have done, and which inputs tipped the choice. Or play out possible futures from this moment." },
  { page: "/race/", selector: "main .card", title: "A fair race",
    text: "Identical copies of one hospital, one per decision-maker, run side by side. Same patients for everyone, so luck cancels out." },
  { page: "/experiments/", selector: "#overall", title: "The study",
    text: "Six decision-makers, a normal day and four crises, 50 sealed test runs each. Lower cost is better." },
  { page: "/experiments/", selector: "#beat", title: "The headline",
    text: "The latest agent has a significantly lower cost than the best hand-tuned rule on normal days and in staff shortages, and is never significantly worse." },
  { page: "/experiments/", selector: "#trust", title: "Check it yourself",
    text: "Any of the 1,500 test runs can be re-run live on the server; the result matches the published number to every digit." },
];

const PAD = 10;
const norm = (p: string | null) => (p ?? "/").replace(/\/+$/, "") || "/";

export default function GuidedTour() {
  const router = useRouter();
  const pathname = usePathname();
  const [step, setStep] = useState<number | null>(null);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const target = useRef<Element | null>(null);

  const end = useCallback(() => { setStep(null); setRect(null); target.current = null; }, []);
  const go = useCallback((i: number) => {
    if (i < 0) return;
    if (i >= STEPS.length) { end(); return; }
    setRect(null); target.current = null; setStep(i);
  }, [end]);

  useEffect(() => {
    const start = () => go(0);
    window.addEventListener("nt-tour-start", start);
    return () => window.removeEventListener("nt-tour-start", start);
  }, [go]);

  // open the step's page, then wait for its element to exist and bring it into view
  useEffect(() => {
    if (step === null) return;
    const s = STEPS[step];
    if (norm(pathname) !== norm(s.page)) { router.push(s.page + (s.query ?? "")); return; }
    let tries = 0;
    const id = window.setInterval(() => {
      const el = document.querySelector(s.selector);
      if (el || ++tries > 60) {
        window.clearInterval(id);
        if (!el) { go(step + 1); return; } // element missing (e.g. data not loaded): skip it
        target.current = el;
        el.scrollIntoView({ block: el.getBoundingClientRect().height > window.innerHeight * 0.6 ? "start" : "center", behavior: "smooth" });
      }
    }, 150);
    return () => window.clearInterval(id);
  }, [step, pathname, router, go]);

  // keep the spotlight on the element while scrolling and resizing
  useEffect(() => {
    if (step === null) return;
    let raf = 0;
    const track = () => {
      if (target.current) {
        const r = target.current.getBoundingClientRect();
        setRect((old) => (old && Math.abs(old.top - r.top) < 0.5 && Math.abs(old.height - r.height) < 0.5 && Math.abs(old.left - r.left) < 0.5 ? old : r));
      }
      raf = requestAnimationFrame(track);
    };
    raf = requestAnimationFrame(track);
    return () => cancelAnimationFrame(raf);
  }, [step]);

  useEffect(() => {
    if (step === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") end();
      else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); go(step + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); go(step - 1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, go, end]);

  if (step === null) return null;
  const s = STEPS[step];
  const vh = typeof window !== "undefined" ? window.innerHeight : 800;
  // spotlight clipped to the viewport (tall elements are larger than the screen)
  const hole = rect && {
    top: Math.max(8, rect.top - PAD), left: Math.max(8, rect.left - PAD),
    width: Math.min(rect.width + 2 * PAD, (typeof window !== "undefined" ? window.innerWidth : 1200) - 16),
    height: Math.min(rect.bottom + PAD, vh - 8) - Math.max(8, rect.top - PAD),
  };
  const captionBelow = !hole || hole.top + hole.height < vh * 0.62;

  return (
    <div className="tour" role="dialog" aria-modal="false" aria-label={`Guided tour, step ${step + 1} of ${STEPS.length}`}>
      {hole && hole.height > 0
        ? <div className="tour-hole" style={{ top: hole.top, left: hole.left, width: hole.width, height: hole.height }} />
        : <div className="tour-dim" />}
      <div className={`tour-card ${captionBelow ? "below" : "above"}`} key={step}>
        <div className="tour-progress">{STEPS.map((_, i) => <i key={i} className={i <= step ? "on" : ""} />)}</div>
        <span className="eyebrow">Step {step + 1} of {STEPS.length}</span>
        <h3>{s.title}</h3>
        <p>{s.text}</p>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn ghost" onClick={end}>End tour</button>
          <span className="spacer" />
          <button className="btn ghost" onClick={() => go(step - 1)} disabled={step === 0}>Back</button>
          <button className="btn glow" onClick={() => go(step + 1)}>
            {step === STEPS.length - 1 ? "Finish" : <>Next <Icon name="arrow" size={14} /></>}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Starts the tour from anywhere (header button, hero link). */
export function startTour() {
  window.dispatchEvent(new Event("nt-tour-start"));
}
