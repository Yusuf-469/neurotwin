"use client";

import { startTour } from "./GuidedTour";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Icon, Logo, type IconName } from "./icons";

const LINKS: { href: string; label: string; icon: IconName }[] = [
  { href: "/", label: "Overview", icon: "hospital" },
  { href: "/simulation/", label: "Live hospital", icon: "pulse" },
  { href: "/race/", label: "Race", icon: "flag" },
  { href: "/experiments/", label: "Results", icon: "chart" },
];

type Theme = "system" | "light" | "dark";
const THEME_NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const THEME_ICON: Record<Theme, IconName> = { system: "eye", light: "sun", dark: "moon" };
const THEME_LABEL: Record<Theme, string> = { system: "Theme: automatic", light: "Theme: light", dark: "Theme: dark" };

/** Persisted per viewer; the page works without storage (try/catch). */
function usePref<T extends string>(key: string, initial: T, apply: (v: T) => void): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(initial);
  useEffect(() => {
    try {
      const saved = localStorage.getItem(key) as T | null;
      if (saved) setValue(saved);
    } catch { /* storage unavailable */ }
  }, [key]);
  useEffect(() => {
    apply(value);
    try { localStorage.setItem(key, value); } catch { /* ignore */ }
  }, [key, value, apply]);
  return [value, setValue];
}

const applyTheme = (t: Theme) => {
  const root = document.documentElement;
  if (t === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", t);
};
// Motion stays on by default (the project owner's choice); this switch lets any visitor reduce it.
const applyMotion = (m: "full" | "reduced") => document.documentElement.classList.toggle("reduce-motion", m === "reduced");

export default function NavBar() {
  const path = usePathname() ?? "/";
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = usePref<Theme>("nt-theme", "system", applyTheme);
  const [motion, setMotion] = usePref<"full" | "reduced">("nt-motion", "full", applyMotion);
  const [scrolled, setScrolled] = useState(false);
  const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, "")));

  useEffect(() => setOpen(false), [path]);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const controls = (
    <>
      <button type="button" className="tour-btn" onClick={() => { setOpen(false); startTour(); }} title="A guided walk through the project">
        <Icon name="play" size={13} /> Tour
      </button>
      <button type="button" className="icon-btn" onClick={() => setTheme(THEME_NEXT[theme])} aria-label={THEME_LABEL[theme]} title={THEME_LABEL[theme]}>
        <Icon name={THEME_ICON[theme]} size={17} />
      </button>
      <button type="button" className={`icon-btn${motion === "reduced" ? " on" : ""}`} onClick={() => setMotion(motion === "full" ? "reduced" : "full")}
        aria-pressed={motion === "reduced"} aria-label="Reduce motion" title={motion === "reduced" ? "Motion reduced (click for full motion)" : "Reduce motion"}>
        <Icon name="pause" size={15} />
      </button>
    </>
  );

  return (
    <header className={`topbar${scrolled ? " scrolled" : ""}`}>
      <div className="topbar-inner">
        <Link href="/" className="brand" aria-label="NeuroTwin home">
          <Logo size={30} />
          <span>NeuroTwin</span>
        </Link>
        <nav className="nav" aria-label="Main">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} className={isActive(l.href) ? "active" : ""} aria-current={isActive(l.href) ? "page" : undefined}>
              <Icon name={l.icon} size={15} />{l.label}
            </Link>
          ))}
        </nav>
        <div className="nav-tools">{controls}</div>
        <button type="button" className="icon-btn menu-btn" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-controls="mobile-menu" aria-label="Menu">
          <span className={`burger${open ? " x" : ""}`}><i /><i /><i /></span>
        </button>
      </div>
      <div id="mobile-menu" className={`mobile-menu${open ? " open" : ""}`} aria-hidden={!open}>
        {LINKS.map((l, i) => (
          <Link key={l.href} href={l.href} className={isActive(l.href) ? "active" : ""} style={{ transitionDelay: open ? `${60 + i * 40}ms` : "0ms" }}
            tabIndex={open ? 0 : -1}>
            <Icon name={l.icon} size={18} />{l.label}
          </Link>
        ))}
        <div className="mobile-tools">{controls}<span className="muted">Theme · motion</span></div>
      </div>
    </header>
  );
}
