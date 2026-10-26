"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ui";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/simulation/", label: "Live hospital" },
  { href: "/experiments/", label: "Results" },
];

export default function NavBar() {
  const path = usePathname() ?? "/";
  const isActive = (href: string) =>
    href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, ""));
  return (
    <header className="topbar">
      <Link href="/" className="brand" aria-label="NeuroTwin home">
        <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
          <rect x="2" y="2" width="28" height="28" rx="8" fill="var(--accent)" />
          <path d="M9 17h4l2-5 3 9 2-4h3" fill="none" stroke="#fff" strokeWidth="2.2"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        NeuroTwin
      </Link>
      <nav className="nav">
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} className={isActive(l.href) ? "active" : ""}>{l.label}</Link>
        ))}
      </nav>
      <ThemeToggle />
    </header>
  );
}
