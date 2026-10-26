"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "./icons";
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
        <Logo size={32} />
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
