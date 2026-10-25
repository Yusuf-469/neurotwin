import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroTwin",
  description: "Research simulation of hospital bed allocation (not a clinical tool)",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="banner">
          Research simulation. All numbers come from a simplified model, not from a real
          hospital. This is not a clinical or decision-support tool.
        </div>
        <nav className="nav">
          <span className="brand">NeuroTwin</span>
          <Link href="/">Overview</Link>
          <Link href="/simulation">Simulation</Link>
          <Link href="/experiments">Study results</Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
