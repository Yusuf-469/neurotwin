import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import type { ReactNode } from "react";
import NavBar from "@/components/NavBar";
import { HospitalBackdrop, PointerFX, ScrollProgress } from "@/components/motion";
import "./globals.css";

// next/font downloads the fonts at build time and serves them from this site (no runtime calls to
// Google), so the static export stays self-contained. The mono face is used for live readouts.
const body = Inter({ subsets: ["latin"], variable: "--font-body", display: "swap" });
const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "NeuroTwin · Hospital digital twin",
  description:
    "Research simulation comparing reinforcement learning, optimisation and rules for hospital bed allocation. Not a clinical tool.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#04070d" },
    { media: "(prefers-color-scheme: light)", color: "#f3f6fb" },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${body.variable} ${display.variable} ${mono.variable}`}>
      <body>
        <a href="#main" className="skip-link">Skip to content</a>
        <HospitalBackdrop />
        <PointerFX />
        <ScrollProgress />
        <div className="banner" role="note">
          <span className="banner-dot" aria-hidden="true" />
          <span className="banner-long">Research simulation: every number comes from a simplified model, not a real hospital. Not a clinical or decision-support tool.</span>
          <span className="banner-short">Research simulation · not a clinical tool</span>
        </div>
        <NavBar />
        <main id="main">{children}</main>
        <footer className="footer">
          <div className="footer-inner">
            <span className="footer-brand">NeuroTwin</span>
            <span className="muted">A digital twin for hospital bed allocation · simulation only</span>
            <span className="spacer" />
            <a href="https://github.com/Yusuf-469/neurotwin" target="_blank" rel="noreferrer">Source code &amp; reports ↗</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
