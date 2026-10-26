import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import type { ReactNode } from "react";
import NavBar from "@/components/NavBar";
import { HospitalBackdrop, PointerFX, ScrollProgress } from "@/components/motion";
import "./globals.css";

// next/font downloads the fonts at build time and serves them from this site (no runtime calls to
// Google), so the static export stays self-contained.
const body = Inter({ subsets: ["latin"], variable: "--font-body", display: "swap" });
const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display", display: "swap" });

export const metadata: Metadata = {
  title: "NeuroTwin · Hospital digital twin",
  description:
    "Research simulation comparing reinforcement learning, optimisation and rules for hospital bed allocation. Not a clinical tool.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${body.variable} ${display.variable}`}>
      <body>
        <HospitalBackdrop />
        <PointerFX />
        <ScrollProgress />
        <div className="banner">
          Research simulation: every number comes from a simplified model, not a real hospital.
          This is not a clinical or decision-support tool.
        </div>
        <NavBar />
        <main>{children}</main>
        <footer className="footer">
          <div className="footer-inner">
            <strong style={{ color: "var(--text-primary)" }}>NeuroTwin</strong>
            <span>Final-year research project: a digital twin for hospital bed allocation.</span>
            <span className="spacer" />
            <a href="https://github.com/Yusuf-469/neurotwin" target="_blank" rel="noreferrer">Source code &amp; reports ↗</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
