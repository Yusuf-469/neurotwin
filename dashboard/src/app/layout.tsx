import type { Metadata } from "next";
import type { ReactNode } from "react";
import NavBar from "@/components/NavBar";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroTwin · Hospital digital twin",
  description:
    "Research simulation comparing reinforcement learning, optimisation and rules for hospital bed allocation. Not a clinical tool.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="banner">
          Research simulation: every number comes from a simplified model, not a real hospital.
          This is not a clinical or decision-support tool.
        </div>
        <NavBar />
        <main>{children}</main>
      </body>
    </html>
  );
}
