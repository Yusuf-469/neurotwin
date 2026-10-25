"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, API_URL } from "@/lib/api";

export default function Overview() {
  const [health, setHealth] = useState<string>("checking…");

  useEffect(() => {
    api.health().then(() => setHealth("connected")).catch(() => setHealth("not reachable"));
  }, []);

  return (
    <>
      <div className="panel">
        <h2>NeuroTwin: hospital bed allocation in a validated digital twin</h2>
        <p>
          NeuroTwin simulates patient flow through an emergency department, medical and surgical
          wards and an ICU, with nurse-limited capacity, off-service placement, ICU step-down and
          deterioration. It compares a reinforcement-learning policy (PPO, implemented in-house)
          with rule-based policies and an exactly solved optimisation model under normal and
          stressed conditions.
        </p>
        <p className="muted">
          API ({API_URL}): <strong>{health}</strong>. Start it with{" "}
          <code>uvicorn neurotwin.api.app:app --port 8000</code>.
        </p>
      </div>
      <div className="grid">
        <div className="panel">
          <h2>Simulation</h2>
          <p className="muted">
            Run a policy step by step, watch occupancy and queues, inspect decisions with
            post-hoc explanations, and run what-if comparisons from the current state.
          </p>
          <Link href="/simulation">Open →</Link>
        </div>
        <div className="panel">
          <h2>Study results</h2>
          <p className="muted">
            Pre-computed results of the study: 50 test seeds per scenario, paired statistics,
            robustness, variance reduction and policy structure.
          </p>
          <Link href="/experiments">Open →</Link>
        </div>
      </div>
    </>
  );
}
