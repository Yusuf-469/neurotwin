"use client";

import { useState } from "react";
import { api, type DecisionRecord, type Explanation, type Insight } from "@/lib/api";
import { PRIORITY_COLORS, PRIORITY_NAMES, RL_POLICY_KEYS, UNIT_LABELS } from "@/lib/policies";
import AgentInsight from "./AgentInsight";
import { Card } from "./ui";

interface Props {
  sessionId: string;
  decisions: DecisionRecord[];
  total: number;
  /** Who is deciding; RL agents also get the "Inside the agent" panel. */
  policy: string;
}

export default function DecisionLog({ sessionId, decisions, total, policy }: Props) {
  const [selected, setSelected] = useState<number | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [insight, setInsight] = useState<Insight | null>(null);
  const [error, setError] = useState("");
  const isRL = RL_POLICY_KEYS.includes(policy);

  const explain = async (id: number) => {
    setSelected(id);
    setError("");
    setInsight(null);
    try {
      const [ex, ins] = await Promise.all([
        api.explanation(sessionId, id),
        isRL ? api.insight(sessionId, id).catch(() => null) : Promise.resolve(null),
      ]);
      setExplanation(ex.explanation);
      setInsight(ins);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <Card title="Every decision, explained"
      subtitle={`${total} decisions so far`}
      explain={isRL
        ? "Click any decision to look inside the agent: how its networks voted, what the best rule would have done, and which inputs tipped it."
        : "Click any decision on the left to see why it was made. Explanations are written afterwards from logged facts and never change what happens."}>
      <div className="grid grid-2 decision-grid">
        <div className="feed">
          {[...decisions].reverse().map((d) => (
            <div key={d.decision_id} className={`feed-item${selected === d.decision_id ? " sel" : ""}`}
              onClick={() => explain(d.decision_id)} role="button" tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && explain(d.decision_id)}>
              <span className="tag">#{d.decision_id}</span>
              <span className="muted" style={{ minWidth: 64 }}>day {(d.time / 24).toFixed(1)}</span>
              {d.action === "PLACE" ? (
                <>
                  <i className="dot" style={{ background: PRIORITY_COLORS[d.priority ?? 5] }} />
                  <span>{PRIORITY_NAMES[d.priority ?? 5]} patient → <strong>{UNIT_LABELS[d.unit ?? ""] ?? d.unit}</strong>
                    {d.off_service && <span className="muted"> (wrong ward: own ward full)</span>}</span>
                  <span className="spacer" />
                  <span className="muted">waited {d.waited_hours?.toFixed(1)} h</span>
                </>
              ) : (
                <><span>Kept beds free for now</span><span className="spacer" /><span className="muted">{d.n_waiting} waiting</span></>
              )}
            </div>
          ))}
          {decisions.length === 0 && <span className="muted">Advance the simulation to see decisions.</span>}
        </div>
        <div>
          {error && <p className="error">{error}</p>}
          {insight && <AgentInsight key={insight.decision_id} data={insight} policy={policy} />}
          {explanation ? (
            <div className="explain stack" style={{ gap: 10, marginTop: insight ? 18 : 0 }}>
              {insight && <div className="eyebrow">In plain words</div>}
              <div className="row">
                <span className="tag">decision #{explanation.decision_id}</span>
                <span className="chip">{explanation.generated_by === "template" ? "written from the decision log" : `written by ${explanation.generated_by}, checked against the log`}</span>
              </div>
              <strong>{explanation.summary}</strong>
              <span className="secondary">{explanation.reasoning}</span>
              <div>
                <div className="eyebrow">Facts used (from the decision log)</div>
                <ul className="muted" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                  {explanation.facts_used.map((f) => <li key={f}>{f}</li>)}
                </ul>
              </div>
              <span className="muted">
                Reliability indicator (grounding, not a statistical confidence): {explanation.reliability_indicator}
              </span>
              <span className="muted" style={{ fontSize: 12 }}>{explanation.limitations}</span>
            </div>
          ) : (
            <p className="muted">{isRL
              ? "Choose a decision on the left to look inside the agent. Nothing here influences the simulation."
              : "Choose a decision on the left. Explanations never influence the simulation."}</p>
          )}
        </div>
      </div>
    </Card>
  );
}
