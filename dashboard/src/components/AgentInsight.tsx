"use client";

// "Inside the agent": for one decision, what the RL agent's networks actually computed.
// Everything shown comes from the backend (neurotwin/rl/introspect.py), computed from the real
// networks and the inputs recorded at decision time; nothing here is generated text.

import type { Insight, InsightPlacement } from "@/lib/api";
import { ACTION_LABELS, PRIORITY_NAMES, UNIT_LABELS, policyMeta } from "@/lib/policies";
import { Icon } from "./icons";
import { TechDetails } from "./ui";

const AGREEMENT: Record<Insight["agreement"], { text: string; tone: "same" | "diff" }> = {
  same: { text: "Same move as the best rule", tone: "same" },
  both_wait: { text: "Both chose to wait", tone: "same" },
  same_patient_other_ward: { text: "Same patient, different ward", tone: "diff" },
  different_patient: { text: "Different patient", tone: "diff" },
  agent_waits: { text: "Agent waits, the rule would place someone", tone: "diff" },
  rule_waits: { text: "Agent places someone, the rule would wait", tone: "diff" },
};

const REQUEST_NAMES: Record<string, string> = {
  admit_general: "ward admission",
  admit_icu: "ICU admission",
  step_down: "step-down from ICU",
  escalation: "escalation to ICU",
};

function Move({ who, p }: { who: string; p: InsightPlacement | null }) {
  return (
    <div className="ai-move">
      <span className="eyebrow">{who}</span>
      {p ? (
        <span>
          <b>{PRIORITY_NAMES[p.priority] ?? `Priority ${p.priority}`}</b> {REQUEST_NAMES[p.request_type] ?? p.request_type}
          {" "}<Icon name="arrow" size={12} /> <b>{UNIT_LABELS[p.unit] ?? p.unit}</b>
          {p.unit !== p.primary_unit && p.request_type === "admit_general" && <span className="muted"> (not their own ward)</span>}
        </span>
      ) : <span><b>Wait</b>, keep the beds free</span>}
    </div>
  );
}

export default function AgentInsight({ data, policy }: { data: Insight; policy: string }) {
  const chosen = data.chosen_action;
  const att = data.attribution;
  const maxAbs = att ? Math.max(1e-9, ...att.inputs.map((r) => Math.abs(r.contribution))) : 1;
  const agree = AGREEMENT[data.agreement];
  const sorted = [...data.actions].sort((a, b) => b.probability - a.probability);

  return (
    <div className="ai stack" style={{ gap: 14 }}>
      <div className="row" style={{ gap: 8 }}>
        <span className="ai-badge"><Icon name="cpu" size={14} /> Inside the agent</span>
        <span className="tag">decision #{data.decision_id}</span>
        <span className="muted">day {(data.time / 24).toFixed(2)}</span>
      </div>

      {/* 1. the vote */}
      <section>
        <div className="ai-h">
          <b>The vote</b>
          <span className="muted">{policyMeta(policy).label}: {data.n_agents > 1
            ? `${data.agents_agreeing} of ${data.n_agents} independently trained copies picked this move`
            : "a single network"}</span>
        </div>
        <div className="ai-votes">
          {sorted.map((a, i) => (
            <div key={a.action} className={`ai-vote${a.action === chosen ? " win" : ""}`}>
              <span className="ai-vote-label">{ACTION_LABELS[a.action] ?? a.action}</span>
              <span className="ai-bar"><i style={{ transform: `scaleX(${a.probability})`, animationDelay: `${i * 70}ms` }} /></span>
              <span className="ai-pct">{(100 * a.probability).toFixed(a.probability > 0.995 || a.probability < 0.005 ? 1 : 0)}%</span>
              {data.n_agents > 1 && (
                <span className="ai-agents" aria-label={`each copy: ${a.per_agent.map((p) => `${(100 * p).toFixed(0)}%`).join(", ")}`}>
                  {a.per_agent.map((p, k) => (
                    <i key={k} style={{ opacity: 0.12 + 0.88 * p }} title={`copy ${k + 1}: ${(100 * p).toFixed(1)}%`} />
                  ))}
                </span>
              )}
            </div>
          ))}
        </div>
        {data.n_agents > 1 && <p className="ai-note">Squares are the five copies; brighter means more confident. The move with the highest average wins.</p>}
      </section>

      {/* 2. against the best rule */}
      <section>
        <div className="ai-h">
          <b>Against the best rule</b>
          <span className={`ai-agree ${agree.tone}`}><Icon name={agree.tone === "same" ? "check" : "alert"} size={13} /> {agree.text}</span>
        </div>
        <div className="ai-moves">
          <Move who="The agent did" p={data.agent_choice} />
          <Move who={`${policyMeta(data.reference_rule).label} would have`} p={data.rule_choice} />
        </div>
        <p className="ai-note">Asked in the same moment, with the same information. The rule is only consulted; it never changes what happens.</p>
      </section>

      {/* 3. why this move */}
      <section>
        <div className="ai-h">
          <b>Why this move?</b>
          {att && <span className="muted">rather than: {ACTION_LABELS[att.runner_up] ?? att.runner_up}</span>}
        </div>
        {att ? (
          <>
            <div className="ai-attr">
              {att.inputs.map((r, i) => {
                const w = Math.abs(r.contribution) / maxAbs;
                const pos = r.contribution >= 0;
                return (
                  <div key={r.key} className="ai-attr-row">
                    <span className="ai-attr-label">{r.label}<span className="ai-attr-val">{r.value}</span></span>
                    <span className="ai-attr-track">
                      <i className={pos ? "pos" : "neg"} style={{ transform: `scaleX(${w})`, animationDelay: `${i * 60}ms` }} />
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="ai-legend">
              <span><i className="neg" /> pushed towards the runner-up</span>
              <span><i className="pos" /> pushed towards this move</span>
            </div>
            <p className="ai-note">
              Compared with an average situation from training, these inputs tipped the network most. This shows what the network
              responded to; it is not proof of cause.
            </p>
            <TechDetails>
              <p>Integrated gradients (48 steps) of the logit margin, chosen minus runner-up, averaged over the {data.n_agents} network{data.n_agents > 1 ? "s" : ""};
                baseline = mean training observation. Margin here {att.margin.toFixed(2)} vs {att.baseline_margin.toFixed(2)} at the baseline; contributions
                sum to the difference (completeness error {att.completeness_error.toFixed(4)}). Remaining inputs together: {att.other_inputs_total.toFixed(3)}.
                Time of day and day of week combine their sine and cosine inputs.</p>
            </TechDetails>
          </>
        ) : (
          <p className="muted">Only one move was possible here, so there was no choice to explain.</p>
        )}
      </section>
    </div>
  );
}
