"""Explanations of logged allocation decisions.

Two backends share one output type:

* ``TemplateExplainer`` (default, offline, deterministic): every sentence is computed from the
  logged facts. Its reliability indicator is "high" because it cannot state anything that is not
  in the record. It describes *what the policy's rule does*. For PPO it reports the learned
  action probabilities rather than inventing reasons.
* ``LLMExplainer``: builds a grounded prompt from the facts and calls a user-supplied
  ``LLMClient``. No provider is configured in this repository; connecting one sends decision
  data to an external service and requires the project owner's confirmation (AGENTS.md). The
  generated text is checked by ``verify_numbers``: every number it contains must appear in
  the facts. Failures lower the reliability indicator and are listed in ``limitations``.

"Reliability indicator" is a qualitative label about grounding. It is **not** a statistical
confidence (AGENTS.md rule 15).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

POLICY_RULES = {
    "fcfs": "serves the request that has waited longest; places a patient off-service only "
    "after the off-service waiting threshold",
    "priority": "serves the most urgent request first (priority 1 = most urgent), ties broken "
    "by waiting time; off-service only after the waiting threshold",
    "load_balancing": "serves the most urgent request first and places it in the other ward "
    "when the primary ward is much fuller than the other",
    "trunk_reservation": "serves the most urgent request first but keeps a reserve of free beds "
    "that only priority 1-2 patients may use",
    "mip": "solves an exact one-step optimisation that weighs each waiting request's cost of "
    "waiting against the cost of off-service placement",
    "ppo": "is a learned policy (PPO); its choice reflects patterns learned in training, not an "
    "explicit rule",
}
LIMITATIONS = (
    "Post-hoc explanation of a simulated decision. It describes the policy's rule "
    "and the logged state; it does not show that the decision was good, and it is "
    "not clinical advice."
)


@dataclass
class Explanation:
    """A post-hoc explanation with its facts, alternatives and limitations."""

    decision_id: int
    timestamp: float
    policy: str
    summary: str
    reasoning: str
    reliability_indicator: str  # qualitative grounding label, NOT a confidence
    facts_used: list[str]
    alternatives_considered: list[str]
    limitations: str
    generated_by: str
    verification_issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _facts(record: dict[str, Any]) -> list[str]:
    f = [
        f"time: {record['time']:.1f} h",
        f"waiting requests: {record['n_waiting']} "
        f"({record['n_waiting_high_priority']} with priority 1-2)",
        f"longest wait: {record['oldest_wait_hours']:.1f} h",
        "free staffed beds: " + ", ".join(f"{u} {n}" for u, n in record["free_staffed"].items()),
    ]
    if record["action"] == "PLACE":
        f.append(
            f"chosen request: {record['request_type']}, priority {record['priority']}, "
            f"waited {record['waited_hours']:.1f} h, primary ward {record['primary_unit']}"
        )
        f.append(
            f"placed in: {record['unit']}" + (" (off-service)" if record["off_service"] else "")
        )
    probs = record.get("policy_details", {}).get("action_probabilities")
    if probs:
        f.append(
            "policy action probabilities: "
            + ", ".join(f"{a} {p:.2f}" for a, p in sorted(probs.items(), key=lambda kv: -kv[1]))
        )
    return f


class TemplateExplainer:
    """Deterministic explainer built only from logged facts (offline)."""

    name = "template"

    def explain(self, record: dict[str, Any]) -> Explanation:
        policy = record["policy"]
        rule = POLICY_RULES.get(policy, "applies its own decision rule")
        alternatives = [a for a in record.get("feasible_actions", []) if a != "HOLD"]
        if record["action"] == "PLACE":
            where = record["unit"] + (" (off-service)" if record["off_service"] else "")
            request_type = record["request_type"].replace("_", " ")
            summary = (
                f"Placed a priority-{record['priority']} {request_type} "
                f"request that had waited {record['waited_hours']:.1f} h in {where}."
            )
            reasoning = f"The {policy} policy {rule}."
            if record["off_service"]:
                reasoning += (
                    f" The patient's primary ward ({record['primary_unit']}) had "
                    f"{record['free_staffed'].get(record['primary_unit'], 0)} free staffed beds."
                )
        else:
            summary = f"Held: no placement made with {record['n_waiting']} requests waiting."
            reasoning = (
                f"The {policy} policy {rule}. At this moment it chose not to place any "
                f"further patient although {len(alternatives)} placement action(s) were feasible."
            )
        probs = record.get("policy_details", {}).get("action_probabilities")
        if probs:
            top = max(probs, key=probs.get)
            reasoning += (
                f" The learned policy assigned probability {probs[top]:.2f} to {top}; "
                "this is the policy's output, not a confidence in the decision's quality."
            )
        return Explanation(
            record["decision_id"],
            record["time"],
            policy,
            summary,
            reasoning,
            "high (generated only from logged facts)",
            _facts(record),
            alternatives,
            LIMITATIONS,
            self.name,
        )


class LLMClient(Protocol):
    """Minimal interface an LLM provider adapter must implement."""

    def complete(self, prompt: str) -> str: ...


def build_prompt(record: dict[str, Any]) -> str:
    """Grounded prompt: the model may only use the listed facts and must not judge quality."""
    return (
        "You explain a decision made by a hospital bed-allocation policy in a research "
        "SIMULATION. Use ONLY the facts below. Do not invent numbers, patients or reasons. "
        "Do not say whether the decision was good. Do not give clinical advice. "
        "Write 2-3 sentences.\n\n"
        f"Policy rule: {POLICY_RULES.get(record['policy'], 'unknown')}\n"
        "Facts:\n- " + "\n- ".join(_facts(record)) + "\n\n"
        f"Decision record (JSON): {json.dumps(record, sort_keys=True)}\n"
    )


# A leading minus counts only when not attached to a word ("priority-1" contains 1, not -1).
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def verify_numbers(text: str, record: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that do not occur in the record or its derived facts (rounded forms
    count, e.g. 3.4 for 3.4127). Returns the list of unsupported numbers."""
    allowed: set[str] = set()
    source = json.dumps(record) + " " + " ".join(_facts(record))
    for tok in _NUMBER.findall(source):
        v = float(tok)
        allowed.update({tok, f"{v:g}", f"{v:.1f}", f"{v:.2f}", str(int(round(v)))})
    allowed.update(str(i) for i in range(1, 8))  # priorities, action counts, small ordinals
    return [n for n in _NUMBER.findall(text) if n not in allowed]


class LLMExplainer:
    """LLM-written explanation with a fact-check guard; falls back to the template on failure."""

    name = "llm"

    def __init__(self, client: LLMClient, fallback: TemplateExplainer | None = None) -> None:
        self.client = client
        self.fallback = fallback or TemplateExplainer()

    def explain(self, record: dict[str, Any]) -> Explanation:
        base = self.fallback.explain(record)
        try:
            text = self.client.complete(build_prompt(record)).strip()
        except Exception as exc:  # provider failure: never block the system
            base.verification_issues = [f"LLM call failed ({type(exc).__name__}); template used"]
            return base
        unsupported = verify_numbers(text, record)
        base.reasoning = text
        base.generated_by = self.name
        if unsupported:
            base.reliability_indicator = "low (text contains numbers not found in the logged facts)"
            base.verification_issues = [f"unsupported number: {n}" for n in unsupported]
        else:
            base.reliability_indicator = "medium (LLM text; all numbers match logged facts)"
        base.limitations = LIMITATIONS + " Generated text may misdescribe the policy's reasoning."
        return base
