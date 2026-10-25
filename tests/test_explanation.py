"""Explanation layer: grounded in logged facts, cannot affect decisions, guards LLM output."""

import json

import numpy as np

from neurotwin.explanation import LLMExplainer, TemplateExplainer, build_prompt, verify_numbers
from neurotwin.policies import DecisionLog, LoggingPolicy, PriorityAllocator
from neurotwin.policies.ppo_policy import PPOPolicy
from neurotwin.rl.actions import N_ACTIONS
from neurotwin.rl.features import N_FEATURES
from neurotwin.rl.ppo import PPOAgent
from neurotwin.rl.ppo.networks import ActorCritic
from neurotwin.rl.ppo.normalize import RunningMeanStd
from neurotwin.simulation import SCENARIOS, HospitalSimulation, load_default_config, run_episode

CFG = load_default_config().with_overrides(horizon_hours=300.0, warmup_hours=100.0)


def _records(policy, scenario="B", seed=3):
    logged = LoggingPolicy(policy, DecisionLog())
    result = run_episode(HospitalSimulation(CFG, SCENARIOS[scenario], seed=seed), logged)
    return logged.log.records, result


def test_logging_with_rich_context_is_still_read_only():
    plain = run_episode(HospitalSimulation(CFG, SCENARIOS["B"], seed=3), PriorityAllocator())
    records, logged = _records(PriorityAllocator())
    assert plain.boarding_times == logged.boarding_times
    assert all("feasible_actions" in r and "waiting_by_priority" in r for r in records)
    assert [r["decision_id"] for r in records] == list(range(len(records)))


def test_template_explanations_use_only_logged_facts():
    records, _ = _records(PriorityAllocator())
    explainer = TemplateExplainer()
    for r in records[:300]:
        e = explainer.explain(r)
        assert e.decision_id == r["decision_id"] and e.policy == "priority"
        assert not verify_numbers(e.summary + " " + e.reasoning, r), (e.summary, e.reasoning)
        assert "not clinical advice" in e.limitations and "confidence" not in e.reliability_indicator
        json.dumps(e.as_dict())
    assert any("off-service" in explainer.explain(r).summary for r in records if r.get("off_service"))


def test_ppo_explanations_report_probabilities_not_confidence():
    agent = PPOAgent(ActorCritic(N_FEATURES, N_ACTIONS), RunningMeanStd((N_FEATURES,)))
    records, _ = _records(PPOPolicy(agent))
    assert all("action_probabilities" in r["policy_details"] for r in records)
    e = TemplateExplainer().explain(records[10])
    assert "probability" in e.reasoning and "not a confidence" in e.reasoning


class FakeLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.text


class BrokenLLM:
    def complete(self, prompt):
        raise TimeoutError


def test_llm_guard_flags_invented_numbers_and_falls_back_on_failure():
    records, _ = _records(PriorityAllocator())
    r = next(x for x in records if x["action"] == "PLACE")
    honest = LLMExplainer(FakeLLM(f"The policy placed a priority {r['priority']} patient.")).explain(r)
    assert honest.generated_by == "llm" and honest.reliability_indicator.startswith("medium")
    invented = LLMExplainer(FakeLLM("This saved 987.65 hours of waiting.")).explain(r)
    assert invented.reliability_indicator.startswith("low")
    assert invented.verification_issues == ["unsupported number: 987.65"]
    failed = LLMExplainer(BrokenLLM()).explain(r)
    assert failed.generated_by == "template" and "failed" in failed.verification_issues[0]
    prompt = build_prompt(r)
    assert "Use ONLY the facts" in prompt and "SIMULATION" in prompt


def test_explanation_package_cannot_reach_the_simulation():
    """Structural guard for AGENTS.md rule 15: the explanation layer imports no engine module."""
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "neurotwin" / "explanation"
    for path in src.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [a.name for a in node.names]
                assert "simulation" not in module and "policies" not in module, path
                assert not any("simulation" in n or "policies" in n for n in names), path
