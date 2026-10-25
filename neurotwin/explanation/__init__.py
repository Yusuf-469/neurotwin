"""Post-hoc explanation layer (RESEARCH.md v2 §5.8 of PROJECT_SPEC; ARCHITECTURE.md §11).

Correct flow: Policy -> Decision -> Simulation -> Decision Log -> Explanation. Explanations are
generated only from logged decision records (facts available to the policy at decision time,
plus the decision). Nothing in this package can read or modify a simulation or a policy.
"""

from .explainer import (Explanation, LLMClient, LLMExplainer, TemplateExplainer,
                        build_prompt, verify_numbers)

__all__ = ["Explanation", "TemplateExplainer", "LLMExplainer", "LLMClient", "build_prompt",
           "verify_numbers"]
