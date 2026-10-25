"""Claude adapter for the LLM explainer (optional; used only when ANTHROPIC_API_KEY is set).

* The key is read from the environment by the SDK; it is never stored, logged or returned.
* Model: ``NEUROTWIN_CLAUDE_MODEL`` (default ``claude-opus-5-5``) at low effort, since
  explanations are a few sentences.
* Server-side refusal fallbacks are enabled (``fallbacks="default"``). A response that still
  ends in a refusal, or any API error, raises; ``LLMExplainer`` then falls back to the
  deterministic template explainer, so the site never depends on the provider.
* The client sends only the grounded prompt built from one decision-log record (simulated,
  non-personal data).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "claude-opus-5-5"


class LLMUnavailable(RuntimeError):
    """The provider did not return a usable explanation."""


class AnthropicLLMClient:
    """``LLMClient`` implementation using the official Anthropic SDK."""

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # optional dependency: imported only when an LLM is configured

        self.model = model or os.environ.get("NEUROTWIN_CLAUDE_MODEL", DEFAULT_MODEL)
        self._client = anthropic.Anthropic(timeout=60.0, max_retries=2)

    def complete(self, prompt: str) -> str:
        """Return the model's text for ``prompt``; raise ``LLMUnavailable`` otherwise."""
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMUnavailable("the model declined to explain this decision")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMUnavailable("empty response")
        return text


def llm_configured() -> bool:
    """True when an Anthropic credential is present in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
