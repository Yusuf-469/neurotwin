"""Provider-neutral LLM adapter for any OpenAI-compatible chat-completions endpoint.

Why: the project budget is zero, and several providers offer free tiers behind this common
protocol (e.g. Google Gemini's OpenAI-compatible endpoint, Groq, a local Ollama server). The
adapter uses only the Python standard library, so it adds no dependency.

Configuration (environment variables; the key is a hosting secret, never in code):
    NEUROTWIN_LLM_BASE_URL   e.g. https://generativelanguage.googleapis.com/v1beta/openai
                             or   https://api.groq.com/openai/v1
    NEUROTWIN_LLM_MODEL      model name offered by that provider
    NEUROTWIN_LLM_API_KEY    the provider's API key

The same guards apply as for every LLM backend: the grounded prompt, the numeric fact-check
(``verify_numbers``), the hourly cap and the template fallback (see explainer.py, api/app.py).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatibleClient:
    """``LLMClient`` for POST {base_url}/chat/completions."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 30.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        """Build from NEUROTWIN_LLM_* variables (raises KeyError if any is missing)."""
        return cls(
            os.environ["NEUROTWIN_LLM_BASE_URL"],
            os.environ["NEUROTWIN_LLM_MODEL"],
            os.environ["NEUROTWIN_LLM_API_KEY"],
        )

    def complete(self, prompt: str) -> str:
        """Return the assistant text; raise RuntimeError on any HTTP or format problem."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {type(exc).__name__}") from None
        try:
            text = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            raise RuntimeError("unexpected LLM response format") from None
        if not text:
            raise RuntimeError("empty LLM response")
        return text


def openai_compatible_configured() -> bool:
    """True when all three NEUROTWIN_LLM_* variables are set."""
    return all(
        os.environ.get(k)
        for k in ("NEUROTWIN_LLM_BASE_URL", "NEUROTWIN_LLM_MODEL", "NEUROTWIN_LLM_API_KEY")
    )
