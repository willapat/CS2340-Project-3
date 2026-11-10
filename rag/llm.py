"""LLM provider, implementing rag.types.LLMProvider.

Story:  US-06: Grounded responses
Owner:  Max  |  Sprint 1

User story:
    As an emergency management official, I want the assistant's responses to
    be grounded in my jurisdiction's official documents, so that I can trust
    its guidance is not hallucinated.

The LLM only ever sees the question plus the retrieved passages (see rag/pipeline.py); this
module just sends that prompt to Claude and returns the text.

Configuration (environment variables; never hard-code keys):
    ANTHROPIC_API_KEY  Claude API key (required in production).
    LLM_MODEL          Model id (default claude-opus-5-5).
    LLM_EFFORT         low | medium | high (default medium). Lower is faster and cheaper.
    LLM_MAX_TOKENS     Longest answer allowed, in tokens (default 2048).
    LLM_TIMEOUT        Seconds before giving up on one request (default 60).
"""
from __future__ import annotations

import os
from typing import Any

from rag.types import LLMProvider

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 60.0
# If a safety classifier declines a request, the API re-runs it on a fallback model it picks.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The LLM could not produce an answer (network, auth, rate limit, refusal...)."""


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


class ClaudeLLM(LLMProvider):
    """Claude via the official Anthropic SDK."""

    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or _env("LLM_MODEL", DEFAULT_MODEL)
        self.effort = effort or _env("LLM_EFFORT", DEFAULT_EFFORT)
        self.max_tokens = max_tokens or int(_env("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        self._timeout = timeout or float(_env("LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            # Credentials come from ANTHROPIC_API_KEY. The SDK retries 429/5xx twice by itself.
            self._client = anthropic.Anthropic(timeout=self._timeout)
        return self._client

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Send one prompt and return the model's text. Raises LLMError on any failure."""
        import anthropic

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        try:
            response = self.client.beta.messages.create(**request)
        except anthropic.AuthenticationError as e:
            raise LLMError("The LLM API key is missing or invalid. Set ANTHROPIC_API_KEY.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("The LLM is rate-limited right now. Try again in a minute.") from e
        except anthropic.APIConnectionError as e:  # includes timeouts
            raise LLMError("Could not reach the LLM service.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"The LLM service returned an error ({e.status_code}).") from e
        except anthropic.AnthropicError as e:  # e.g. no credentials configured at all
            raise LLMError(f"The LLM is not configured: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMError("The LLM declined to answer this question.")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError("The LLM returned an empty answer.")
        return text


_llm: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """The app's LLM, created once per process."""
    global _llm
    if _llm is None:
        _llm = ClaudeLLM()
    return _llm
