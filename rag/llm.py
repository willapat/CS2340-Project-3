"""LLM provider, implementing rag.types.LLMProvider.

Story:  US-06: Grounded responses
Owner:  Max  |  Sprint 1

User story:
    As an emergency management official, I want the assistant's responses to
    be grounded in my jurisdiction's official documents, so that I can trust
    its guidance is not hallucinated.

The LLM only ever sees the question plus the retrieved passages (see rag/pipeline.py); this
module just sends that prompt to Gemini and returns the text. Code elsewhere depends only on
rag.types.LLMProvider, so swapping providers means adding a class here and returning it from get_llm().

Configuration (environment variables; never hard-code keys):
    GEMINI_API_KEY   Gemini API key (free tier: https://aistudio.google.com/apikey). Required for real answers.
    LLM_MODEL        Model id (default gemini-flash-latest, an alias for Google's current Flash model).
    LLM_MAX_TOKENS   Longest answer allowed, in tokens (default 8192; Gemini's thinking counts toward it).
    LLM_TIMEOUT      Seconds before giving up on one request (default 60).
"""
from __future__ import annotations

import os
from typing import Any

from rag.types import LLMProvider

DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 60.0


class LLMError(RuntimeError):
    """The LLM could not produce an answer (network, auth, rate limit, refusal...)."""


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


class GeminiLLM(LLMProvider):
    """Gemini via Google's official google-genai SDK."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or _env("LLM_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens or int(_env("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        self._timeout = timeout or float(_env("LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            api_key = _env("GEMINI_API_KEY", "")
            if not api_key:
                raise LLMError("The LLM API key is missing. Set GEMINI_API_KEY.")
            # HttpOptions.timeout is in milliseconds.
            self._client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(self._timeout * 1000))
            )
        return self._client

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Send one prompt and return the model's text. Raises LLMError on any failure."""
        import httpx2
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.ClientError as e:
            if e.code == 429:
                raise LLMError("The LLM is rate-limited right now. Try again in a minute.") from e
            if e.code in (401, 403) or "api key" in str(e).lower():
                raise LLMError("The LLM API key is missing or invalid. Set GEMINI_API_KEY.") from e
            if e.code == 404:
                raise LLMError(f"The Gemini model '{self.model}' was not found. Check LLM_MODEL.") from e
            raise LLMError(f"The LLM service rejected the request ({e.code}).") from e
        except errors.APIError as e:  # ServerError (5xx) and anything else from the API
            raise LLMError(f"The LLM service returned an error ({e.code}).") from e
        except httpx2.TransportError as e:  # connection failures and timeouts
            raise LLMError("Could not reach the LLM service.") from e

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMError("The LLM declined to answer this question.")
        candidates = response.candidates or []
        blocked = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}
        if candidates and getattr(candidates[0].finish_reason, "name", "") in blocked:
            raise LLMError("The LLM declined to answer this question.")
        text = (response.text or "").strip()
        if not text:
            raise LLMError("The LLM returned an empty answer.")
        return text


_llm: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """The app's LLM, created once per process."""
    global _llm
    if _llm is None:
        _llm = GeminiLLM()
    return _llm
