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

Model fallback (to stretch the free tier, where every model has its own rate limits):
    Models are tried in order. When one is rate-limited, overloaded, unavailable, times out, or
    returns nothing, the next one is tried, and the failing model is skipped for a cooldown (the
    retry delay Google sends, or a default). Bad API keys and safety blocks are NOT retried on
    other models. GeminiLLM.last_model records which model produced the last answer.

Configuration (environment variables; never hard-code keys):
    GEMINI_API_KEY   Gemini API key (free tier: https://aistudio.google.com/apikey). Required for real answers.
    LLM_MODELS       Comma-separated fallback chain, best first (default: DEFAULT_MODELS below).
    LLM_MAX_TOKENS   Longest answer allowed, in tokens (default 8192; Gemini's thinking counts toward it).
    LLM_TIMEOUT      Seconds before giving up on one model's request (default 30).
    LLM_TOTAL_TIMEOUT  Seconds before giving up on the whole chain (default 90).
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

from rag.types import LLMProvider

# Free-tier limits as of 2026-11 (requests per minute / per day; "?" = not shown in AI Studio).
# Best quality first, then the high-quota models so the app keeps answering when daily limits run out.
# Not included: gemini-2.5-flash and gemini-2.5-flash-lite ("no longer available to new users").
DEFAULT_MODELS = (
    "gemini-3.8-flash",  # 5 / 20
    "gemini-3.7-flash",  # ? / ?
    "gemini-3.6-flash",  # 5 / 20
    "gemini-3.5-flash",  # ? / ?
    "gemini-3.5-flash-lite",  # 15 / 500
    "gemini-3.1-flash-lite",  # 15 / 500
    "gemma-4-31b-it",  # 30 / 14,400 (slow: ~25 s per answer)
    "gemma-4-26b-a4b-it",  # 30 / 14,400
)
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 30.0
DEFAULT_TOTAL_TIMEOUT = 90.0

# How long to skip a model after a failure, when Google doesn't say.
COOLDOWN_RATE_LIMIT = 60.0
COOLDOWN_DAILY_QUOTA = 3600.0
COOLDOWN_UNAVAILABLE = 3600.0  # model not found / not enabled for this key
COOLDOWN_SERVER_ERROR = 30.0

_BLOCKED_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}
_RETRY_DELAY = re.compile(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s")


class LLMError(RuntimeError):
    """The LLM could not produce an answer (network, auth, rate limit, refusal...)."""


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


class _Retryable(Exception):
    """This model failed in a way another model might not; try the next one."""

    def __init__(self, reason: str, cooldown: float = 0.0) -> None:
        super().__init__(reason)
        self.reason, self.cooldown = reason, cooldown


def _models_from_env() -> list[str]:
    raw = _env("LLM_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()] or list(DEFAULT_MODELS)


def _rate_limit_cooldown(error: Exception) -> float:
    """Use Google's retryDelay if it sent one; a daily-quota error means wait much longer."""
    detail = str(getattr(error, "details", "")) + str(error)
    if "PerDay" in detail:
        return COOLDOWN_DAILY_QUOTA
    match = _RETRY_DELAY.search(detail)
    return float(match.group(1)) if match else COOLDOWN_RATE_LIMIT


class GeminiLLM(LLMProvider):
    """Gemini via Google's official google-genai SDK, with a fallback chain of models."""

    def __init__(
        self,
        models: list[str] | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        total_timeout: float | None = None,
        client: Any = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.models = list(models) if models else _models_from_env()
        self.max_tokens = max_tokens or int(_env("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        self._timeout = timeout or float(_env("LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self._total_timeout = total_timeout or float(_env("LLM_TOTAL_TIMEOUT", str(DEFAULT_TOTAL_TIMEOUT)))
        self._client = client
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self.last_model: str | None = None

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

    def available_models(self) -> list[str]:
        """Models not currently cooling down, in chain order."""
        now = self._clock()
        with self._lock:
            return [m for m in self.models if self._cooldown_until.get(m, 0.0) <= now]

    def _cool_down(self, model: str, seconds: float) -> None:
        if seconds > 0:
            with self._lock:
                self._cooldown_until[model] = self._clock() + seconds

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Return the first model's answer that succeeds. Raises LLMError if none do."""
        client = self.client  # raises LLMError right away if the key is missing
        deadline = self._clock() + self._total_timeout
        candidates = self.available_models()
        if not candidates:
            raise LLMError("Every LLM model is rate-limited right now. Try again in a minute.")

        failures: list[str] = []
        for model in candidates:
            if self._clock() >= deadline:
                failures.append("out of time")
                break
            try:
                text = self._generate_once(client, model, prompt, system, kwargs)
            except _Retryable as e:
                self._cool_down(model, e.cooldown)
                failures.append(f"{model}: {e.reason}")
                continue
            self.last_model = model
            return text

        if failures and all("rate-limited" in f for f in failures):
            raise LLMError("Every LLM model is rate-limited right now. Try again in a minute.")
        raise LLMError("The LLM service could not answer right now (" + "; ".join(failures) + ").")

    def _generate_once(self, client: Any, model: str, prompt: str, system: str | None, kwargs: dict) -> str:
        import httpx2
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
        except errors.ClientError as e:
            if e.code == 429:
                raise _Retryable("rate-limited", _rate_limit_cooldown(e)) from e
            if e.code in (401, 403) or "api key" in str(e).lower():
                raise LLMError("The LLM API key is missing or invalid. Set GEMINI_API_KEY.") from e
            if e.code == 404:
                raise _Retryable("model not available", COOLDOWN_UNAVAILABLE) from e
            raise LLMError(f"The LLM service rejected the request ({e.code}).") from e
        except errors.APIError as e:  # ServerError (5xx: overloaded, internal) and anything else
            raise _Retryable(f"server error {e.code}", COOLDOWN_SERVER_ERROR) from e
        except httpx2.TransportError as e:  # connection failures and timeouts
            raise _Retryable("unreachable or timed out") from e

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMError("The LLM declined to answer this question.")
        candidates = response.candidates or []
        if candidates and getattr(candidates[0].finish_reason, "name", "") in _BLOCKED_FINISH:
            raise LLMError("The LLM declined to answer this question.")
        text = (response.text or "").strip()
        if not text:
            raise _Retryable("empty answer")
        return text


_llm: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """The app's LLM, created once per process."""
    global _llm
    if _llm is None:
        _llm = GeminiLLM()
    return _llm
