"""Retrieving relevant chunks for a question.

Story:  US-06: Grounded responses
Owner:  Max  |  Sprint 1

User story:
    As an emergency management official, I want the assistant's responses to
    be grounded in my jurisdiction's official documents, so that I can trust
    its guidance is not hallucinated.

Retrieval is the first half of grounding: only passages that are actually relevant to the
question are handed to the LLM. A passage scoring below RAG_MIN_SCORE is dropped, and if
nothing is left the pipeline refuses to answer instead of letting the model guess.

Configuration (environment variables, all optional):
    RAG_TOP_K       How many passages to retrieve per question (default 5).
    RAG_MIN_SCORE   Minimum relevance score, a cosine similarity in [-1, 1] (default 0.25).
"""
from __future__ import annotations

import os
import re

from rag.types import RetrievedChunk, VectorStore

DEFAULT_TOP_K = 5
# all-MiniLM-L6-v2 cosine similarity: unrelated text usually scores below ~0.2, a passage that
# answers the question usually above ~0.35. 0.25 keeps borderline-relevant passages.
DEFAULT_MIN_SCORE = 0.25

_store: VectorStore | None = None


def top_k() -> int:
    return int(os.getenv("RAG_TOP_K", "").strip() or DEFAULT_TOP_K)


def min_score() -> float:
    return float(os.getenv("RAG_MIN_SCORE", "").strip() or DEFAULT_MIN_SCORE)


def get_vector_store() -> VectorStore:
    """The app's knowledge base (US-04's Chroma store), opened once per process."""
    global _store
    if _store is None:
        from rag.vectorstore import ChromaVectorStore  # imported lazily: Chroma is slow to load

        _store = ChromaVectorStore()
    return _store


def _normalised(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def retrieve(
    question: str,
    store: VectorStore | None = None,
    k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return up to k relevant passages for `question`, most relevant first.

    Passages below `threshold` are dropped, as are passages whose text duplicates a
    higher-ranked one (the same paragraph can exist in two versions of a plan).
    """
    question = question.strip()
    if not question:
        return []
    store = store if store is not None else get_vector_store()
    k = top_k() if k is None else k
    threshold = min_score() if threshold is None else threshold

    results = sorted(store.query(question, k=k), key=lambda r: r.score, reverse=True)
    kept: list[RetrievedChunk] = []
    seen: set[str] = set()
    for result in results:
        if result.score < threshold:
            continue
        key = _normalised(result.chunk.text)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(result)
    return kept
