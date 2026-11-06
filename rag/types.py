"""Shared contracts: the few types that cross story boundaries.

Kept deliberately small. Each owner designs their own internals; add fields
or types here when you need them, and tag the team in the PR, since others
code against this file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """A whole ingested document (output of US-02)."""

    doc_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a Document that gets indexed (output of US-03)."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A Chunk returned by a search, with its source info and relevance score."""

    chunk: Chunk
    score: float
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """What /api/chat returns to the front end."""

    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    prompt_type: str = ""
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ingestor(ABC):
    """One ingestion method (US-02)."""

    @abstractmethod
    def ingest(self, source: str) -> list[Document]: ...


class VectorStore(ABC):
    """Swappable vector store (US-04). Chroma is the default backend."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def query(self, text: str, k: int = 5) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Remove every chunk belonging to `doc_id` (e.g. before re-adding an updated document)."""

    @abstractmethod
    def clear(self) -> None: ...


class LLMProvider(ABC):
    """Swappable LLM, configured via environment variables (US-06)."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class PromptSelector(ABC):
    """Picks a prompt at runtime from the question (US-08)."""

    @abstractmethod
    def select(self, question: str) -> Any: ...
