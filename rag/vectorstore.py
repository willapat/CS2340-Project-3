"""Vector store (ChromaDB by default), implementing rag.types.VectorStore.

Story:  US-04: Define / update the dataset
Owner:  Ryon  |  Sprint 1

User story:
    As a system administrator, I want to define and update the set of
    official protocol documents the assistant draws from, so that its
    answers always reflect my jurisdiction's current protocols.

TODO(US-04): Acceptance criteria: see docs/stories.md (Product Owner to fill in).

Keep chromadb imports inside this module so the backend stays swappable.

Configuration:
    CHROMA_PATH         Directory for the persistent Chroma database.
                        Defaults to <project root>/chroma_db (gitignored).
    CHROMA_EMBEDDER     "default" (Chroma's local all-MiniLM-L6-v2, needs onnxruntime) or
                        "gemini" (Gemini embedding API, needs GEMINI_API_KEY). Default "default".
                        Production (PythonAnywhere) has no onnxruntime, so it must use "gemini".
    GEMINI_EMBED_MODEL  Gemini embedding model (default gemini-embedding-001).

Each collection records which embedder built it. Vectors from different embedders
can't be compared, so opening a non-empty store with a different embedder raises
EmbedderMismatchError: clear() and re-add every document after switching.

To add, update or remove whole documents, use rag.dataset rather than this module.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from rag.types import Chunk, RetrievedChunk, VectorStore

# Takes a batch of texts and returns one embedding vector per text.
EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]

DEFAULT_COLLECTION = "alert_assist"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "chroma_db"

DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"
# 768 of gemini-embedding-001's 3072 dimensions: a quarter of the disk, little quality lost.
GEMINI_EMBED_DIMENSIONS = 768
# The Gemini API accepts at most 100 texts per embedding request.
GEMINI_BATCH_SIZE = 100

EMBEDDERS = ("default", "gemini")


class EmbedderMismatchError(RuntimeError):
    """The store was built with a different embedder than the one it's being opened with."""


@dataclass
class StoredDocument:
    """Summary of one document in the store, built from its chunks' metadata."""

    doc_id: str
    source: str
    chunk_count: int
    content_hash: str = ""
    indexed_at: str = ""


def chroma_path() -> Path:
    """Storage directory: $CHROMA_PATH if set and non-empty, else DEFAULT_PATH."""
    value = os.getenv("CHROMA_PATH", "").strip()
    return Path(value) if value else DEFAULT_PATH


def default_embedding_function() -> EmbedFn:
    """Chroma's built-in embedder (all-MiniLM-L6-v2 via ONNX; downloads the model on first use)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


class GeminiEmbedder:
    """Embeds texts with the Gemini API. The client is created on first use, so building
    a store (e.g. to list documents) doesn't need an API key.

    `client` is for tests; normally one is made from GEMINI_API_KEY.
    """

    def __init__(self, model: str | None = None, dimensions: int = GEMINI_EMBED_DIMENSIONS, client: Any = None):
        self.model = model or os.getenv("GEMINI_EMBED_MODEL", "").strip() or DEFAULT_GEMINI_EMBED_MODEL
        self.dimensions = dimensions
        self._client = client

    @property
    def name(self) -> str:
        return f"gemini:{self.model}:{self.dimensions}"

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("CHROMA_EMBEDDER=gemini needs GEMINI_API_KEY to be set.")
            from google import genai

            self._client = genai.Client(api_key=api_key)
        return self._client

    def __call__(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), GEMINI_BATCH_SIZE):
            response = client.models.embed_content(
                model=self.model,
                contents=texts[start : start + GEMINI_BATCH_SIZE],
                config={"output_dimensionality": self.dimensions},
            )
            vectors.extend(list(embedding.values) for embedding in response.embeddings)
        if len(vectors) != len(texts):
            raise RuntimeError(f"Gemini returned {len(vectors)} embeddings for {len(texts)} texts.")
        return vectors


def embedder_from_env() -> tuple[str, EmbedFn]:
    """The (name, embed function) chosen by CHROMA_EMBEDDER. This is the one place to add embedders."""
    kind = os.getenv("CHROMA_EMBEDDER", "").strip().lower() or "default"
    if kind == "gemini":
        embedder = GeminiEmbedder()
        return embedder.name, embedder
    if kind == "default":
        return "default:all-MiniLM-L6-v2", default_embedding_function()
    raise ValueError(f"CHROMA_EMBEDDER must be one of {', '.join(EMBEDDERS)}; got {kind!r}.")


def flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Make metadata acceptable to Chroma, which only stores flat scalar values.

    str/int/float/bool are kept as-is, None values are dropped, and anything
    else (lists, dicts, dates, ...) is JSON-encoded to a string, falling back to
    str(). This is one-way: query results return the string, not the original object.
    """
    flat: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[str(key)] = value
        else:
            flat[str(key)] = json.dumps(value, default=str, sort_keys=True)
    return flat


def _summarise(metadatas: list[dict[str, Any]]) -> list[StoredDocument]:
    docs: dict[str, StoredDocument] = {}
    for meta in metadatas:
        doc_id = str(meta.get("doc_id", ""))
        doc = docs.get(doc_id)
        if doc is None:
            docs[doc_id] = StoredDocument(
                doc_id=doc_id,
                source=str(meta.get("source", "")),
                chunk_count=1,
                content_hash=str(meta.get("content_hash", "")),
                indexed_at=str(meta.get("indexed_at", "")),
            )
        else:
            doc.chunk_count += 1
    return sorted(docs.values(), key=lambda d: d.doc_id)


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB-backed vector store.

    Embeddings are computed here with `embed` and handed to Chroma directly, so
    the collection itself carries no embedding function and the embedder can be
    swapped without touching Chroma config. With no `embed`, CHROMA_EMBEDDER picks
    one. `embedder_name` identifies a custom `embed` (default "custom").
    """

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embed: EmbedFn | None = None,
        embedder_name: str | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else chroma_path()
        self.collection_name = collection_name
        if embed is None:
            self.embedder_name, self._embed = embedder_from_env()
        else:
            self.embedder_name, self._embed = embedder_name or "custom", embed
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._batch_size = self._client.get_max_batch_size()
        self._collection = self._client.get_or_create_collection(
            self.collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        self._check_embedder()

    def _check_embedder(self) -> None:
        stored = (self._collection.metadata or {}).get("embedder")
        if stored == self.embedder_name:
            return
        if stored and self._collection.count() > 0:
            raise EmbedderMismatchError(
                f"The knowledge base at {self.path} was built with embedder {stored!r}, but this store uses "
                f"{self.embedder_name!r}. Clear it and re-add every document after switching embedders."
            )
        self._collection.modify(metadata={"embedder": self.embedder_name})

    def _prepare(self, chunks: list[Chunk]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        metadatas = []
        for chunk in chunks:
            meta = flatten_metadata(chunk.metadata)
            meta["doc_id"] = chunk.doc_id
            meta["source"] = str(chunk.metadata.get("source") or "")
            metadatas.append(meta)
        return [chunk.chunk_id for chunk in chunks], [chunk.text for chunk in chunks], metadatas

    def _upsert(self, ids: list[str], texts: list[str], embeddings, metadatas: list[dict[str, Any]]) -> None:
        for start in range(0, len(ids), self._batch_size):
            end = start + self._batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def _all_metadatas(self, where: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            page = self._collection.get(where=where, include=["metadatas"], limit=self._batch_size, offset=offset)
            if not page["ids"]:
                return
            yield from (dict(meta or {}) for meta in page["metadatas"])
            offset += len(page["ids"])

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks. Re-adding an existing chunk_id overwrites it (upsert).

        Each chunk's Chroma metadata is its flattened Chunk.metadata plus
        `doc_id` (from Chunk.doc_id, used by delete) and `source` (from
        Chunk.metadata["source"], "" if missing, used for RetrievedChunk.source).
        """
        if not chunks:
            return
        ids, texts, metadatas = self._prepare(chunks)
        self._upsert(ids, texts, self._embed(texts), metadatas)

    def replace_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Swap all of `doc_id`'s chunks for `chunks`, leaving no stale chunks behind.

        The new chunks are embedded before anything is deleted, so if embedding fails
        (e.g. the API is down) the old version stays searchable.
        """
        if not chunks:
            raise ValueError("replace_document needs at least one chunk; use delete() to remove a document.")
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError(f"Every chunk passed to replace_document must have doc_id {doc_id!r}.")
        ids, texts, metadatas = self._prepare(chunks)
        embeddings = self._embed(texts)
        self.delete(doc_id)
        self._upsert(ids, texts, embeddings, metadatas)

    def query(self, text: str, k: int = 5) -> list[RetrievedChunk]:
        """Return up to k chunks, most relevant first.

        Score: the collection uses cosine distance, which Chroma reports as
        ``distance = 1 - cosine_similarity`` in [0, 2]. We return
        ``score = 1 - distance`` (i.e. the cosine similarity), so scores fall in
        [-1, 1] and higher means more relevant; 1.0 is an exact direction match.
        """
        count = self._collection.count()
        if k <= 0 or count == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embed([text]),
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for chunk_id, document, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            meta = dict(meta or {})
            chunk_meta = {key: value for key, value in meta.items() if key != "doc_id"}
            chunk = Chunk(chunk_id=chunk_id, doc_id=str(meta.get("doc_id", "")), text=document or "", metadata=chunk_meta)
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=1.0 - float(distance), source=str(meta.get("source", "")), metadata=meta)
            )
        return retrieved

    def documents(self) -> list[StoredDocument]:
        """Every document in the store, sorted by doc_id."""
        return _summarise(list(self._all_metadatas()))

    def document(self, doc_id: str) -> StoredDocument | None:
        """The stored summary of one document, or None if it isn't in the store."""
        found = _summarise(list(self._all_metadatas(where={"doc_id": doc_id})))
        return found[0] if found else None

    def delete(self, doc_id: str) -> None:
        """Remove every chunk whose doc_id matches. A no-op if there are none."""
        self._collection.delete(where={"doc_id": doc_id})

    def clear(self) -> None:
        """Remove all chunks.

        Deletes ids rather than dropping the collection, so other open stores on the
        same path (e.g. the chat process's long-lived one) keep working.
        """
        while True:
            ids = self._collection.get(include=[], limit=self._batch_size)["ids"]
            if not ids:
                return
            self._collection.delete(ids=ids)
