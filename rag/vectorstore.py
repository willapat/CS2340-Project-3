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
    CHROMA_PATH  Directory for the persistent Chroma database.
                 Defaults to <project root>/chroma_db (gitignored).

Updating a document: call ``delete(doc_id)`` and then ``add(new_chunks)``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chromadb

from rag.types import Chunk, RetrievedChunk, VectorStore

# Takes a batch of texts and returns one embedding vector per text.
EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]

DEFAULT_COLLECTION = "alert_assist"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "chroma_db"


def chroma_path() -> Path:
    """Storage directory: $CHROMA_PATH if set and non-empty, else DEFAULT_PATH."""
    value = os.getenv("CHROMA_PATH", "").strip()
    return Path(value) if value else DEFAULT_PATH


def default_embedding_function() -> EmbedFn:
    """Chroma's built-in embedder (all-MiniLM-L6-v2 via ONNX; downloads the model on first use).

    This is the one place to change when swapping embedders. Any callable
    matching EmbedFn can also be passed straight to ChromaVectorStore.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


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


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB-backed vector store.

    Embeddings are computed here with `embed` and handed to Chroma directly, so
    the collection itself carries no embedding function and the embedder can be
    swapped without touching Chroma config. Note that vectors from different
    embedders are not comparable: clear() and re-add after switching.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embed: EmbedFn | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else chroma_path()
        self.collection_name = collection_name
        self._embed = embed if embed is not None else default_embedding_function()
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            self.collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks. Re-adding an existing chunk_id overwrites it (upsert).

        Each chunk's Chroma metadata is its flattened Chunk.metadata plus
        `doc_id` (from Chunk.doc_id, used by delete) and `source` (from
        Chunk.metadata["source"], "" if missing, used for RetrievedChunk.source).
        """
        if not chunks:
            return
        metadatas = []
        for chunk in chunks:
            meta = flatten_metadata(chunk.metadata)
            meta["doc_id"] = chunk.doc_id
            meta["source"] = str(chunk.metadata.get("source") or "")
            metadatas.append(meta)
        texts = [chunk.text for chunk in chunks]
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=self._embed(texts),
            documents=texts,
            metadatas=metadatas,
        )

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

    def delete(self, doc_id: str) -> None:
        """Remove every chunk whose doc_id matches. A no-op if there are none."""
        self._collection.delete(where={"doc_id": doc_id})

    def clear(self) -> None:
        """Remove all chunks by dropping and recreating the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_collection()
