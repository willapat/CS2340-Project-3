"""Defining and updating the knowledge-base dataset.

Story:  US-04: Define / update the dataset
Owner:  Ryon  |  Sprint 1

User story:
    As a system administrator, I want to define and update the set of
    official protocol documents the assistant draws from, so that its
    answers always reflect my jurisdiction's current protocols.

TODO(US-04): Acceptance criteria: see docs/stories.md (Product Owner to fill in).

Where this sits in the pipeline:

    US-02 ingestors -> list[Document] -> sync() / upsert_document() -> preprocess (US-03) -> vector store

Operations (all default to the app's shared store, rag.retrieval.get_vector_store(), so the
chat process sees changes immediately):
    upsert_document(doc)   Add a document, or replace its previous version. Re-submitting an
                           unchanged document is a no-op (no re-embedding). If preprocessing or
                           embedding fails, the previous version stays in place.
    remove_document(id)    Remove a document and all its chunks.
    list_documents()       What's in the knowledge base: id, source, chunk count, when indexed.
    sync(docs)             Make the knowledge base match `docs` exactly: add new, update changed,
                           skip unchanged, and (with prune=True) remove documents not in `docs`.

"Changed" means the text, source or metadata differ, compared by a content hash stored on
every chunk. After changing preprocessing settings, pass force=True to re-index anyway.

Command line:  python -m rag.dataset list
               python -m rag.dataset remove <doc_id>
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv

from rag.preprocessing import DEFAULT_CONFIG, PreprocessConfig, preprocess_document
from rag.types import Chunk, Document
from rag.vectorstore import StoredDocument


class DatasetStore(Protocol):
    """What this module needs from a store (ChromaVectorStore provides it)."""

    def documents(self) -> list[StoredDocument]: ...

    def document(self, doc_id: str) -> StoredDocument | None: ...

    def replace_document(self, doc_id: str, chunks: list[Chunk]) -> None: ...

    def delete(self, doc_id: str) -> None: ...


class DatasetError(ValueError):
    """A document can't be indexed as given (bad id, no usable text, duplicate ids, ...)."""


@dataclass
class UpdateResult:
    doc_id: str
    status: Literal["added", "updated", "unchanged"]
    chunk_count: int


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # doc_id -> error message

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (
            f"{len(self.added)} added, {len(self.updated)} updated, {len(self.unchanged)} unchanged, "
            f"{len(self.removed)} removed, {len(self.failed)} failed"
        )


def _default_store() -> DatasetStore:
    from rag.retrieval import get_vector_store

    return get_vector_store()  # type: ignore[return-value]


def content_hash(doc: Document) -> str:
    """SHA-256 over the document's text, source and metadata."""
    payload = json.dumps(
        {"text": doc.text, "source": doc.source, "metadata": doc.metadata}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_doc_id(doc_id: str) -> None:
    if not isinstance(doc_id, str) or not doc_id.strip() or doc_id != doc_id.strip():
        raise DatasetError(f"doc_id must be a non-empty string without surrounding whitespace; got {doc_id!r}.")


def upsert_document(
    doc: Document,
    store: DatasetStore | None = None,
    *,
    config: PreprocessConfig = DEFAULT_CONFIG,
    force: bool = False,
) -> UpdateResult:
    """Add `doc`, or replace the stored version if it changed. See the module docstring."""
    _check_doc_id(doc.doc_id)
    store = store if store is not None else _default_store()
    digest = content_hash(doc)
    existing = store.document(doc.doc_id)
    if existing is not None and existing.content_hash == digest and not force:
        return UpdateResult(doc.doc_id, "unchanged", existing.chunk_count)

    indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamped = Document(
        doc_id=doc.doc_id,
        text=doc.text,
        source=doc.source,
        metadata={**doc.metadata, "content_hash": digest, "indexed_at": indexed_at},
    )
    chunks = preprocess_document(stamped, config)
    if not chunks:
        raise DatasetError(f"Document {doc.doc_id!r} has no usable text after preprocessing.")
    store.replace_document(doc.doc_id, chunks)
    return UpdateResult(doc.doc_id, "updated" if existing is not None else "added", len(chunks))


def remove_document(doc_id: str, store: DatasetStore | None = None) -> bool:
    """Remove a document. Returns False if it wasn't in the knowledge base."""
    store = store if store is not None else _default_store()
    if store.document(doc_id) is None:
        return False
    store.delete(doc_id)
    return True


def list_documents(store: DatasetStore | None = None) -> list[StoredDocument]:
    """Every document in the knowledge base, sorted by doc_id."""
    store = store if store is not None else _default_store()
    return store.documents()


def sync(
    documents: list[Document],
    store: DatasetStore | None = None,
    *,
    prune: bool = True,
    config: PreprocessConfig = DEFAULT_CONFIG,
    force: bool = False,
) -> SyncReport:
    """Make the knowledge base match `documents`. See the module docstring.

    Raises DatasetError before changing anything if doc_ids are invalid or repeated.
    A document that fails to index is recorded in report.failed (its previous version,
    if any, is kept) and the rest of the sync carries on.
    """
    seen: set[str] = set()
    for doc in documents:
        _check_doc_id(doc.doc_id)
        if doc.doc_id in seen:
            raise DatasetError(f"doc_id {doc.doc_id!r} appears more than once.")
        seen.add(doc.doc_id)

    store = store if store is not None else _default_store()
    report = SyncReport()
    for doc in documents:
        try:
            result = upsert_document(doc, store, config=config, force=force)
        except Exception as e:  # keep going; the report says what failed
            report.failed[doc.doc_id] = f"{type(e).__name__}: {e}"
            continue
        getattr(report, result.status).append(doc.doc_id)

    if prune:
        for stored in store.documents():
            if stored.doc_id not in seen:
                store.delete(stored.doc_id)
                report.removed.append(stored.doc_id)
    return report


def _main(argv: list[str], store: DatasetStore | None = None) -> int:
    usage = "usage: python -m rag.dataset list | remove <doc_id>"
    if store is None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # CHROMA_PATH etc., as Django would
    if argv[1:] == ["list"]:
        docs = list_documents(store)
        for doc in docs:
            print(f"{doc.doc_id}\t{doc.chunk_count} chunks\tindexed {doc.indexed_at or '?'}\t{doc.source}")
        print(f"{len(docs)} documents")
        return 0
    if len(argv) == 3 and argv[1] == "remove":
        if remove_document(argv[2], store):
            print(f"Removed {argv[2]}")
            return 0
        print(f"No document with doc_id {argv[2]!r}")
        return 1
    print(usage)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
