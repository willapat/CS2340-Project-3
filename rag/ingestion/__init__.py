"""Ingestion methods. Add one file per method in this folder; each can subclass rag.types.Ingestor.

Story:  US-02: Two or more ingestion methods
Owner:  Ryon  |  Sprint 1

User story:
    As a system administrator, I want to load official documents into Alert
    Assist through at least two different ingestion methods, so that the
    knowledge base can be built from the formats our jurisdiction publishes
    its emergency-alert protocols in.

TODO(US-02): Acceptance criteria: see docs/stories.md (Product Owner to fill in).

Methods:
    files.FileIngestor   Local files or folders: PDF, plain text, Markdown, HTML.
    web.URLIngestor      A web page or online PDF, by URL.

Where this sits in the pipeline:

    ingest() -> list[Document] -> rag.preprocessing (US-03) -> rag.dataset / vector store (US-04)

Try it:  python -m rag.ingestion <file | folder | url>
"""
from __future__ import annotations

from pathlib import Path

from rag.ingestion.extract import IngestionError
from rag.ingestion.files import FileIngestor
from rag.ingestion.web import URLIngestor, is_url
from rag.types import Document

__all__ = ["FileIngestor", "IngestionError", "URLIngestor", "ingest", "is_url"]


def ingest(source: str, root: str | Path | None = None) -> list[Document]:
    """Load `source` with the right method: URLIngestor for http(s) URLs, else FileIngestor.

    `root` is FileIngestor's root for doc_ids (default data/protocols). For folders, files
    that couldn't be read are skipped; use FileIngestor directly to see them in `.errors`.
    """
    if is_url(source):
        return URLIngestor().ingest(source)
    return FileIngestor(root).ingest(source)
