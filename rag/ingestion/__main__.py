"""python -m rag.ingestion <file | folder | url>: show what would be loaded, without loading it."""
from __future__ import annotations

import sys

from rag.ingestion import FileIngestor, IngestionError, URLIngestor, is_url
from rag.preprocessing import preprocess_document


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m rag.ingestion <file | folder | url>")
        return 2
    source = argv[1]
    ingestor = URLIngestor() if is_url(source) else FileIngestor()
    try:
        docs = ingestor.ingest(source)
    except IngestionError as e:
        print(f"error: {e}")
        return 1

    for doc in docs:
        preview = " ".join(doc.text.split())[:200]
        chunks = preprocess_document(doc)
        print(f"--- {doc.doc_id}\n    source: {doc.source}\n    metadata: {doc.metadata}")
        print(f"    {len(doc.text)} chars -> {len(chunks)} chunks\n    {preview}")
    errors = getattr(ingestor, "errors", {})
    for path, reason in errors.items():
        print(f"skipped {path}: {reason}")
    print(f"\n{len(docs)} documents, {len(errors)} skipped")
    return 1 if errors and not docs else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
