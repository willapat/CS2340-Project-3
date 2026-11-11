"""Ingestion method 1 (US-02): local files and folders.

Supported: .pdf, .txt, .md/.markdown, .html/.htm. A folder is read recursively in sorted
order; hidden files and folders (".gitkeep", ".DS_Store", ...) are skipped.

doc_id is "file:" + the path relative to `root` (default data/protocols), e.g.
"file:kerr-county/flood-plan.pdf", so the same file always gets the same id and US-04 can
tell an updated file from a new one. Files outside `root` use their absolute path, which
is only stable on that machine. Moving or renaming a file gives it a new id.
"""
from __future__ import annotations

from pathlib import Path

from rag.ingestion.extract import IngestionError, decode_text, html_to_text, pdf_to_text
from rag.types import Document, Ingestor

DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "protocols"

FORMATS = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
}


class FileIngestor(Ingestor):
    """Reads one file, or every supported file in a folder.

    A problem with a single file raises IngestionError. In a folder, problem files
    (unreadable, unsupported type, no text) are recorded in `self.errors`
    ({path: reason}, reset on each call) and the rest are still returned.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.errors: dict[str, str] = {}

    def ingest(self, source: str) -> list[Document]:
        self.errors = {}
        path = Path(source)
        if path.is_file():
            return [self._read(path)]
        if not path.is_dir():
            raise IngestionError(f"{source}: no such file or folder")

        docs = []
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            if any(part.startswith(".") for part in file.relative_to(path).parts):
                continue
            try:
                docs.append(self._read(file))
            except IngestionError as e:
                self.errors[str(file)] = str(e)
        return docs

    def doc_id(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return f"file:{resolved.relative_to(self.root.resolve()).as_posix()}"
        except ValueError:
            return f"file:{resolved.as_posix()}"

    def _read(self, path: Path) -> Document:
        fmt = FORMATS.get(path.suffix.lower())
        if fmt is None:
            supported = ", ".join(sorted(FORMATS))
            raise IngestionError(f"{path}: unsupported file type {path.suffix or '(none)'!r} (supported: {supported})")
        try:
            data = path.read_bytes()
        except OSError as e:
            raise IngestionError(f"{path}: couldn't be read ({e.strerror or e})") from e

        title = None
        metadata: dict = {"format": fmt, "filename": path.name}
        try:
            if fmt == "pdf":
                text, title, pages = pdf_to_text(data)
                metadata["pages"] = pages
            elif fmt == "html":
                text, title = html_to_text(decode_text(data))
            else:
                text = decode_text(data)
        except IngestionError as e:
            raise IngestionError(f"{path}: {e}") from e
        if not text.strip():
            raise IngestionError(f"{path}: the file has no text")

        return Document(doc_id=self.doc_id(path), text=text, source=title or path.name, metadata=metadata)
