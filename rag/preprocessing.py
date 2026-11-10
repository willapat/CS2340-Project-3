"""Cleaning and chunking of ingested documents (Document -> Chunk).

Story:  US-03: Clean and preprocess data
Owner:  Max  |  Sprint 1

User story:
    As a system administrator, I want ingested documents to be cleaned and
    split into well-formed chunks, so that retrieval returns relevant,
    readable passages instead of noise.

Where this sits in the pipeline:

    US-02 ingestors -> list[Document] -> preprocess_documents() -> list[Chunk] -> VectorStore.add()

Cleaning (clean_text):
    - Unicode NFKC normalisation (PDF ligatures like "ﬁ" -> "fi", full-width characters);
      accents are kept, so Spanish/Valencian text survives untouched.
    - HTML entities decoded and stray tags removed (scraped pages).
    - Invisible/control characters removed.
    - Repeated page headers/footers and page-number lines removed (pages split on form feed "\\f",
      which is what most PDF text extractors emit between pages).
    - Words hyphenated across a line break re-joined ("evacu-\\nation" -> "evacuation").
    - Hard-wrapped lines inside a paragraph joined; bullets normalised to "- "; whitespace collapsed.

Chunking (chunk_text):
    - Section-aware: a heading ("4.2 Warning Triggers", "SECTION 3", "# Title", "Artículo 5") starts a
      new chunk, and every chunk records the heading it falls under in metadata["section"].
    - Paragraphs are packed up to max_chars; an oversized paragraph is split on sentence boundaries,
      and an oversized sentence on word boundaries, so no chunk ends mid-word.
    - Consecutive chunks of the same section overlap by up to overlap_chars of whole sentences, so a
      fact that straddles a boundary is still retrievable.

Filtering (preprocess_document / preprocess_documents):
    - Chunks with too little real text (min_chars) are dropped, unless it's the document's only chunk.
    - Exact duplicate chunks (after normalising case/punctuation/whitespace) are dropped, within and
      across documents.

Try it on a file:  python -m rag.preprocessing path/to/file.txt
"""
from __future__ import annotations

import hashlib
import html
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag.types import Chunk, Document


@dataclass(frozen=True)
class PreprocessConfig:
    """Tuning knobs. Defaults suit all-MiniLM-L6-v2 (Chroma's default embedder, ~256-token window)."""

    max_chars: int = 1200
    overlap_chars: int = 200
    min_chars: int = 40
    # A line is treated as a repeated header/footer if it appears on at least this share of pages...
    header_footer_page_ratio: float = 0.5
    # ...and the document has at least this many pages...
    header_footer_min_pages: int = 3
    # ...and it is among the first/last this-many non-blank lines of the page (so a bullet that
    # happens to repeat in the middle of several pages, like "- Call 911", is never dropped).
    header_footer_edge_lines: int = 3


DEFAULT_CONFIG = PreprocessConfig()

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Zero-width chars and direction marks, line/paragraph separators, word joiner, BOM, soft hyphen.
_INVISIBLE = re.compile("[\u200b-\u200f\u2028\u2029\u2060\ufeff\u00ad]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")  # keeps \t \n \f \r
_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,200}>")
_PAGE_NUMBER = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:page|p[aá]gina|p\.)?\s*\d{1,4}(?:\s*(?:of|de|/)\s*\d{1,4})?(?:\s*[-–—])?\s*$",
    re.IGNORECASE,
)
_HYPHEN_BREAK = re.compile(r"(\w)-\n[ \t]*([a-zà-ÿ])")
_BULLET = re.compile(r"^[ \t]*[•●▪◦■□➢►‣∙·*][ \t]*", re.MULTILINE)
_LIST_ITEM = re.compile(r"^(?:- |\(?\d{1,3}[.)]\s|\(?[a-z][.)]\s)")
_SPACES = re.compile(r"[ \t]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_KEYWORD_HEADING = re.compile(
    r"^(section|article|art[ií]culo|chapter|cap[ií]tulo|annex|anexo|appendix|ap[eé]ndice|part|parte)\s+[\w.]+",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\.?\s+[A-ZÀ-Ý]")


def _edge_lines(page: str, n: int) -> set[int]:
    """Indexes of the first and last n non-blank lines of a page (where headers/footers live)."""
    non_blank = [i for i, line in enumerate(page.splitlines()) if line.strip()]
    return set(non_blank[:n] + non_blank[-n:]) if n > 0 else set()


def _remove_headers_and_footers(text: str, config: PreprocessConfig) -> str:
    """Drop page-number lines everywhere, and lines repeated at the top/bottom of most pages."""
    pages = text.split("\f")
    edges = [_edge_lines(page, config.header_footer_edge_lines) for page in pages]
    repeated: set[str] = set()
    if len(pages) >= config.header_footer_min_pages:
        counts: Counter[str] = Counter()
        for page, edge in zip(pages, edges):
            lines = page.splitlines()
            counts.update({lines[i].strip().lower() for i in edge})
        threshold = max(2, int(len(pages) * config.header_footer_page_ratio))
        repeated = {line for line, n in counts.items() if n >= threshold}

    kept_pages = []
    for page, edge in zip(pages, edges):
        lines = [
            line
            for i, line in enumerate(page.splitlines())
            if not _PAGE_NUMBER.match(line) and not (i in edge and line.strip().lower() in repeated)
        ]
        kept_pages.append("\n".join(lines))
    # A page break is also a paragraph break.
    return "\n\n".join(kept_pages)


def _is_heading(line: str) -> bool:
    """Heuristic: does this single line look like a section heading?"""
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if line.startswith("#"):
        return bool(re.match(r"^#{1,6}\s+\S", line))
    if line.endswith((".", ",", ";", ":")):
        return False
    if _KEYWORD_HEADING.match(line):
        return True
    words = line.split()
    if _NUMBERED_HEADING.match(line) and len(words) <= 10:
        return True
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and len(words) <= 10 and line == line.upper() and not _LIST_ITEM.match(line)


def _unwrap_paragraph(paragraph: str) -> str:
    """Join hard-wrapped lines, but keep list items and headings on their own lines."""
    out: list[str] = []
    for raw in paragraph.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if out and not _LIST_ITEM.match(line) and not _is_heading(line) and not _is_heading(out[-1]):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def clean_text(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> str:
    """Return a cleaned copy of raw extracted text. Paragraphs are separated by one blank line."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\t", " ").replace("\u00a0", " ")  # tabs and non-breaking spaces
    text = _remove_headers_and_footers(text, config)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BULLET.sub("- ", text)
    text = _SPACES.sub(" ", text)

    paragraphs = [_unwrap_paragraph(p) for p in re.split(r"\n[ \t]*\n", text)]
    text = "\n\n".join(p for p in paragraphs if p)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Split after ., !, ? (optionally followed by a closing quote/bracket) when whitespace follows.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+")


def _split_sentences(paragraph: str) -> list[str]:
    parts: list[str] = []
    for line in paragraph.split("\n"):  # list items are already one per line
        parts.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return parts


def _split_long(piece: str, max_chars: int) -> list[str]:
    """Split a single sentence longer than max_chars on word boundaries."""
    out, current = [], ""
    for word in piece.split():
        if current and len(current) + 1 + len(word) > max_chars:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        out.append(current)
    return out


def _units(paragraph: str, max_chars: int) -> tuple[list[str], bool]:
    """Pieces of a paragraph that each fit in max_chars, and whether the paragraph stayed whole."""
    if len(paragraph) <= max_chars:
        return [paragraph], True
    units: list[str] = []
    for sentence in _split_sentences(paragraph):
        units.extend(_split_long(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    return units, False


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """The last whole sentences of `text` that fit in overlap_chars ("" if none fit)."""
    if overlap_chars <= 0:
        return ""
    tail: list[str] = []
    size = 0
    for sentence in reversed(_split_sentences(text)):
        if size + len(sentence) + 1 > overlap_chars:
            break
        tail.insert(0, sentence)
        size += len(sentence) + 1
    return " ".join(tail)


def chunk_text(text: str, config: PreprocessConfig = DEFAULT_CONFIG) -> list[tuple[str, str | None]]:
    """Split cleaned text into (chunk_text, section_heading) pairs. Headings start a new chunk."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for paragraph in text.split("\n\n"):
        first_line, _, rest = paragraph.partition("\n")
        if _is_heading(first_line):
            heading = first_line.lstrip("#").strip()
            sections.append((heading, [heading]))
            if rest.strip():
                sections[-1][1].append(rest.strip())
        else:
            sections[-1][1].append(paragraph)

    chunks: list[tuple[str, str | None]] = []
    for heading, paragraphs in sections:
        current = ""
        for paragraph in paragraphs:
            units, whole = _units(paragraph, config.max_chars)
            sep = "\n\n" if whole else " "
            for unit in units:
                if current and len(current) + len(sep) + len(unit) > config.max_chars:
                    chunks.append((current, heading))
                    tail = _overlap_tail(current, config.overlap_chars)
                    current = f"{tail} {unit}" if tail and len(tail) + 1 + len(unit) <= config.max_chars else unit
                else:
                    current = f"{current}{sep}{unit}" if current else unit
        # Skip empty sections and headings with no body (e.g. a chapter title right before a sub-heading).
        if current and current != heading:
            chunks.append((current, heading))
    return chunks


# ---------------------------------------------------------------------------
# Document -> Chunk
# ---------------------------------------------------------------------------

def _fingerprint(text: str) -> str:
    normalised = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def _has_enough_text(text: str, min_chars: int) -> bool:
    return sum(c.isalnum() for c in text) >= min_chars


def preprocess_document(
    doc: Document,
    config: PreprocessConfig = DEFAULT_CONFIG,
    seen: set[str] | None = None,
) -> list[Chunk]:
    """Clean and chunk one Document.

    Chunk ids are "<doc_id>:<n>" (stable, so re-ingesting a document overwrites its chunks).
    Each chunk's metadata is the document's metadata plus: source, chunk_index, char_count and,
    when known, section. `seen` holds fingerprints of chunks already produced; pass the same set
    across documents to drop cross-document duplicates.
    """
    seen = set() if seen is None else seen
    pieces = chunk_text(clean_text(doc.text, config), config)
    keep_single = len(pieces) == 1

    chunks: list[Chunk] = []
    for text, section in pieces:
        if not keep_single and not _has_enough_text(text, config.min_chars):
            continue
        fingerprint = _fingerprint(text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        index = len(chunks)
        metadata = {**doc.metadata, "source": doc.source, "chunk_index": index, "char_count": len(text)}
        if section:
            metadata["section"] = section
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}:{index}", doc_id=doc.doc_id, text=text, metadata=metadata))
    return chunks


def preprocess_documents(docs: list[Document], config: PreprocessConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Clean and chunk many Documents, dropping duplicate chunks across all of them."""
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(preprocess_document(doc, config, seen))
    return chunks


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m rag.preprocessing <text-file>")
        return 2
    path = Path(argv[1])
    doc = Document(doc_id=path.stem, text=path.read_text(encoding="utf-8", errors="replace"), source=path.name)
    chunks = preprocess_document(doc)
    for chunk in chunks:
        print(f"--- {chunk.chunk_id}  section={chunk.metadata.get('section')!r}  chars={len(chunk.text)}")
        print(chunk.text)
    print(f"\n{len(doc.text)} raw chars -> {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
