"""Text extraction shared by the ingestion methods (US-02): PDF, HTML and plain text.

Output is raw text for rag.preprocessing.clean_text (US-03), which does the real cleanup.
Two things are done here so preprocessing can use them:
    - PDF pages are joined with a form feed ("\\f"), which clean_text uses to find and
      strip repeated page headers/footers.
    - HTML headings become Markdown headings ("## Evacuation"), which chunk_text uses to
      start new, section-labelled chunks.
"""
from __future__ import annotations

import io
import re
from html.parser import HTMLParser

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class IngestionError(Exception):
    """A source couldn't be read. The message says which one and why."""


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def decode_text(data: bytes, charset: str | None = None) -> str:
    """Decode bytes using `charset` if given and valid, else UTF-8, else Windows-1252.

    Windows-1252 covers files saved by older Windows tools (curly quotes, accented
    letters); Latin-1 is the last resort because it never fails.
    """
    for encoding in (charset, "utf-8-sig", "cp1252"):
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("latin-1")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def pdf_to_text(data: bytes) -> tuple[str, str | None, int]:
    """Return (text with pages joined by "\\f", title from the PDF's metadata or None, page count)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many official PDFs are "encrypted" only to block editing, with an empty open password.
            try:
                opened = reader.decrypt("")
            except Exception:  # e.g. AES without the optional cryptography package
                opened = 0
            if not opened:
                raise IngestionError("the PDF is password-protected")
        pages = [page.extract_text() or "" for page in reader.pages]
        title = reader.metadata.title if reader.metadata else None
    except IngestionError:
        raise
    except PdfReadError as e:
        raise IngestionError(f"the file is damaged or isn't a real PDF ({e})") from e
    except Exception as e:  # pypdf raises assorted errors on malformed files; one bad PDF mustn't stop a folder
        raise IngestionError(f"the PDF couldn't be read ({type(e).__name__}: {e})") from e

    text = "\f".join(pages)
    if not text.strip():
        raise IngestionError("the PDF has no extractable text (scanned PDFs need OCR, which isn't supported)")
    title = title.strip() if isinstance(title, str) else None
    return text, title or None, len(pages)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

# Content inside these is never text a reader sees, or is site chrome rather than the document.
_SKIP = {"script", "style", "noscript", "template", "svg", "nav", "footer", "head", "iframe", "button"}
# ARIA landmarks that mark the same chrome on sites that don't use <nav>/<footer>.
_SKIP_ROLES = {"navigation", "banner", "contentinfo", "search"}
# Heuristic for older sites that use neither: an id/class containing one of these words
# (e.g. <div class="topnav">, <ul id="main-menu">, <div class="breadcrumb">).
_CHROME_NAME = re.compile(r"nav|menu|breadcrumb|footer|sidebar|search", re.IGNORECASE)
# These start and end a paragraph.
_BLOCK = {
    "p", "div", "section", "article", "main", "aside", "header", "blockquote", "pre", "table", "tr",
    "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "address", "form", "fieldset",
}
_HEADINGS = {f"h{n}": "#" * n for n in range(1, 7)}
# Elements that never have an end tag.
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class _HTMLText(HTMLParser):
    """Collects readable text. Keeps a stack of open elements (closing an element also
    closes any unclosed ones inside it, like browsers do) to know when it's inside
    skipped chrome, <pre>, or the page's <main> content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main_parts: list[str] = []
        self.title_parts: list[str] = []
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, skip, main)
        self._in_title = False
        self._seen_title = False

    def _flags(self) -> tuple[bool, bool, bool]:
        skip = any(s for _, s, _ in self._stack)
        main = any(m for _, _, m in self._stack)
        pre = any(tag == "pre" for tag, _, _ in self._stack)
        return skip, main, pre

    def _emit(self, text: str) -> None:
        skip, main, _ = self._flags()
        if skip:
            return
        self.parts.append(text)
        if main:
            self.main_parts.append(text)

    def handle_starttag(self, tag, attrs):
        if tag == "title" and not self._seen_title:  # the page's <title>, not e.g. an <svg>'s
            self._in_title = self._seen_title = True
            return
        if tag in _VOID:
            if tag == "br":
                self._emit("\n")
            return
        attributes = dict(attrs)
        role = (attributes.get("role") or "").lower()
        names = f"{attributes.get('id') or ''} {attributes.get('class') or ''}"
        skip = (
            tag in _SKIP
            or role in _SKIP_ROLES
            or attributes.get("aria-hidden") == "true"
            or bool(_CHROME_NAME.search(names))
        )
        self._stack.append((tag, skip, tag == "main" or role == "main"))
        if tag in _HEADINGS:
            self._emit(f"\n\n{_HEADINGS[tag]} ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag in _BLOCK:
            self._emit("\n\n")
        elif tag in ("td", "th"):
            self._emit(" ")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self._in_title = False
            return
        if not any(open_tag == tag for open_tag, _, _ in self._stack):
            return  # stray end tag
        if tag in _HEADINGS or tag in _BLOCK:
            self._emit("\n\n")
        while self._stack and self._stack.pop()[0] != tag:
            pass

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
            return
        # Source line breaks mean nothing in HTML outside <pre>; structure comes from tags.
        self._emit(data if self._flags()[2] else re.sub(r"\s+", " ", data))


def _tidy(parts: list[str]) -> str:
    lines = [" ".join(line.split()) for line in "".join(parts).splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:  # paragraph breaks, but never more than one blank line in a row
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def html_to_text(html: str) -> tuple[str, str | None]:
    """Return (readable text, page <title> or None). Headings become Markdown headings.

    If the page marks its content with <main> (or role="main"), only that is kept.
    Navigation, footers, scripts and similar chrome are always dropped, including
    elements whose id/class names them (see _CHROME_NAME).
    """
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    text = _tidy(parser.main_parts) or _tidy(parser.parts)
    title = " ".join("".join(parser.title_parts).split())
    return text, title or None
