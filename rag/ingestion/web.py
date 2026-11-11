"""Ingestion method 2 (US-02): a web page or online PDF, by URL.

One URL gives one Document; links are not followed. PDFs (by Content-Type, a ".pdf" path
served as a generic binary, or the "%PDF-" signature) go through the PDF extractor, since
agencies often link straight to their plan PDFs. HTML and plain text are also supported.

doc_id is "url:" + the URL without its #fragment, so re-loading the same URL updates the
same document. Note: PythonAnywhere free accounts can only reach whitelisted domains, so on
the server some agency sites may be unreachable.
"""
from __future__ import annotations

import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urldefrag, urlsplit

from rag.ingestion.extract import IngestionError, decode_text, html_to_text, pdf_to_text
from rag.types import Document, Ingestor

TIMEOUT_SECONDS = 20
MAX_BYTES = 20 * 1024 * 1024
USER_AGENT = "AlertAssist/1.0 (CS 2340 emergency-protocol assistant)"

_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_GENERIC_BINARY = {"application/octet-stream", "binary/octet-stream", "application/x-download"}


@dataclass
class FetchResult:
    url: str  # final URL, after redirects
    content_type: str  # e.g. "text/html", lowercase, without parameters
    charset: str | None
    body: bytes


Fetcher = Callable[[str], FetchResult]


def fetch(url: str) -> FetchResult:
    """GET `url` with a timeout and a size limit. Raises IngestionError on any failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > MAX_BYTES:
                raise IngestionError(f"{url}: larger than the {MAX_BYTES // (1024 * 1024)} MB limit")
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise IngestionError(f"{url}: larger than the {MAX_BYTES // (1024 * 1024)} MB limit")
            return FetchResult(
                url=response.geturl(),
                content_type=response.headers.get_content_type(),
                charset=response.headers.get_content_charset(),
                body=body,
            )
    except urllib.error.HTTPError as e:
        raise IngestionError(f"{url}: the server returned HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise IngestionError(f"{url}: couldn't connect ({e.reason})") from e
    except (TimeoutError, socket.timeout) as e:
        raise IngestionError(f"{url}: timed out after {TIMEOUT_SECONDS}s") from e


def is_url(source: str) -> bool:
    return urlsplit(source).scheme.lower() in ("http", "https")


def _kind(result: FetchResult) -> str | None:
    if result.content_type == "application/pdf" or result.body.startswith(b"%PDF-"):
        return "pdf"
    if result.content_type in _GENERIC_BINARY and urlsplit(result.url).path.lower().endswith(".pdf"):
        return "pdf"
    if result.content_type in _HTML_TYPES:
        return "html"
    if result.content_type == "text/plain":
        return "text"
    return None


class URLIngestor(Ingestor):
    """Reads one web page or online PDF. `fetcher` is for tests; normally fetch() is used."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetch = fetcher if fetcher is not None else fetch

    def ingest(self, source: str) -> list[Document]:
        url = urldefrag(source.strip()).url
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            raise IngestionError(f"{source}: only http:// and https:// URLs are supported")

        result = self._fetch(url)
        kind = _kind(result)
        if kind is None:
            raise IngestionError(f"{url}: unsupported content type {result.content_type!r}")

        title = None
        metadata: dict = {"format": kind, "url": url}
        if result.url != url:
            metadata["final_url"] = result.url
        try:
            if kind == "pdf":
                text, title, pages = pdf_to_text(result.body)
                metadata["pages"] = pages
            elif kind == "html":
                text, title = html_to_text(decode_text(result.body, result.charset))
            else:
                text = decode_text(result.body, result.charset)
        except IngestionError as e:
            raise IngestionError(f"{url}: {e}") from e
        if not text.strip():
            raise IngestionError(f"{url}: the page has no text")

        return [Document(doc_id=f"url:{url}", text=text, source=title or url, metadata=metadata)]
