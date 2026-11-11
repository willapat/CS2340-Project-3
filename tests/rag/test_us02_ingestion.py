"""Tests for US-02: Two or more ingestion methods. Owner: Ryon."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rag.ingestion import FileIngestor, IngestionError, URLIngestor, ingest, is_url
from rag.ingestion import __main__ as cli
from rag.ingestion import web
from rag.ingestion.extract import decode_text, html_to_text, pdf_to_text
from rag.ingestion.web import FetchResult
from rag.preprocessing import preprocess_documents
from tests.rag.pdf_fixtures import blank_pdf, encrypt_pdf, make_pdf

FLOOD_PDF_PAGES = [
    ["Kerr County Flood Plan", "4.2 Warning Triggers", "When the river reaches flood stage, sound the siren."],
    ["Kerr County Flood Plan", "Evacuate low-lying areas along the river."],
]

FLOOD_HTML = """<!doctype html>
<html><head><title>Flash Flood Alerts | Kerr County</title><style>p {color: red}</style></head>
<body>
  <nav><a href="/">Home</a> <a href="/news">News</a></nav>
  <main>
    <h1>Flash Flood
        Alerts</h1>
    <p>Issue a <b>Wireless Emergency Alert</b> when the gauge
       exceeds 20&nbsp;ft &amp; rising.</p>
    <h2>Who decides</h2>
    <ul><li>The county judge orders evacuations</li><li>The emergency manager sends the alert</li></ul>
    <svg><title>icon</title></svg>
    <script>trackVisitor();</script>
  </main>
  <footer>Copyright 2026</footer>
</body></html>"""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_pdf_pages_joined_with_form_feed():
    text, title, pages = pdf_to_text(make_pdf(FLOOD_PDF_PAGES, title="Kerr Flood Plan"))

    assert pages == 2
    assert title == "Kerr Flood Plan"
    first, second = text.split("\f")
    assert "4.2 Warning Triggers" in first
    assert "Evacuate low-lying areas" in second


def test_pdf_with_empty_open_password_is_read():
    text, _, _ = pdf_to_text(encrypt_pdf(make_pdf(FLOOD_PDF_PAGES), user_password=""))

    assert "sound the siren" in text


@pytest.mark.parametrize(
    "data, message",
    [
        (encrypt_pdf(make_pdf(FLOOD_PDF_PAGES), user_password="secret"), "password-protected"),
        (blank_pdf(), "no extractable text"),
        (b"%PDF-1.4 truncated", "damaged"),
    ],
)
def test_unreadable_pdfs_raise_clear_errors(data, message):
    with pytest.raises(IngestionError, match=message):
        pdf_to_text(data)


def test_html_to_text_keeps_content_and_structure():
    text, title = html_to_text(FLOOD_HTML)

    assert title == "Flash Flood Alerts | Kerr County"
    assert text.startswith("# Flash Flood Alerts\n\n")
    assert "Issue a Wireless Emergency Alert when the gauge exceeds 20 ft & rising." in text
    assert "## Who decides\n\n- The county judge orders evacuations\n- The emergency manager sends the alert" in text
    for chrome in ("Home", "News", "Copyright", "trackVisitor", "color: red", "icon"):
        assert chrome not in text


def test_html_pre_keeps_line_breaks():
    text, _ = html_to_text("<pre>Step 1\nStep 2</pre><p>a\nb</p>")

    assert "Step 1\nStep 2" in text
    assert "a b" in text


def test_decode_text_fallbacks():
    assert decode_text("Évacuation".encode("utf-8")) == "Évacuation"
    assert decode_text("\ufeffBOM".encode("utf-8")) == "BOM"
    assert decode_text("“Shelter” – now".encode("cp1252")) == "“Shelter” – now"
    assert decode_text("ñ".encode("latin-1"), charset="latin-1") == "ñ"
    assert decode_text(b"plain", charset="not-a-charset") == "plain"


# ---------------------------------------------------------------------------
# Method 1: local files and folders
# ---------------------------------------------------------------------------


@pytest.fixture
def protocols(tmp_path):
    root = tmp_path / "protocols"
    (root / "kerr").mkdir(parents=True)
    (root / "kerr" / "flood-plan.pdf").write_bytes(make_pdf(FLOOD_PDF_PAGES, title="Kerr Flood Plan"))
    (root / "kerr" / "alerts.html").write_text(FLOOD_HTML, encoding="utf-8")
    (root / "wildfire.md").write_text("# Wildfire\n\nOpen shelters when smoke is reported.", encoding="utf-8")
    (root / "notes.txt").write_bytes("Use “plain language” in every alert.".encode("cp1252"))
    (root / ".gitkeep").write_text("")
    (root / ".drafts").mkdir()
    (root / ".drafts" / "secret.txt").write_text("draft")
    (root / "budget.docx").write_bytes(b"PK\x03\x04")
    (root / "scanned.pdf").write_bytes(blank_pdf())
    return root


def test_file_ingestor_reads_a_pdf(protocols):
    [doc] = FileIngestor(root=protocols).ingest(str(protocols / "kerr" / "flood-plan.pdf"))

    assert doc.doc_id == "file:kerr/flood-plan.pdf"
    assert doc.source == "Kerr Flood Plan"
    assert doc.metadata == {"format": "pdf", "filename": "flood-plan.pdf", "pages": 2}
    assert "\f" in doc.text


def test_file_ingestor_reads_a_folder(protocols):
    ingestor = FileIngestor(root=protocols)

    docs = ingestor.ingest(str(protocols))

    assert [d.doc_id for d in docs] == [
        "file:kerr/alerts.html",
        "file:kerr/flood-plan.pdf",
        "file:notes.txt",
        "file:wildfire.md",
    ]
    by_id = {d.doc_id: d for d in docs}
    assert by_id["file:kerr/alerts.html"].source == "Flash Flood Alerts | Kerr County"
    assert by_id["file:notes.txt"].text == "Use “plain language” in every alert."
    assert by_id["file:wildfire.md"].source == "wildfire.md"
    assert by_id["file:wildfire.md"].metadata["format"] == "markdown"

    assert set(ingestor.errors) == {str(protocols / "budget.docx"), str(protocols / "scanned.pdf")}
    assert "unsupported file type '.docx'" in ingestor.errors[str(protocols / "budget.docx")]
    assert "no extractable text" in ingestor.errors[str(protocols / "scanned.pdf")]


def test_file_ids_are_stable_across_runs(protocols):
    first = [d.doc_id for d in FileIngestor(root=protocols).ingest(str(protocols))]
    second = [d.doc_id for d in FileIngestor(root=protocols).ingest(str(protocols / "."))]

    assert first == second


def test_file_outside_root_uses_absolute_path(tmp_path, protocols):
    [doc] = FileIngestor(root=tmp_path / "elsewhere").ingest(str(protocols / "wildfire.md"))

    assert doc.doc_id == f"file:{(protocols / 'wildfire.md').resolve().as_posix()}"


def test_single_bad_file_raises(protocols):
    ingestor = FileIngestor(root=protocols)

    with pytest.raises(IngestionError, match="budget.docx: unsupported"):
        ingestor.ingest(str(protocols / "budget.docx"))
    with pytest.raises(IngestionError, match="no such file or folder"):
        ingestor.ingest(str(protocols / "missing.pdf"))
    (protocols / "empty.txt").write_text("  \n ")
    with pytest.raises(IngestionError, match="no text"):
        ingestor.ingest(str(protocols / "empty.txt"))


# ---------------------------------------------------------------------------
# Method 2: URLs
# ---------------------------------------------------------------------------


def fake_fetcher(content_type, body, charset=None, final_url=None):
    calls = []

    def fetcher(url):
        calls.append(url)
        return FetchResult(url=final_url or url, content_type=content_type, charset=charset, body=body)

    fetcher.calls = calls
    return fetcher


def test_url_ingestor_reads_a_web_page():
    fetcher = fake_fetcher("text/html", FLOOD_HTML.encode("utf-8"), charset="utf-8")

    [doc] = URLIngestor(fetcher).ingest("https://kerrcounty.gov/alerts#flood")

    assert fetcher.calls == ["https://kerrcounty.gov/alerts"]
    assert doc.doc_id == "url:https://kerrcounty.gov/alerts"
    assert doc.source == "Flash Flood Alerts | Kerr County"
    assert doc.metadata == {"format": "html", "url": "https://kerrcounty.gov/alerts"}
    assert "## Who decides" in doc.text


def test_url_ingestor_records_redirects():
    fetcher = fake_fetcher("text/plain", b"Shelter in place.", final_url="https://new.example.gov/plan.txt")

    [doc] = URLIngestor(fetcher).ingest("http://example.gov/plan.txt")

    assert doc.doc_id == "url:http://example.gov/plan.txt"
    assert doc.metadata["final_url"] == "https://new.example.gov/plan.txt"
    assert doc.source == "http://example.gov/plan.txt"


@pytest.mark.parametrize(
    "content_type, url",
    [
        ("application/pdf", "https://example.gov/download?id=7"),
        ("application/octet-stream", "https://example.gov/plan.pdf"),
        ("text/html", "https://example.gov/mislabelled"),  # %PDF- signature wins
    ],
)
def test_url_ingestor_reads_online_pdfs(content_type, url):
    fetcher = fake_fetcher(content_type, make_pdf(FLOOD_PDF_PAGES, title="Kerr Flood Plan"))

    [doc] = URLIngestor(fetcher).ingest(url)

    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["pages"] == 2
    assert doc.source == "Kerr Flood Plan"


def test_url_ingestor_errors():
    with pytest.raises(IngestionError, match="only http"):
        URLIngestor(fake_fetcher("text/html", b"")).ingest("file:///etc/passwd")
    with pytest.raises(IngestionError, match="unsupported content type 'image/png'"):
        URLIngestor(fake_fetcher("image/png", b"\x89PNG")).ingest("https://example.gov/map.png")
    with pytest.raises(IngestionError, match="has no text"):
        URLIngestor(fake_fetcher("text/html", b"<html><script>x()</script></html>")).ingest("https://example.gov/")
    with pytest.raises(IngestionError, match="https://example.gov/scan.pdf: the PDF has no extractable text"):
        URLIngestor(fake_fetcher("application/pdf", blank_pdf())).ingest("https://example.gov/scan.pdf")


@pytest.fixture
def local_server():
    """A real HTTP server on localhost, so fetch() is tested without the internet."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/page":
                body = "<title>Café plan</title><p>Évacuez.</p>".encode("latin-1")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=ISO-8859-1")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/big-declared":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b"x" * 1000)
            elif self.path == "/big-streamed":  # no Content-Length: size is only known by reading
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"x" * 1000)
            else:
                self.send_error(404, "Not Found")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_fetch_and_ingest_over_http(local_server):
    [doc] = URLIngestor().ingest(f"{local_server}/page")

    assert doc.source == "Café plan"
    assert doc.text == "Évacuez."


def test_fetch_http_error(local_server):
    with pytest.raises(IngestionError, match="HTTP 404"):
        web.fetch(f"{local_server}/missing")


@pytest.mark.parametrize("path", ["/big-declared", "/big-streamed"])
def test_fetch_size_limit(local_server, monkeypatch, path):
    monkeypatch.setattr(web, "MAX_BYTES", 100)

    with pytest.raises(IngestionError, match="larger than"):
        web.fetch(f"{local_server}{path}")


def test_fetch_unreachable_host():
    with pytest.raises(IngestionError, match="couldn't connect"):
        web.fetch("http://127.0.0.1:9/")  # discard port: nothing listens there


# ---------------------------------------------------------------------------
# Dispatch, pipeline, CLI
# ---------------------------------------------------------------------------


def test_is_url_and_ingest_dispatch(protocols, monkeypatch):
    assert is_url("https://example.gov") and is_url("HTTP://example.gov")
    assert not is_url("data/protocols") and not is_url("C:/plans/flood.pdf")

    assert [d.doc_id for d in ingest(str(protocols / "wildfire.md"), root=protocols)] == ["file:wildfire.md"]
    monkeypatch.setattr(web, "fetch", fake_fetcher("text/plain", b"Shelter in place."))
    assert ingest("https://example.gov/plan.txt")[0].doc_id == "url:https://example.gov/plan.txt"


def test_ingested_documents_feed_preprocessing(protocols):
    # US-03 strips repeated page headers only from documents with 3+ pages.
    three_pages = FLOOD_PDF_PAGES + [["Kerr County Flood Plan", "Reopen roads only after inspection."]]
    (protocols / "kerr" / "flood-plan.pdf").write_bytes(make_pdf(three_pages, title="Kerr Flood Plan"))
    docs = FileIngestor(root=protocols).ingest(str(protocols))

    chunks = preprocess_documents(docs)

    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}
    pdf_chunks = [c for c in chunks if c.doc_id == "file:kerr/flood-plan.pdf"]
    assert all(c.metadata["source"] == "Kerr Flood Plan" for c in pdf_chunks)
    assert all("Kerr County Flood Plan" not in c.text for c in pdf_chunks)  # repeated page header stripped
    html_sections = {c.metadata.get("section") for c in chunks if c.doc_id == "file:kerr/alerts.html"}
    assert "Who decides" in html_sections


def test_cli(protocols, capsys):
    assert cli.main(["rag.ingestion", str(protocols / "wildfire.md")]) == 0
    assert "chunks" in capsys.readouterr().out

    assert cli.main(["rag.ingestion", str(protocols)]) == 0
    out = capsys.readouterr().out
    assert "4 documents, 2 skipped" in out

    assert cli.main(["rag.ingestion", str(protocols / "missing")]) == 1
    assert cli.main(["rag.ingestion"]) == 2


def test_html_prefers_main_content():
    html = """<body><div class="menu"><ul><li>Home</li><li>Forecast</li></ul></div>
    <div role="main"><h1>Turn Around</h1><div><p>Never drive into floodwater.</p></div></div>
    <div>Site map</div></body>"""

    text, _ = html_to_text(html)

    assert text == "# Turn Around\n\nNever drive into floodwater."


def test_html_skips_aria_chrome_and_survives_unclosed_tags():
    html = """<div role="navigation"><div><a>Home</a></div><div><a>News</a></div></div>
    <div role="banner">Kerr County</div><span aria-hidden="true">*</span>
    <p>First paragraph<p>Second paragraph</div></span><ul><li>One<li>Two</ul>"""

    text, _ = html_to_text(html)

    assert text == "First paragraph\n\nSecond paragraph\n\n- One\n- Two"


def test_html_skips_chrome_named_by_class_or_id():
    html = """<div class="topnav"><ul id="topnav"><li>HOME</li><li>FORECAST</li></ul></div>
    <div class="div-full breadcrumb">Home > Safety</div><div id="site-search">Search</div>
    <div class="content"><h1>Turn Around Don't Drown</h1><p>Just 6 inches of water can knock you down.</p></div>
    <div class="footer-links">Privacy</div>"""

    text, _ = html_to_text(html)

    assert text == "# Turn Around Don't Drown\n\nJust 6 inches of water can knock you down."
