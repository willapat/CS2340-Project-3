"""Tests for US-03: Clean and preprocess data. Owner: Max."""
from rag.preprocessing import PreprocessConfig, chunk_text, clean_text, preprocess_document, preprocess_documents
from rag.types import Chunk, Document

# Three "pages" of a county plan as a PDF extractor would emit them: running header on every
# page, page-number footers, a word hyphenated across a line break, a ligature, bullets.
PLAN = (
    "KERR COUNTY EMERGENCY OPERATIONS PLAN\nAnnex A - Warning\n\n"
    "4.2 Warning Triggers\n\n"
    "The Emergency Manage-\nment Coordinator shall issue a ﬂash flood\nwarning when the "
    "Guadalupe River gauge at Hunt\nexceeds 12 feet.\n\n"
    "• Notify the County Judge\n• Activate WEA for the river valley\n\nPage 1 of 3\f"
    "KERR COUNTY EMERGENCY OPERATIONS PLAN\nAnnex A - Warning\n\n"
    "4.3 Authority\n\nThe County Judge has final authority over public warnings. If the judge cannot "
    "be reached within 10 minutes, the Coordinator may send the alert.\n\nPage 2 of 3\f"
    "KERR COUNTY EMERGENCY OPERATIONS PLAN\nAnnex A - Warning\n\n"
    "ARTÍCULO 5\n\nEl técnico de Protección Civil redactará el mensaje ES-Alert.\n\nPage 3 of 3"
)


def doc(text, doc_id="plan", source="plan.pdf", **metadata):
    return Document(doc_id=doc_id, text=text, source=source, metadata=metadata)


# --- Cleaning -------------------------------------------------------------------------------


def test_removes_repeated_headers_and_page_numbers():
    cleaned = clean_text(PLAN)

    assert "EMERGENCY OPERATIONS PLAN" not in cleaned
    assert "Annex A - Warning" not in cleaned
    assert "Page 1 of 3" not in cleaned and "Page 3 of 3" not in cleaned


def test_keeps_repeated_line_in_middle_of_page():
    body = "\n".join(f"Detail line {n} of the procedure." for n in range(4))
    pages = [f"Header\nIntro text for page {i}.\n{body}\n- Call 911\n{body}\nEnd of page {i}.\nFooter" for i in range(4)]

    cleaned = clean_text("\f".join(pages))

    assert cleaned.count("- Call 911") == 4
    assert "Header" not in cleaned and "Footer" not in cleaned


def test_rejoins_hyphenated_words_and_unwraps_lines():
    cleaned = clean_text(PLAN)

    assert "Emergency Management Coordinator shall issue a flash flood warning when the Guadalupe" in cleaned


def test_normalises_ligatures_bullets_and_keeps_accents():
    cleaned = clean_text(PLAN)

    assert "ﬂ" not in cleaned
    assert "- Notify the County Judge\n- Activate WEA for the river valley" in cleaned
    assert "El técnico de Protección Civil redactará" in cleaned


def test_strips_html_entities_and_invisible_characters():
    cleaned = clean_text("<p>Fire &amp; smoke</p>​zero width﻿\x07")

    assert cleaned == "Fire & smoke zero width"


def test_collapses_whitespace_into_single_blank_lines():
    cleaned = clean_text("First   paragraph.\n\n\n\n\nSecond\t\tparagraph.")

    assert cleaned == "First paragraph.\n\nSecond paragraph."


def test_empty_text():
    assert clean_text("") == ""
    assert preprocess_document(doc("   \n\f  ")) == []


# --- Chunking -------------------------------------------------------------------------------


def test_headings_start_new_chunks_and_are_recorded_as_section():
    chunks = preprocess_document(doc(PLAN), PreprocessConfig(min_chars=10))

    assert [c.metadata["section"] for c in chunks] == ["4.2 Warning Triggers", "4.3 Authority", "ARTÍCULO 5"]
    assert chunks[0].text.startswith("4.2 Warning Triggers")
    assert "12 feet" in chunks[0].text and "County Judge has final authority" in chunks[1].text


def test_chunks_respect_max_size_and_never_split_words():
    words = [f"word{i}" for i in range(2000)]
    text = " ".join(f"{' '.join(words[i:i + 12])}." for i in range(0, 2000, 12))
    config = PreprocessConfig(max_chars=300, overlap_chars=60)

    chunks = chunk_text(clean_text(text), config)

    assert len(chunks) > 5
    assert all(len(c) <= 300 for c, _ in chunks)
    vocabulary = set(words)
    assert all(token.rstrip(".") in vocabulary for c, _ in chunks for token in c.split())


def test_oversized_sentence_without_punctuation_is_still_split():
    chunks = chunk_text("evacuate " * 500, PreprocessConfig(max_chars=200, overlap_chars=0))

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c, _ in chunks)


def test_consecutive_chunks_overlap_by_whole_sentences():
    sentences = [f"Sentence number {i} about river levels." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), PreprocessConfig(max_chars=300, overlap_chars=100))

    first, second = chunks[0][0], chunks[1][0]
    last_sentence_of_first = first.split(". ")[-1].rstrip(".") + "."
    assert second.startswith(last_sentence_of_first) or last_sentence_of_first in second[:100]


# --- Document -> Chunk ----------------------------------------------------------------------


def test_chunk_ids_metadata_and_types():
    chunks = preprocess_document(doc(PLAN, county="Kerr"), PreprocessConfig(min_chars=10))

    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.chunk_id for c in chunks] == ["plan:0", "plan:1", "plan:2"]
    assert all(c.doc_id == "plan" for c in chunks)
    first = chunks[0].metadata
    assert first["source"] == "plan.pdf" and first["county"] == "Kerr"
    assert first["chunk_index"] == 0 and first["char_count"] == len(chunks[0].text)


def test_reprocessing_gives_identical_chunk_ids():
    assert preprocess_document(doc(PLAN)) == preprocess_document(doc(PLAN))


def test_drops_near_empty_chunks_but_keeps_a_short_single_chunk_document():
    text = "# Overview\n\nOk.\n\n# Evacuation\n\n" + "Residents in Zone 3 must evacuate to the high school shelter. " * 3

    chunks = preprocess_document(doc(text))

    assert [c.metadata["section"] for c in chunks] == ["Evacuation"]
    assert [c.text for c in preprocess_document(doc("Shelter open."))] == ["Shelter open."]


def test_drops_duplicate_chunks_within_and_across_documents():
    para = "Sound the siren when the river gauge exceeds 12 feet at the Hunt crossing."
    # Same section twice in one document, differing only in case/punctuation...
    a = doc(f"# Sirens\n\n{para}\n\n# SIRENS\n\n{para.upper()}!", doc_id="a")
    # ...and the same section again in a second document (e.g. an older copy of the plan).
    b = doc(f"# Sirens\n\n{para}\n\n# Shelters\n\nShelters open at the high school when sirens sound.", doc_id="b")

    chunks = preprocess_documents([a, b])

    assert [c.chunk_id for c in chunks] == ["a:0", "b:0"]
    assert chunks[1].metadata["section"] == "Shelters"
