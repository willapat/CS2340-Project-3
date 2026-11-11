"""Tests for US-04: Define / update the dataset. Owner: Ryon."""
import math
import os
import re
from types import SimpleNamespace

import pytest

from rag import dataset
from rag.types import Chunk, Document, VectorStore
from rag.vectorstore import (
    DEFAULT_GEMINI_EMBED_MODEL,
    GEMINI_BATCH_SIZE,
    GEMINI_EMBED_DIMENSIONS,
    ChromaVectorStore,
    EmbedderMismatchError,
    GeminiEmbedder,
    embedder_from_env,
    flatten_metadata,
)

VOCAB = ["flood", "fire", "evacuate", "shelter", "wind", "siren", "river", "smoke"]


def fake_embed(texts):
    """Deterministic bag-of-words embedder so tests need no model download."""
    vectors = []
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        vec = [float(words.count(term)) for term in VOCAB] + [0.01]  # bias keeps the vector non-zero
        norm = math.sqrt(sum(x * x for x in vec))
        vectors.append([x / norm for x in vec])
    return vectors


def make_chunk(chunk_id, doc_id, text, **metadata):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, metadata=metadata)


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed)


@pytest.fixture
def two_docs():
    return [
        make_chunk("flood-0", "flood-plan", "River flood: evacuate low areas", source="Flood Plan.pdf"),
        make_chunk("flood-1", "flood-plan", "Sound the flood siren", source="Flood Plan.pdf"),
        make_chunk("fire-0", "fire-plan", "Wildfire smoke: open a fire shelter", source="Fire Plan.pdf"),
    ]


def test_is_a_vector_store(store):
    assert isinstance(store, VectorStore)


def test_add_then_query_ranks_relevant_chunk_first(store, two_docs):
    store.add(two_docs)

    results = store.query("fire and smoke", k=2)

    assert len(results) == 2
    top = results[0]
    assert top.chunk.chunk_id == "fire-0"
    assert top.chunk.doc_id == "fire-plan"
    assert top.chunk.text == "Wildfire smoke: open a fire shelter"
    assert top.source == "Fire Plan.pdf"
    assert top.metadata["doc_id"] == "fire-plan"
    assert top.score > results[1].score
    assert -1.0 <= results[1].score <= top.score <= 1.0 + 1e-6


def test_exact_match_scores_near_one(store, two_docs):
    store.add(two_docs)

    top = store.query("Sound the flood siren", k=1)[0]

    assert top.chunk.chunk_id == "flood-1"
    assert top.score == pytest.approx(1.0, abs=1e-3)


def test_query_caps_k_at_store_size(store, two_docs):
    store.add(two_docs)

    assert len(store.query("flood", k=50)) == 3
    assert store.query("flood", k=0) == []


def test_delete_removes_only_that_document(store, two_docs):
    store.add(two_docs)

    store.delete("flood-plan")

    results = store.query("flood river siren", k=10)
    assert [r.chunk.chunk_id for r in results] == ["fire-0"]


def test_delete_then_add_replaces_document(store, two_docs):
    store.add(two_docs)

    store.delete("flood-plan")
    store.add([make_chunk("flood-0", "flood-plan", "Updated: shelter in place during flood", source="Flood Plan v2.pdf")])

    flood_results = [r for r in store.query("flood", k=10) if r.chunk.doc_id == "flood-plan"]
    assert [r.chunk.text for r in flood_results] == ["Updated: shelter in place during flood"]
    assert flood_results[0].source == "Flood Plan v2.pdf"


def test_delete_unknown_doc_is_noop(store, two_docs):
    store.add(two_docs)

    store.delete("does-not-exist")

    assert len(store.query("flood", k=10)) == 3


def test_clear_empties_store_and_store_still_works(store, two_docs):
    store.add(two_docs)

    store.clear()

    assert store.query("flood", k=5) == []
    store.add(two_docs[:1])
    assert [r.chunk.chunk_id for r in store.query("flood", k=5)] == ["flood-0"]


def test_query_empty_store_returns_empty_list(store):
    assert store.query("anything", k=5) == []


def test_add_empty_list_is_noop(store):
    store.add([])

    assert store.query("flood") == []


def test_re_adding_chunk_id_overwrites(store):
    store.add([make_chunk("c1", "d1", "flood", source="v1")])
    store.add([make_chunk("c1", "d1", "flood river", source="v2")])

    results = store.query("flood", k=10)
    assert len(results) == 1
    assert results[0].chunk.text == "flood river"
    assert results[0].source == "v2"


def test_data_persists_across_instances(tmp_path, two_docs):
    ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed).add(two_docs)

    reopened = ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed)

    assert len(reopened.query("flood", k=10)) == 3


def test_missing_source_defaults_to_empty_string(store):
    store.add([make_chunk("c1", "d1", "flood")])

    assert store.query("flood")[0].source == ""


def test_non_scalar_metadata_is_flattened(store):
    store.add([make_chunk("c1", "d1", "flood", source="s", page=3, tags=["a", "b"], extra={"k": 1}, skip=None)])

    result = store.query("flood")[0]

    assert result.chunk.metadata == {"source": "s", "page": 3, "tags": '["a", "b"]', "extra": '{"k": 1}'}


def test_flatten_metadata_keeps_scalars():
    assert flatten_metadata({"s": "x", "i": 1, "f": 1.5, "b": True}) == {"s": "x", "i": 1, "f": 1.5, "b": True}


@pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="Downloads Chroma's default embedding model; set RUN_SLOW=1.")
def test_default_embedding_function_end_to_end(tmp_path):
    store = ChromaVectorStore(path=tmp_path / "chroma")
    store.add(
        [
            make_chunk("a", "d1", "Issue an evacuation order for the river flood zone.", source="flood"),
            make_chunk("b", "d2", "Wildfire smoke advisory for downwind neighborhoods.", source="fire"),
        ]
    )

    assert store.query("flooding near the river", k=1)[0].source == "flood"


# ---------------------------------------------------------------------------
# Store: clear() with several instances, embedder tracking, document summaries
# ---------------------------------------------------------------------------


def test_clear_keeps_other_open_instances_working(tmp_path, two_docs):
    chat_store = ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed)
    admin_store = ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed)
    chat_store.add(two_docs)

    admin_store.clear()

    assert chat_store.query("flood", k=5) == []
    chat_store.add(two_docs[:1])
    assert [r.chunk.chunk_id for r in admin_store.query("flood", k=5)] == ["flood-0"]


def test_reopening_with_different_embedder_raises(tmp_path, two_docs):
    ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed, embedder_name="a").add(two_docs)

    with pytest.raises(EmbedderMismatchError, match="'a'"):
        ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed, embedder_name="b")


def test_empty_store_can_switch_embedder(tmp_path, two_docs):
    store = ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed, embedder_name="a")
    store.add(two_docs)
    store.clear()

    switched = ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed, embedder_name="b")
    switched.add(two_docs)

    with pytest.raises(EmbedderMismatchError):
        ChromaVectorStore(path=tmp_path / "chroma", embed=fake_embed, embedder_name="a")


def test_documents_summarises_store(store, two_docs):
    store.add(two_docs)

    assert [(d.doc_id, d.source, d.chunk_count) for d in store.documents()] == [
        ("fire-plan", "Fire Plan.pdf", 1),
        ("flood-plan", "Flood Plan.pdf", 2),
    ]
    assert store.document("flood-plan").chunk_count == 2
    assert store.document("missing") is None


def test_replace_document_leaves_no_stale_chunks(store, two_docs):
    store.add(two_docs)

    store.replace_document("flood-plan", [make_chunk("flood-0", "flood-plan", "River flood shelter", source="v2")])

    assert store.document("flood-plan").chunk_count == 1
    assert store.document("fire-plan").chunk_count == 1


def test_replace_document_keeps_old_version_if_embedding_fails(tmp_path, two_docs):
    calls = {"fail": False}

    def flaky_embed(texts):
        if calls["fail"]:
            raise RuntimeError("embedding API down")
        return fake_embed(texts)

    store = ChromaVectorStore(path=tmp_path / "chroma", embed=flaky_embed)
    store.add(two_docs)
    calls["fail"] = True

    with pytest.raises(RuntimeError, match="API down"):
        store.replace_document("flood-plan", [make_chunk("flood-0", "flood-plan", "new text")])

    assert store.document("flood-plan").chunk_count == 2


def test_replace_document_rejects_mismatched_or_empty_chunks(store):
    with pytest.raises(ValueError, match="doc_id"):
        store.replace_document("a", [make_chunk("b:0", "b", "flood")])
    with pytest.raises(ValueError, match="delete"):
        store.replace_document("a", [])


# ---------------------------------------------------------------------------
# Embedder selection
# ---------------------------------------------------------------------------


class FakeGeminiClient:
    """Stands in for google.genai.Client: records calls, returns one vector per text."""

    def __init__(self):
        self.calls = []
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(t)), 1.0]) for t in contents])


def test_gemini_embedder_batches_requests(monkeypatch):
    monkeypatch.delenv("GEMINI_EMBED_MODEL", raising=False)
    client = FakeGeminiClient()
    embedder = GeminiEmbedder(client=client)

    vectors = embedder([f"text {i}" for i in range(GEMINI_BATCH_SIZE + 5)])

    assert len(vectors) == GEMINI_BATCH_SIZE + 5
    assert [len(contents) for _, contents, _ in client.calls] == [GEMINI_BATCH_SIZE, 5]
    model, _, config = client.calls[0]
    assert model == DEFAULT_GEMINI_EMBED_MODEL
    assert config == {"output_dimensionality": GEMINI_EMBED_DIMENSIONS}
    assert embedder.name == f"gemini:{DEFAULT_GEMINI_EMBED_MODEL}:{GEMINI_EMBED_DIMENSIONS}"


def test_gemini_embedder_needs_api_key_only_when_used(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    embedder = GeminiEmbedder()  # no key needed yet

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embedder(["flood"])


def test_gemini_embedder_works_in_store(tmp_path, two_docs):
    embedder = GeminiEmbedder(client=FakeGeminiClient())
    store = ChromaVectorStore(path=tmp_path / "chroma", embed=embedder, embedder_name=embedder.name)

    store.add(two_docs)

    assert len(store.query("flood", k=10)) == 3


def test_embedder_from_env(monkeypatch):
    monkeypatch.setenv("CHROMA_EMBEDDER", "gemini")
    monkeypatch.setenv("GEMINI_EMBED_MODEL", "some-model")
    name, embed = embedder_from_env()
    assert name == f"gemini:some-model:{GEMINI_EMBED_DIMENSIONS}"
    assert isinstance(embed, GeminiEmbedder)

    monkeypatch.setenv("CHROMA_EMBEDDER", "nope")
    with pytest.raises(ValueError, match="CHROMA_EMBEDDER"):
        embedder_from_env()


# ---------------------------------------------------------------------------
# rag.dataset: define / update the knowledge base
# ---------------------------------------------------------------------------

FLOOD_V1 = "Flood warning protocol.\n\nWhen the river reaches flood stage, sound the siren and evacuate low areas."
FLOOD_V2 = "Flood warning protocol, revised.\n\nAt flood stage, residents shelter on high ground; no siren."
FIRE = "Wildfire protocol.\n\nWhen smoke is reported, open the fire shelter and evacuate downwind areas."


def doc(doc_id, text, source=None, **metadata):
    return Document(doc_id=doc_id, text=text, source=source or f"{doc_id}.pdf", metadata=metadata)


def test_upsert_adds_then_skips_unchanged(store):
    first = dataset.upsert_document(doc("flood", FLOOD_V1), store)
    again = dataset.upsert_document(doc("flood", FLOOD_V1), store)

    assert (first.status, again.status) == ("added", "unchanged")
    assert again.chunk_count == first.chunk_count >= 1
    stored = store.document("flood")
    assert stored.source == "flood.pdf"
    assert stored.content_hash == dataset.content_hash(doc("flood", FLOOD_V1))
    assert stored.indexed_at


def test_upsert_unchanged_does_not_re_embed(tmp_path):
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    store = ChromaVectorStore(path=tmp_path / "chroma", embed=counting_embed)
    dataset.upsert_document(doc("flood", FLOOD_V1), store)
    embedded = len(calls)

    dataset.upsert_document(doc("flood", FLOOD_V1), store)

    assert len(calls) == embedded
    assert dataset.upsert_document(doc("flood", FLOOD_V1), store, force=True).status == "updated"
    assert len(calls) > embedded


def test_upsert_changed_document_replaces_old_version(store):
    dataset.upsert_document(doc("flood", FLOOD_V1), store)

    result = dataset.upsert_document(doc("flood", FLOOD_V2, source="flood-v2.pdf"), store)

    assert result.status == "updated"
    texts = " ".join(r.chunk.text for r in store.query("flood siren river", k=10))
    assert "revised" in texts
    assert "sound the siren" not in texts
    assert store.document("flood").source == "flood-v2.pdf"


def test_metadata_change_counts_as_update(store):
    dataset.upsert_document(doc("flood", FLOOD_V1, county="Kerr"), store)

    assert dataset.upsert_document(doc("flood", FLOOD_V1, county="Travis"), store).status == "updated"
    assert store.query("flood")[0].metadata["county"] == "Travis"


def test_upsert_rejects_bad_doc_id_and_empty_text(store):
    for bad in ["", "  ", " flood"]:
        with pytest.raises(dataset.DatasetError, match="doc_id"):
            dataset.upsert_document(doc(bad, FLOOD_V1), store)
    with pytest.raises(dataset.DatasetError, match="no usable text"):
        dataset.upsert_document(doc("empty", "   \n\n "), store)


def test_failed_update_keeps_previous_version(store):
    dataset.upsert_document(doc("flood", FLOOD_V1), store)

    with pytest.raises(dataset.DatasetError):
        dataset.upsert_document(doc("flood", ""), store)

    assert store.document("flood").content_hash == dataset.content_hash(doc("flood", FLOOD_V1))


def test_remove_and_list_documents(store):
    dataset.upsert_document(doc("flood", FLOOD_V1), store)
    dataset.upsert_document(doc("fire", FIRE), store)

    assert [d.doc_id for d in dataset.list_documents(store)] == ["fire", "flood"]
    assert dataset.remove_document("flood", store) is True
    assert dataset.remove_document("flood", store) is False
    assert [d.doc_id for d in dataset.list_documents(store)] == ["fire"]


def test_sync_makes_store_match_definition(store):
    dataset.sync([doc("flood", FLOOD_V1), doc("fire", FIRE)], store)

    report = dataset.sync([doc("flood", FLOOD_V2), doc("evac", "Evacuation routes follow the river road.")], store)

    assert report.added == ["evac"]
    assert report.updated == ["flood"]
    assert report.unchanged == []
    assert report.removed == ["fire"]
    assert report.ok
    assert report.summary() == "1 added, 1 updated, 0 unchanged, 1 removed, 0 failed"
    assert [d.doc_id for d in dataset.list_documents(store)] == ["evac", "flood"]


def test_sync_without_prune_keeps_other_documents(store):
    dataset.sync([doc("fire", FIRE)], store)

    report = dataset.sync([doc("flood", FLOOD_V1)], store, prune=False)

    assert report.removed == []
    assert [d.doc_id for d in dataset.list_documents(store)] == ["fire", "flood"]


def test_sync_records_failures_and_continues(store):
    dataset.sync([doc("flood", FLOOD_V1)], store)

    report = dataset.sync([doc("flood", ""), doc("fire", FIRE)], store)

    assert not report.ok
    assert "no usable text" in report.failed["flood"]
    assert report.added == ["fire"]
    assert store.document("flood") is not None  # failed doc is still wanted, so not pruned


def test_sync_rejects_duplicate_ids_before_changing_anything(store):
    dataset.sync([doc("fire", FIRE)], store)

    with pytest.raises(dataset.DatasetError, match="more than once"):
        dataset.sync([doc("flood", FLOOD_V1), doc("flood", FLOOD_V2)], store)

    assert [d.doc_id for d in dataset.list_documents(store)] == ["fire"]


def test_cli_list_and_remove(store, capsys):
    dataset.upsert_document(doc("flood", FLOOD_V1), store)

    assert dataset._main(["rag.dataset", "list"], store) == 0
    assert "flood" in capsys.readouterr().out
    assert dataset._main(["rag.dataset", "remove", "flood"], store) == 0
    assert dataset._main(["rag.dataset", "remove", "flood"], store) == 1
    assert dataset._main(["rag.dataset"], store) == 2
