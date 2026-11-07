"""Tests for US-04: Define / update the dataset. Owner: Ryon."""
import math
import os
import re

import pytest

from rag.types import Chunk, VectorStore
from rag.vectorstore import ChromaVectorStore, flatten_metadata

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
