"""Tests for US-06: Grounded responses. Owner: Max.

No network: a fake LLM records the prompt it receives and returns a scripted reply, and the
vector store is either an in-memory fake or the real Chroma store with a bag-of-words embedder.
"""
import json
import math
import re
from types import SimpleNamespace

import pytest
from django.urls import reverse

from rag import pipeline
from google.genai import errors as gemini_errors

from rag.llm import GeminiLLM, LLMError
from rag.pipeline import NO_ANSWER_TEXT, NOT_FOUND_TOKEN, SYSTEM_PROMPT, UNCITED_WARNING, answer_question
from rag.preprocessing import preprocess_document
from rag.retrieval import retrieve
from rag.types import Chunk, Document, LLMProvider, RetrievedChunk, VectorStore
from rag.vectorstore import ChromaVectorStore


class FakeLLM(LLMProvider):
    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.error:
            raise self.error
        return self.reply


class FakeStore(VectorStore):
    """Returns fixed results with fixed scores, whatever the question."""

    def __init__(self, results):
        self.results = results

    def add(self, chunks): ...
    def delete(self, doc_id): ...
    def clear(self): ...

    def query(self, text, k=5):
        return self.results[:k]


def hit(chunk_id, text, score, source="County EOP.pdf", **metadata):
    chunk = Chunk(chunk_id=chunk_id, doc_id=chunk_id.split(":")[0], text=text, metadata=metadata)
    return RetrievedChunk(chunk=chunk, score=score, source=source, metadata=metadata)


TRIGGER = hit("eop:0", "Issue a flash flood warning when the Hunt gauge exceeds 12 feet.", 0.82, section="4.2 Triggers")
AUTHORITY = hit("eop:1", "The County Judge has final authority over public warnings.", 0.61, section="4.3 Authority")
SHELTER = hit("eop:2", "Shelters open at the high school.", 0.40)
NOISE = hit("misc:0", "Library opening hours are 9 to 5.", 0.05)


@pytest.fixture
def store():
    return FakeStore([TRIGGER, AUTHORITY, SHELTER, NOISE])


# --- Retrieval ------------------------------------------------------------------------------


def test_retrieve_drops_irrelevant_and_duplicate_passages():
    dup = hit("eop-v2:0", TRIGGER.chunk.text.upper(), 0.80)
    results = retrieve("flood trigger", store=FakeStore([TRIGGER, dup, AUTHORITY, NOISE]), threshold=0.25)

    assert [r.chunk.chunk_id for r in results] == ["eop:0", "eop:1"]


def test_retrieve_threshold_and_k_come_from_environment(monkeypatch, store):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")
    monkeypatch.setenv("RAG_TOP_K", "2")

    assert [r.chunk.chunk_id for r in retrieve("q", store=store)] == ["eop:0", "eop:1"]


# --- Grounding ------------------------------------------------------------------------------


def test_prompt_contains_only_retrieved_passages_and_grounding_rules(store):
    llm = FakeLLM("Issue a warning above 12 feet [1].")

    answer_question("When do we warn?", store=store, llm=llm)

    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert "ONLY with information stated in the numbered passages" in SYSTEM_PROMPT
    prompt = call["prompt"]
    assert "[1] Source: County EOP.pdf | Section: 4.2 Triggers" in prompt
    assert TRIGGER.chunk.text in prompt and AUTHORITY.chunk.text in prompt
    assert NOISE.chunk.text not in prompt
    assert prompt.rstrip().endswith("Question: When do we warn?")


def test_no_relevant_documents_refuses_without_calling_llm():
    llm = FakeLLM("should not be used")

    answer = answer_question("What is the capital of France?", store=FakeStore([NOISE]), llm=llm)

    assert llm.calls == []
    assert answer.text == NO_ANSWER_TEXT
    assert answer.sources == []
    assert answer.metadata["grounded"] is False and answer.metadata["reason"] == "no_relevant_documents"


def test_llm_saying_not_in_documents_becomes_a_refusal(store):
    answer = answer_question("Who maintains the sirens?", store=store, llm=FakeLLM(NOT_FOUND_TOKEN))

    assert answer.text == NO_ANSWER_TEXT
    assert answer.sources == []
    assert answer.metadata == {"grounded": False, "reason": "not_in_documents", "retrieved_count": 3, "cited_count": 0}


def test_sources_are_only_cited_passages_renumbered_to_match(store):
    reply = "The County Judge decides [2]. Warn above 12 feet [1][2]. Bogus claim [7]."

    answer = answer_question("Who sends the alert and when?", store=store, llm=FakeLLM(reply))

    assert answer.text == "The County Judge decides [1]. Warn above 12 feet [2][1]. Bogus claim."
    assert [s.chunk.chunk_id for s in answer.sources] == ["eop:1", "eop:0"]  # [n] == sources[n-1]
    assert answer.metadata == {"grounded": True, "retrieved_count": 3, "cited_count": 2, "model": None}
    assert answer.prompt_type == "grounded_qa"
    assert answer.latency_ms >= 0


def test_uncited_answer_is_flagged_as_not_grounded(store):
    answer = answer_question("When do we warn?", store=store, llm=FakeLLM("Probably soon."))

    assert answer.text.endswith(UNCITED_WARNING)
    assert answer.sources == []
    assert answer.metadata["grounded"] is False


def test_empty_question_is_rejected(store):
    with pytest.raises(ValueError):
        answer_question("   ", store=store, llm=FakeLLM())


def test_llm_errors_propagate(store):
    with pytest.raises(LLMError):
        answer_question("q", store=store, llm=FakeLLM(error=LLMError("down")))


def test_answer_serialises_for_the_api(store):
    data = answer_question("q", store=store, llm=FakeLLM("Warn [1].")).to_dict()

    assert data["sources"][0]["source"] == "County EOP.pdf"
    assert data["sources"][0]["chunk"]["text"] == TRIGGER.chunk.text
    json.dumps(data)


# --- End to end: Document -> US-03 preprocessing -> US-04 Chroma store -> US-06 answer ------

VOCAB = ["flood", "gauge", "feet", "warning", "judge", "authority", "shelter", "library"]


def bag_of_words(texts):
    vectors = []
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        vec = [float(sum(w.startswith(term) for w in words)) for term in VOCAB] + [0.01]
        norm = math.sqrt(sum(x * x for x in vec))
        vectors.append([x / norm for x in vec])
    return vectors


def test_end_to_end_with_real_preprocessing_and_vector_store(tmp_path):
    plan = Document(
        doc_id="kerr-eop",
        source="Kerr County EOP.pdf",
        text=(
            "4.2 Warning Triggers\n\nIssue a flash flood warning when the river gauge exceeds 12 feet.\n\n"
            "4.3 Authority\n\nThe County Judge has final authority over every public warning.\n\n"
            "9.1 Library\n\nThe county library opens at nine in the morning on weekdays."
        ),
    )
    store = ChromaVectorStore(path=tmp_path / "chroma", embed=bag_of_words)
    store.add(preprocess_document(plan))
    llm = FakeLLM("Issue the warning once the gauge passes 12 feet [1].")

    answer = answer_question("At how many feet on the flood gauge do we issue a warning?", store=store, llm=llm)

    assert "12 feet [1]" in answer.text
    assert answer.sources[0].source == "Kerr County EOP.pdf"
    assert answer.sources[0].chunk.metadata["section"] == "4.2 Warning Triggers"
    assert "library opens" not in llm.calls[0]["prompt"]


# --- Gemini provider (SDK mocked) and the /api/chat endpoint --------------------------------


def gemini_response(text="Answer [1].", finish_reason="STOP", block_reason=None):
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))
    return SimpleNamespace(text=text, candidates=[candidate], prompt_feedback=SimpleNamespace(block_reason=block_reason))


class FakeModels:
    """Scripted per-model results: a response object, or an exception to raise."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        result = self.script.get(model, gemini_response())
        if isinstance(result, Exception):
            raise result
        return result


def fake_gemini(**script):
    """fake_gemini(m1=..., m2=...) -> (client, models); unknown models answer normally."""
    models = FakeModels({name.replace("_", "-"): result for name, result in script.items()})
    return SimpleNamespace(models=models), models


def rate_limited(daily=False, delay=None):
    detail = [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}] if daily else []
    if delay:
        detail.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{delay}s"})
    return gemini_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": detail}})


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_gemini_llm_sends_system_prompt_and_returns_text():
    client, models = fake_gemini()

    llm = GeminiLLM(models=["m1"], client=client)
    text = llm.generate("the prompt", system="the rules")

    assert text == "Answer [1]." and llm.last_model == "m1"
    call = models.calls[0]
    assert call["model"] == "m1" and call["contents"] == "the prompt"
    assert call["config"].system_instruction == "the rules"


def test_gemini_models_come_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODELS", " a , b ,c")

    assert GeminiLLM(client=object()).models == ["a", "b", "c"]


def test_falls_back_to_next_model_when_rate_limited():
    client, models = fake_gemini(m1=rate_limited(), m2=gemini_response("From m2 [1]."))

    llm = GeminiLLM(models=["m1", "m2", "m3"], client=client)

    assert llm.generate("p") == "From m2 [1]."
    assert llm.last_model == "m2"
    assert [c["model"] for c in models.calls] == ["m1", "m2"]


@pytest.mark.parametrize(
    "failure",
    [
        gemini_errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND"}}),
        gemini_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}),
        gemini_errors.ServerError(500, {"error": {"code": 500, "status": "INTERNAL"}}),
        gemini_response(text=""),
    ],
    ids=["not-found", "overloaded", "internal", "empty"],
)
def test_falls_back_on_unavailable_overloaded_or_empty(failure):
    client, _ = fake_gemini(m1=failure)

    llm = GeminiLLM(models=["m1", "m2"], client=client)

    assert llm.generate("p") == "Answer [1]." and llm.last_model == "m2"


def test_rate_limited_model_is_skipped_until_cooldown_ends():
    clock = FakeClock()
    client, models = fake_gemini(m1=rate_limited(delay=40))
    llm = GeminiLLM(models=["m1", "m2"], client=client, clock=clock)

    llm.generate("p")
    llm.generate("p")
    assert [c["model"] for c in models.calls] == ["m1", "m2", "m2"]  # m1 skipped the second time

    clock.now += 41
    models.script["m1"] = gemini_response("m1 is back [1].")
    assert llm.generate("p") == "m1 is back [1]." and llm.last_model == "m1"


def test_daily_quota_cools_down_for_an_hour():
    clock = FakeClock()
    client, _ = fake_gemini(m1=rate_limited(daily=True))
    llm = GeminiLLM(models=["m1", "m2"], client=client, clock=clock)

    llm.generate("p")
    clock.now += 600
    assert llm.available_models() == ["m2"]
    clock.now += 3001
    assert llm.available_models() == ["m1", "m2"]


def test_all_models_rate_limited_raises_clear_error():
    client, _ = fake_gemini(m1=rate_limited(), m2=rate_limited())
    llm = GeminiLLM(models=["m1", "m2"], client=client)

    with pytest.raises(LLMError, match="rate-limited"):
        llm.generate("p")
    with pytest.raises(LLMError, match="rate-limited"):  # both cooling down: fails fast, no calls
        llm.generate("p")


@pytest.mark.parametrize(
    "failure",
    [
        gemini_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid."}}),
        gemini_errors.ClientError(403, {"error": {"code": 403, "status": "PERMISSION_DENIED"}}),
        gemini_response(text="", finish_reason="SAFETY"),
        gemini_response(text=None, block_reason="SAFETY"),
    ],
    ids=["bad-key", "forbidden", "safety-finish", "safety-prompt"],
)
def test_does_not_fall_back_on_bad_key_or_safety_block(failure):
    client, models = fake_gemini(m1=failure)

    with pytest.raises(LLMError):
        GeminiLLM(models=["m1", "m2"], client=client).generate("p")
    assert [c["model"] for c in models.calls] == ["m1"]


def test_gemini_llm_without_key_raises_llm_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM(models=["m1"]).generate("p")


def test_answer_metadata_records_which_model_answered(store):
    class ModelLLM(FakeLLM):
        last_model = "gemini-test"

    answer = answer_question("When?", store=store, llm=ModelLLM("Warn above 12 feet [1]."))

    assert answer.metadata["model"] == "gemini-test"


def test_chat_api_returns_grounded_answer(client, monkeypatch, store):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, store=None, k=None: [TRIGGER])
    monkeypatch.setattr(pipeline, "get_llm", lambda: FakeLLM("Warn above 12 feet [1]."))

    resp = client.post(reverse("chat:chat_api"), data=json.dumps({"question": "When?"}), content_type="application/json")

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Warn above 12 feet [1]." and body["metadata"]["grounded"] is True
    assert body["sources"][0]["chunk"]["chunk_id"] == "eop:0"


def test_chat_api_reports_llm_failure_as_503(client, monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, store=None, k=None: [TRIGGER])
    monkeypatch.setattr(pipeline, "get_llm", lambda: FakeLLM(error=LLMError("Could not reach the LLM service.")))

    resp = client.post(reverse("chat:chat_api"), data=json.dumps({"question": "When?"}), content_type="application/json")

    assert resp.status_code == 503
    assert "Could not reach the LLM service." in resp.json()["error"]
