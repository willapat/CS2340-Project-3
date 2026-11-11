"""Tests for US-05: Ask natural-language questions (chat UI). Owner: Ayan."""
import json

import pytest
from django.urls import reverse

from chat.api import MAX_QUESTION_LENGTH
from rag.llm import LLMError
from rag.types import Answer


def post_json(client, body, raw=False):
    data = body if raw else json.dumps(body)
    return client.post(reverse("chat:chat_api"), data=data, content_type="application/json")


#Chat page

def test_chat_page_renders_input_and_examples(client):
    resp = client.get(reverse("chat:chat_page"))
    assert resp.status_code == 200
    html = resp.content.decode()
    assert 'id="chat-form"' in html
    assert 'id="chat-input"' in html
    assert reverse("chat:chat_api") in html
    assert "csrfmiddlewaretoken" in html
    assert "chat-example" in html


#API: happy path

@pytest.fixture
def fake_pipeline(monkeypatch):
    """The real pipeline needs a knowledge base and an API key (tested in US-06); fake it here."""
    monkeypatch.setattr("chat.api.answer_question", lambda q: Answer(text=f"Answer to {q}", latency_ms=5))


def test_question_returns_answer_text_and_sources(client, fake_pipeline):
    resp = post_json(client, {"question": "What alert level for a flash flood?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"]
    assert isinstance(data["sources"], list)
    assert "latency_ms" in data


def test_question_is_passed_to_pipeline(client, monkeypatch):
    seen = {}

    def fake_answer(question):
        seen["question"] = question
        return Answer(text="Use a Flash Flood Warning.", prompt_type="general")

    monkeypatch.setattr("chat.api.answer_question", fake_answer)
    resp = post_json(client, {"question": "  what   alert\nlevel?  "})
    assert resp.status_code == 200
    assert resp.json()["text"] == "Use a Flash Flood Warning."
    assert seen["question"] == "what alert level?"  # whitespace normalised


def test_answer_includes_sources_for_us07(client, monkeypatch):
    from rag.types import Chunk, RetrievedChunk

    src = RetrievedChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="excerpt"), score=0.9, source="County EOP")
    monkeypatch.setattr("chat.api.answer_question", lambda q: Answer(text="See plan [1].", sources=[src]))
    resp = post_json(client, {"question": "Who can issue an evacuation order?"})
    assert resp.status_code == 200
    assert resp.json()["sources"][0]["source"] == "County EOP"


#API: bad input

@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "   "}, {"question": 42}, {"question": None}])
def test_missing_or_blank_question_is_rejected(client, body):
    resp = post_json(client, body)
    assert resp.status_code == 400
    assert resp.json()["error"]


def test_too_long_question_is_rejected(client):
    resp = post_json(client, {"question": "a" * (MAX_QUESTION_LENGTH + 1)})
    assert resp.status_code == 400
    assert str(MAX_QUESTION_LENGTH) in resp.json()["error"]


def test_max_length_question_is_accepted(client, fake_pipeline):
    resp = post_json(client, {"question": "a" * MAX_QUESTION_LENGTH})
    assert resp.status_code == 200


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"just a string"'])
def test_invalid_json_is_rejected(client, raw):
    resp = post_json(client, raw, raw=True)
    assert resp.status_code == 400
    assert resp.json()["error"]


def test_get_is_not_allowed(client):
    assert client.get(reverse("chat:chat_api")).status_code == 405


# API: failure

def test_pipeline_error_returns_friendly_500(client, monkeypatch):
    def boom(question):
        raise RuntimeError("vector store down")

    monkeypatch.setattr("chat.api.answer_question", boom)
    resp = post_json(client, {"question": "What alert level for a wildfire?"})
    assert resp.status_code == 500
    error = resp.json()["error"]
    assert "vector store" not in error  # internals are not leaked to the user
    assert error


def test_llm_failure_returns_503(client, monkeypatch):
    def llm_down(question):
        raise LLMError("The AI service is busy.")

    monkeypatch.setattr("chat.api.answer_question", llm_down)
    resp = post_json(client, {"question": "What alert level for a flood?"})
    assert resp.status_code == 503
    assert "try again" in resp.json()["error"]
