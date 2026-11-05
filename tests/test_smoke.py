"""Walking-skeleton smoke tests. These must stay green; they are not tied to a story."""
import json

import pytest
from django.urls import reverse


def test_chat_api_returns_placeholder_answer(client):
    resp = client.post(
        reverse("chat:chat_api"),
        data=json.dumps({"question": "What alert level for a flash flood?"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.json()["text"]


@pytest.mark.parametrize(
    "url_name",
    [
        "chat:chat_page",
        "alerts:conditions",
        "alerts:messages",
        "dashboard:home",
        "dashboard:knowledge_base",
        "dashboard:monitoring",
        "dashboard:usage_logs",
    ],
)
def test_pages_render(client, url_name):
    assert client.get(reverse(url_name)).status_code == 200


@pytest.mark.django_db
def test_querylog_model_is_migrated():
    from dashboard.models import QueryLog

    QueryLog.objects.create(question="q", prompt_type="general", latency_ms=12, success=True)
    assert QueryLog.objects.count() == 1
