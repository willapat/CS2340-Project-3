"""POST /api/chat: JSON in {"question": "..."}, JSON out (Answer.to_dict()).

Shared endpoint for US-05 (Ayan, request/response handling), US-06 (Max,
grounded answer), and US-07 (Zane, sources payload).

WALKING SKELETON: currently returns a hard-coded placeholder Answer.
PLACEHOLDER - replace in US-05/06/07.
"""
import json

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from rag.placeholder import placeholder_answer


@require_POST
def chat_api(request):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Request body must be JSON."}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"error": "Field 'question' is required."}, status=400)

    # PLACEHOLDER - replace with rag.pipeline.answer_question(question) in US-05/06/07.
    answer = placeholder_answer(question)
    return JsonResponse(answer.to_dict())
