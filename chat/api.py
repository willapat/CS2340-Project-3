"""POST /api/chat: JSON in {"question": "..."}, JSON out (Answer.to_dict()).

Shared endpoint for US-05 (Ayan, request/response handling), US-06 (Max,
grounded answer), and US-07 (Zane, sources payload).
"""
import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from rag.llm import LLMError
from rag.pipeline import answer_question

logger = logging.getLogger(__name__)


@require_POST
def chat_api(request):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Request body must be JSON."}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"error": "Field 'question' is required."}, status=400)

    try:
        answer = answer_question(question)
    except LLMError as e:
        logger.warning("LLM failure: %s", e)
        return JsonResponse({"error": f"{e} Please try again."}, status=503)
    except Exception:
        logger.exception("Unexpected error answering a question")
        return JsonResponse({"error": "Something went wrong while answering. Please try again."}, status=500)
    return JsonResponse(answer.to_dict())
