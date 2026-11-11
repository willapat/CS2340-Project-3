"""POST /api/chat: JSON in {"question": "..."}, JSON out (Answer.to_dict()).

Shared endpoint for US-05 (Ayan, request/response handling), US-06 (Max,
grounded answer), and US-07 (Zane, sources payload).

US-05 owns: validating the question and turning failures into clear JSON
errors the chat UI can show. US-06 owns answer_question() and LLMError.
"""
import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from rag.llm import LLMError
from rag.pipeline import answer_question

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 1000


def _error(message: str, status: int) -> JsonResponse:
    return JsonResponse({"error": message}, status=status)


def _clean_question(raw) -> str:
    """Collapse runs of whitespace so the pipeline gets a tidy one-line question."""
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())


@require_POST
def chat_api(request):
    try:
        payload = json.loads(request.body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error("Request body must be JSON.", 400)

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)

    question = _clean_question(payload.get("question"))
    if not question:
        return _error("Please type a question.", 400)
    if len(question) > MAX_QUESTION_LENGTH:
        return _error(f"Questions can be at most {MAX_QUESTION_LENGTH} characters.", 400)

    try:
        answer = answer_question(question)
    except LLMError as e:
        # US-06: the LLM is unreachable, rate-limited, etc. Temporary, so 503.
        logger.warning("LLM failure: %s", e)
        return _error(f"{e} Please try again.", 503)
    except Exception:
        logger.exception("Unexpected error answering a question")
        return _error("Something went wrong while answering. Please try again.", 500)

    return JsonResponse(answer.to_dict())
