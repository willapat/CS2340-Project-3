# Alert Assist

A RAG-based chatbot web app that helps government emergency officials execute their
jurisdiction's existing emergency-alert protocols under pressure: finding the right protocol,
recommending alert levels and target zones, and drafting alert messages grounded in official
documents. Motivated by alerting failures in the 2025 Texas Hill Country floods, the 2024
Valencia DANA floods, and the 2025 LA wildfires.

CS 2340, Georgia Tech. Stack: Python 3.11, Django, SQLite, ChromaDB, pytest, PythonAnywhere.

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

## Tests
```bash
pytest
ruff check .
```

## Who owns what
Each story has its own file(s). Build it however you want. The only shared pieces are
`rag/types.py` (the types passed between stories), `dashboard/models.py` (`QueryLog`, used by
US-13 and US-14), and `chat/api.py` (the `/api/chat` endpoint, currently returning a placeholder).

| Story | Sprint | Owner | Files |
|---|---|---|---|
| US-01 Hosted web app | 1 | Will | `config/`, `deploy/`, `templates/`, `static/` |
| US-02 Two+ ingestion methods | 1 | Ryon | `rag/ingestion/` |
| US-03 Clean/preprocess data | 1 | Max | `rag/preprocessing.py` |
| US-04 Define/update dataset | 1 | Ryon | `rag/dataset.py`, `rag/vectorstore.py` |
| US-05 Chat UI | 1 | Ayan | `chat/` |
| US-06 Grounded responses | 1 | Max | `rag/pipeline.py`, `rag/retrieval.py`, `rag/llm.py` |
| US-07 Show sources | 1 | Zane | `rag/citations.py`, `chat/static/chat/sources.*` |
| US-08 Runtime prompt selection | 2 | Will | `rag/prompts.py` |
| US-09 Conditions form → alert level + zones | 2 | Will | `alerts/recommendation.py`, `alerts/views/conditions.py` |
| US-10 Multilingual alert messages | 2 | Ayan | `alerts/messaging.py`, `alerts/views/messages.py`, `alerts/message_templates/` |
| US-11 Accuracy testing | 2 | Max | `evaluation/` |
| US-12 Admin clear/reload KB | 2 | Zane | `dashboard/kb_admin.py`, `dashboard/views/knowledge_base.py` |
| US-13 Admin monitoring | 2 | Ayan | `dashboard/metrics.py`, `dashboard/views/monitoring.py` |
| US-14 Admin usage logs | 2 | Ryon | `dashboard/usage.py`, `dashboard/views/usage_logs.py` |

Tests: `tests/**/test_usNN_*.py`. Stories and acceptance criteria: `docs/stories.md`.
Workflow: `CONTRIBUTING.md`.
