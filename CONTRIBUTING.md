# Contributing

Course rule: each team member writes and tests their own story's code.

- **Branches:** one per story, named `feature/US-05-short-name`.
- **Commits:** `US-05: short summary`.
- **PRs:** into `main`, need one teammate's review, and link the Jira card. CI (ruff + pytest) must pass.
- **Shared files** (`rag/types.py`, `dashboard/models.py`, `chat/api.py`): tag the people affected when you change them.
- **Secrets/data:** never commit `.env`, API keys, `chroma_db/`, or large files.
