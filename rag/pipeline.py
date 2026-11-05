"""End-to-end question answering. chat/api.py calls answer_question().

Story:  US-06: Grounded responses
Owner:  Max  |  Sprint 1

User story:
    As an emergency management official, I want the assistant's responses to
    be grounded in my jurisdiction's official documents, so that I can trust
    its guidance is not hallucinated.

TODO(US-06): Acceptance criteria: see docs/stories.md (Product Owner to fill in).
TODO(US-06): Design and build this however you like; this file is your space.
"""
from rag.types import Answer


def answer_question(question: str) -> Answer:
    """Answer `question` grounded in official documents. US-06 (Max)."""
    raise NotImplementedError("US-06 (Max)")
