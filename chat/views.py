"""Chat page view.

Story:  US-05: Ask natural-language questions (chat UI)
Owner:  Ayan (front-end)
Sprint: 1

User story:
    As an emergency management official, I want to ask questions in plain
    natural language through a chat interface, so that I can quickly find the
    right protocol under pressure without searching through documents.

TODO(US-05): Acceptance criteria: see docs/stories.md#us-05 (Product Owner to fill in).
Front end: chat/templates/chat/chat.html, chat/static/chat/chat.js, chat.css.
"""
from django.shortcuts import render


def chat_page(request):
    """Render the chat page. (Skeleton: works today; Ayan owns the layout.)"""
    return render(request, "chat/chat.html")
