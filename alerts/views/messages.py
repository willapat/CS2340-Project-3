"""Alert message generation page. US-10 (Ayan). Placeholder page for now."""
from django.shortcuts import render


def messages_view(request):
    return render(request, "alerts/messages.html")
