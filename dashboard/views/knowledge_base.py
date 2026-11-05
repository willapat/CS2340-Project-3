"""Knowledge base admin page. US-12 (Zane). Placeholder page for now."""
from django.shortcuts import render


def knowledge_base_view(request):
    #              via dashboard.kb_admin.
    return render(request, "dashboard/knowledge_base.html")
