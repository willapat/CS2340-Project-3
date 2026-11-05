"""Usage logs page. US-14 (Ryon). Placeholder page for now."""
from django.shortcuts import render


def usage_logs_view(request):
    return render(request, "dashboard/usage_logs.html")
