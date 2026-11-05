"""Latency/failure monitoring page. US-13 (Ayan). Placeholder page for now."""
from django.shortcuts import render


def monitoring_view(request):
    return render(request, "dashboard/monitoring.html")
