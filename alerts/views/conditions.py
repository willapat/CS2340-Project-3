"""Guided conditions form page. US-09 (Will). Placeholder page for now."""
from django.shortcuts import render


def conditions_view(request):
    return render(request, "alerts/conditions.html")
