"""Dashboard landing page (shared; links to each admin page)."""
from django.shortcuts import render


def dashboard_home(request):
    return render(request, "dashboard/home.html")
