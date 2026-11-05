"""Root URL configuration. Each app owns its own urls.py to avoid merge conflicts."""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("", include("chat.urls")),
    path("alerts/", include("alerts.urls")),
    path("dashboard/", include("dashboard.urls")),
]
