from django.urls import path

from dashboard.views import home, knowledge_base, monitoring, usage_logs

app_name = "dashboard"

# TODO(US-12/13/14): Admin pages must require staff login before deployment
# (e.g. django.contrib.admin.views.decorators.staff_member_required).
urlpatterns = [
    path("", home.dashboard_home, name="home"),
    path("knowledge-base/", knowledge_base.knowledge_base_view, name="knowledge_base"),  # US-12 (Zane)
    path("monitoring/", monitoring.monitoring_view, name="monitoring"),  # US-13 (Ayan)
    path("usage-logs/", usage_logs.usage_logs_view, name="usage_logs"),  # US-14 (Ryon)
]
