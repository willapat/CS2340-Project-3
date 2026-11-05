from django.contrib import admin

from dashboard.models import QueryLog


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "prompt_type", "latency_ms", "success")
    list_filter = ("success", "prompt_type")
    search_fields = ("question", "error_message")
