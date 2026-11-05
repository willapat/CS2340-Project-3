"""Shared models for the admin dashboard.

QueryLog is read by US-13 (Ayan, monitoring) and US-14 (Ryon, usage logs).
Ryon (US-14) decides how rows get written. Changing fields
needs a migration and a heads-up to both owners.
"""
from django.db import models


class QueryLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    question = models.TextField()
    prompt_type = models.CharField(max_length=64, blank=True, default="")
    latency_ms = models.PositiveIntegerField(default=0)
    success = models.BooleanField(default=True, db_index=True)
    error_message = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        status = "ok" if self.success else "FAILED"
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {status} {self.prompt_type or '-'}: {self.question[:50]}"
