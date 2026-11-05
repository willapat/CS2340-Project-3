from django.urls import path

from alerts.views import conditions, messages

app_name = "alerts"

urlpatterns = [
    path("conditions/", conditions.conditions_view, name="conditions"),  # US-09 (Will)
    path("messages/", messages.messages_view, name="messages"),  # US-10 (Ayan)
]
