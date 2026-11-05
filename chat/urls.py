from django.urls import path

from chat import api, views

app_name = "chat"

urlpatterns = [
    path("", views.chat_page, name="chat_page"),
    path("api/chat", api.chat_api, name="chat_api"),
]
