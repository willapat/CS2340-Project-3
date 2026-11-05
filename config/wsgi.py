"""WSGI entry point. PythonAnywhere's WSGI file imports `application` from here (US-01)."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_wsgi_application()
