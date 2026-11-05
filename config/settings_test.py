"""Settings for pytest/CI: debug on, throwaway secret, no .env required."""
import os

os.environ.setdefault("DJANGO_DEBUG", "true")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")

from config.settings import *  # noqa: E402,F403
