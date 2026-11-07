"""Tests for US-01: Hosted web app / deployment. Owner: Will."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.test import Client

ROOT = Path(__file__).resolve().parent.parent

PROBE = """
import django
from django.conf import settings
django.setup()
print(settings.DEBUG, settings.ALLOWED_HOSTS, settings.SECURE_SSL_REDIRECT,
      settings.SESSION_COOKIE_SECURE, settings.CSRF_COOKIE_SECURE, sep="|")
"""


def run_settings(**env):
    """Load config.settings in a fresh interpreter. Explicit env vars win over .env values."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("DJANGO_")}
    clean.update(DJANGO_SETTINGS_MODULE="config.settings", **env)
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=ROOT, env=clean, capture_output=True, text=True, timeout=60
    )


def test_production_refuses_to_start_without_secret_key():
    result = run_settings(DJANGO_DEBUG="false", DJANGO_SECRET_KEY="")
    assert result.returncode != 0
    assert "DJANGO_SECRET_KEY must be set" in result.stderr


def test_production_settings_are_secure():
    result = run_settings(
        DJANGO_DEBUG="false", DJANGO_SECRET_KEY="x" * 50, DJANGO_ALLOWED_HOSTS="me.pythonanywhere.com, other.com"
    )
    assert result.returncode == 0, result.stderr
    debug, hosts, ssl_redirect, session_secure, csrf_secure = result.stdout.strip().split("|")
    assert debug == "False"
    assert hosts == "['me.pythonanywhere.com', 'other.com']"
    assert (ssl_redirect, session_secure, csrf_secure) == ("True", "True", "True")


def test_debug_mode_runs_without_secret_key():
    result = run_settings(DJANGO_DEBUG="true", DJANGO_SECRET_KEY="")
    assert result.returncode == 0, result.stderr


def test_deploy_check_has_no_errors():
    """`manage.py check --deploy` may warn (e.g. HSTS off) but must not report errors."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("DJANGO_")}
    env.update(DJANGO_DEBUG="false", DJANGO_SECRET_KEY="k" * 60, DJANGO_ALLOWED_HOSTS="example.com")
    result = subprocess.run(
        [sys.executable, "manage.py", "check", "--deploy", "--fail-level", "ERROR"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wsgi_application_loads():
    from config.wsgi import application

    assert callable(application)


@pytest.mark.django_db
def test_site_serves_pages():
    client = Client()
    assert client.get("/").status_code == 200
    assert client.get("/dashboard/").status_code == 200
