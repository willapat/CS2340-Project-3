#!/usr/bin/env bash
# US-01 (Will): run in a PythonAnywhere Bash console after pushing to main.
#   bash deploy/redeploy.sh
# Assumes the virtualenv is active (`workon alert-assist`) and you're in the repo root.
set -euo pipefail

git pull --ff-only
pip install -q --no-cache-dir --no-deps -r deploy/requirements-pythonanywhere.txt
python manage.py migrate --noinput
python manage.py collectstatic --noinput
python manage.py check --deploy

# Touching the WSGI file reloads the web app on PythonAnywhere.
WSGI_FILE="/var/www/$(whoami)_pythonanywhere_com_wsgi.py"
if [ -f "$WSGI_FILE" ]; then
  touch "$WSGI_FILE"
  echo "Reloaded $WSGI_FILE"
else
  echo "WSGI file not found at $WSGI_FILE; reload from the Web tab instead."
fi
