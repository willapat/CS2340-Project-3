# Deployment: PythonAnywhere (US-01, Will)

Live site: **https://willpat.pythonanywhere.com** (deployed 2026-11-06 from `feature/US-01-deployment`)

## Current server status (verified 2026-11-06)
| Check | Result |
|---|---|
| All pages (`/`, `/alerts/*`, `/dashboard/*`) | 200; chat placeholder round-trip works |
| Static files (`/static/css/base.css`) | served |
| `/django-admin/` | login page; superuser created |
| HTTP → HTTPS | redirects (Force HTTPS on) |
| `check --deploy` | only W004 (HSTS off on purpose) |
| SQLite | 3.37.2 (≥ 3.35, so `pysqlite3-binary` not needed) |
| Disk | 304 MB of 512 MB (virtualenv 277 MB) |
| Outbound HTTPS | api.openai.com, api.anthropic.com, generativelanguage.googleapis.com, huggingface.co all reachable |
| **Expiry** | free plan renews 1 month at a time: **click "Run until 1 month from today" in the Web tab before 2026-12-07** (PythonAnywhere emails a week ahead) |

## First-time setup
In a PythonAnywhere **Bash console**:
```bash
git clone https://github.com/willapat/CS2340-Project-3.git
cd CS2340-Project-3
mkvirtualenv --python=python3.11 alert-assist     # later: `workon alert-assist`
pip install --no-cache-dir --no-deps -r deploy/requirements-pythonanywhere.txt
python -c "import sqlite3; print(sqlite3.sqlite_version)"   # need >= 3.35 for ChromaDB
python -c "import chromadb; print('chromadb ok')"            # if this fails on sqlite: pip install pysqlite3-binary
cp .env.example .env
```
Edit `.env`:
```
DJANGO_DEBUG=false
DJANGO_SECRET_KEY=<generate: python -c "import secrets; print(secrets.token_urlsafe(50))">
DJANGO_ALLOWED_HOSTS=<username>.pythonanywhere.com
```
Then:
```bash
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py createsuperuser        # for /django-admin/
```

In the **Web** tab:
1. Add a new web app → **Manual configuration** → **Python 3.11**.
2. **Virtualenv:** `/home/<username>/.virtualenvs/alert-assist`
3. **WSGI configuration file:** replace its contents with:
   ```python
   import os, sys
   path = "/home/<username>/CS2340-Project-3"
   if path not in sys.path:
       sys.path.insert(0, path)
   os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
   from django.core.wsgi import get_wsgi_application
   application = get_wsgi_application()
   ```
4. **Static files:** URL `/static/` → directory `/home/<username>/CS2340-Project-3/staticfiles`
5. **Security:** turn on **Force HTTPS**.
6. Click **Reload**.

## Redeploying
```bash
workon alert-assist && cd ~/CS2340-Project-3 && bash deploy/redeploy.sh
```

## Known limits to watch
- **SQLite ≥ 3.35** for ChromaDB (see the check above; `pysqlite3-binary` is the fallback, and `config/settings.py` picks it up automatically).
- **Disk quota (512 MB free):** the full `requirements.txt` is ~530 MB and doesn't fit. The server uses
  `deploy/requirements-pythonanywhere.txt` instead (~250 MB), which drops chromadb extras we don't use
  (kubernetes, onnxruntime, tokenizers, uvicorn). **So Chroma's built-in local embedder doesn't work on the
  server**; production needs an API embedder passed as `embed=` to `ChromaVectorStore`. Always install
  with `--no-cache-dir`, and check usage with `du -sh ~/.virtualenvs/alert-assist ~/.cache`.
- **Outbound internet:** free accounts can only reach whitelisted domains. That affects the LLM API, URL ingestion (US-02), and the first-run download of Chroma's default embedding model (US-04).
- **Free web apps expire** after a month unless you click "Run until 1 month from today" in the Web tab.
- **HSTS:** HTTPS is confirmed working, so `DJANGO_SECURE_HSTS_SECONDS` can be raised (e.g. `3600`, later
  `31536000`) in the server's `.env`, followed by a reload. That clears the last `check --deploy` warning.
- Error logs: Web tab → **Error log**.
