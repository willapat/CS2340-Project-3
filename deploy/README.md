# Deployment: PythonAnywhere (US-01, Will)

Live site: https://willpat.pythonanywhere.com (TODO: confirm once it's live)

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
DJANGO_SECRET_KEY=<generate: python -c "from django.core.management.utils import get_random_secret_key as k; print(k())">
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
- **Free web apps expire** unless you click "Run until 3 months from today" in the Web tab.
- Error logs: Web tab → **Error log**.
