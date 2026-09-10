#!/usr/bin/env bash
# Launch the deterministic PubMed search web app.
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present (OPENROUTER_API_KEY, NCBI_API_KEY, NCBI_EMAIL, OPENROUTER_MODEL)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Windows venvs (py -m venv / python -m venv under git-bash) lay out
# Scripts/python.exe instead of bin/python -- support both so this script
# works the same on Linux/macOS and on Windows git-bash.
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=""
fi

if [ -z "$PY" ]; then
  python3 -m venv .venv || python -m venv .venv
  [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=.venv/Scripts/python.exe
  "$PY" -m pip install -q -r backend/requirements.txt
fi

if [ ! -f data/mesh.sqlite ]; then
  echo "Building MeSH index (one-time, ~15s)…"
  "$PY" backend/build_index.py
fi

PORT="${PORT:-8077}"
echo "→ http://127.0.0.1:${PORT}"
exec "$PY" -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT}"
