#!/usr/bin/env bash
# Launch the deterministic PubMed search web app.
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present (OPENROUTER_API_KEY, NCBI_API_KEY, NCBI_EMAIL, OPENROUTER_MODEL)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q -r backend/requirements.txt
fi

if [ ! -f data/mesh.sqlite ]; then
  echo "Building MeSH index (one-time, ~15s)…"
  ./.venv/bin/python backend/build_index.py
fi

PORT="${PORT:-8077}"
echo "→ http://127.0.0.1:${PORT}"
exec ./.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT}"
