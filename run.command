#!/bin/bash
# macOS / Linux launcher. Double-click in Finder, or run ./run.command in a terminal.
set -e
cd "$(dirname "$0")"
PORT="${1:-8501}"   # optional port, e.g. ./run.command 8502

PY=""
for p in python3.12 python3.11; do command -v "$p" >/dev/null 2>&1 && { PY="$p"; break; }; done
if [ -z "$PY" ]; then echo "Python 3.12 or 3.11 is required (https://www.python.org/downloads/)."; read -r -p "Press Enter to close"; exit 1; fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: creating the virtual environment and installing packages (a few minutes)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi

if [ ! -f .env ]; then cp .env.example .env; fi
if ! grep -qE '^GEMINI_API_KEY=.+' .env; then
  echo "Add your Gemini API key to the .env file in this folder, then run this again."
  open -e .env 2>/dev/null || true
  read -r -p "Press Enter to close"; exit 1
fi

if [ ! -f data/bird/mini_dev_sqlite.json ]; then
  echo "First run: downloading the BIRD benchmark databases (~800 MB download, ~1.5 GB on disk)..."
  .venv/bin/python scripts/setup_data.py
fi

echo "Starting the app at http://localhost:$PORT  (close this window to stop it)"
.venv/bin/python -m streamlit run app.py --server.port "$PORT" --server.headless false
