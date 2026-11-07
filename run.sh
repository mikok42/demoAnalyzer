#!/usr/bin/env bash
# Launch the local web UI. Works on macOS and Linux.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null; then
  PY=python3
else
  echo "python3 not found; install Python 3.10+ first" >&2
  exit 1
fi
if ! "$PY" -c "import fastapi, uvicorn, demoparser2" 2>/dev/null; then
  echo "upgrading pip/setuptools, then installing poswing[web]..."
  "$PY" -m pip install --upgrade pip setuptools wheel
  "$PY" -m pip install -e ".[web]"
fi
exec "$PY" -m poswing.cli serve "$@"
