#!/usr/bin/env bash
# Launch the modern WeedVision full-stack Web Application (FastAPI + React UI)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [ -f .venv/bin/activate ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
echo "=============================================================="
echo "WeedVision Modern Web Dashboard (FastAPI + React UI)"
echo "Accessible at: http://localhost:8000"
echo "=============================================================="
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload "$@"
