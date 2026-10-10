#!/usr/bin/env bash
# One-time setup. Put Himani_mam_images.zip and the two *_SMOKE_best.pt files next to this folder first.
cd "$(dirname "$0")"
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python tools/prepare_data.py && \
echo "Setup complete: ./run_demo.sh (explore) or ./run_dashboard.sh (your models)"
