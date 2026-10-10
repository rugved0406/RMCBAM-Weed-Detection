#!/usr/bin/env bash
# DEMO MODE: loads the clearly-labelled SMOKE-TEST checkpoints so every page can be explored.
# Their outputs are meaningless (trained a few epochs on partial labels).
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
WV_DEMO=1 WV_MODELS_DIR=demo/models WV_RUNS_DIR=demo/smoke_runs WV_CONF=0.25 python3 -m streamlit run app.py "$@"
