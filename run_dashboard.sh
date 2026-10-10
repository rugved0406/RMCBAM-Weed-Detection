#!/usr/bin/env bash
# Launch the dashboard with your own models (models/, runs/) and dataset (data/).
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
python3 -m streamlit run app.py "$@"
