@echo off
REM DEMO MODE: SMOKE-TEST checkpoints so every page can be explored. Outputs are meaningless.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set WV_DEMO=1
set WV_MODELS_DIR=demo\models
set WV_RUNS_DIR=demo\smoke_runs
set WV_CONF=0.25
python -m streamlit run app.py %*
