@echo off
REM Launch the modern WeedVision full-stack Web Application (FastAPI + React UI)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
echo ==============================================================
echo WeedVision Modern Web Dashboard (FastAPI + React UI)
echo Accessible at: http://localhost:8000
echo ==============================================================
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload %*
