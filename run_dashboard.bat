@echo off
REM Launch the dashboard with your own models (models\, runs\) and dataset (data\).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m streamlit run app.py %*
