@echo off
REM One-time setup. Put Himani_mam_images.zip and the two *_SMOKE_best.pt files next to this folder first.
REM For an NVIDIA GPU, install the CUDA build of torch from pytorch.org after this finishes.
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools\prepare_data.py
echo.
echo Setup complete. Double-click run_demo.bat (explore) or run_dashboard.bat (your models).
pause
