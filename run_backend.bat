@echo off
setlocal

REM Activate local venv if present
if exist "%~dp0venv\Scripts\activate.bat" (
  call "%~dp0venv\Scripts\activate.bat"
)

python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
