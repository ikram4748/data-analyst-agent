@echo off
rem Windows launcher. Double-click this file.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && (py -3.12 -c "" >nul 2>nul && set "PY=py -3.12")
if not defined PY where py >nul 2>nul && (py -3.11 -c "" >nul 2>nul && set "PY=py -3.11")
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.12 or 3.11 is required: https://www.python.org/downloads/
  pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the virtual environment and installing packages ^(a few minutes^)...
  %PY% -m venv .venv || (pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (pause & exit /b 1)
)

if not exist ".env" copy ".env.example" ".env" >nul
findstr /R /C:"^GEMINI_API_KEY=..*" ".env" >nul || (
  echo Add your Gemini API key to the .env file in this folder, then run this again.
  notepad ".env"
  pause & exit /b 1
)

if not exist "data\bird\mini_dev_sqlite.json" (
  echo First run: downloading the BIRD benchmark databases ^(~800 MB download, ~1.5 GB on disk^)...
  ".venv\Scripts\python.exe" scripts\setup_data.py || (pause & exit /b 1)
)

echo Starting the app at http://localhost:8501  ^(close this window to stop it^)
".venv\Scripts\python.exe" -m streamlit run app.py
