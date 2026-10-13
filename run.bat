@echo off
rem Windows launcher. Double-click this file.
setlocal
cd /d "%~dp0"
call :clean_path

rem Optional port, e.g. "run.bat 8502", so several apps can run at once.
set "PORT=%~1"
if not defined PORT set "PORT=8501"

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

echo Starting the app at http://localhost:%PORT%  ^(close this window to stop it^)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT%
exit /b

:clean_path
rem Drop PATH folders that cannot be reached (for example a BitLocker-locked drive); pip fails on them.
setlocal EnableDelayedExpansion
set "CLEAN="
for %%p in ("%PATH:;=" "%") do if not "%%~p"=="" if exist "%%~p\" set "CLEAN=!CLEAN!;%%~p"
endlocal & set "PATH=%CLEAN:~1%"
exit /b
