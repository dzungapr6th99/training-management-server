@echo off
REM Training Studio: set up (first run), create the database, start the web app.   Browse to http://127.0.0.1:8700
REM   First run: creates a Python venv, installs backend\requirements.txt, builds the frontend if frontend\dist is missing
REM   (needs Python 3.10+, internet, and Node 18+ only when dist is missing), then creates backend\data\studio.db.
REM   Options: environment variables or training-management-server\.env (see .env.example):
REM     STUDIO_HOST  default 127.0.0.1      STUDIO_PORT  default 8700      STUDIO_DB  database file
REM     STUDIO_VENV  venv location (default backend\.venv). Windows limits paths to 260 characters and pyarrow has deep
REM                  folders: if pip fails with "No such file or directory ... pyarrow", use a SHORT folder (e.g. C:\studio)
REM                  or set STUDIO_VENV=C:\studio-venv.
REM   Only create/refresh the database:  studio.bat initdb        Recreate it from scratch: studio.bat resetdb
REM   The app has NO login. Keep STUDIO_HOST=127.0.0.1 unless the network is trusted.
setlocal
for %%I in ("%~dp0.") do set "ROOT=%%~fI"
REM Settings from .env (copy .env.example). Variables already set in the environment win; empty values are skipped.
if exist "%ROOT%\.env" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%\.env") do if not defined %%A if not "%%B"=="" set "%%A=%%B"
cd /d "%ROOT%\backend" || exit /b 1
if "%STUDIO_VENV%"=="" set "STUDIO_VENV=%ROOT%\backend\.venv"
if not exist "%STUDIO_VENV%\Scripts\python.exe" (
  echo [studio] first run: creating the Python environment in %STUDIO_VENV%
  where py >nul 2>nul && (py -3 -m venv "%STUDIO_VENV%") || (python -m venv "%STUDIO_VENV%")
  if errorlevel 1 ( echo [studio] could not create the venv: install Python 3.10 or newer from python.org and retry & exit /b 1 )
  "%STUDIO_VENV%\Scripts\python.exe" -m pip install -q -r requirements.txt
  if errorlevel 1 (
    echo [studio] pip install failed: check the internet connection, or the folder path is too long ^(see STUDIO_VENV above^)
    rmdir /s /q "%STUDIO_VENV%" 2>nul
    exit /b 1
  )
)
if not exist "%ROOT%\frontend\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [studio] frontend\dist is missing and Node/npm is not installed: the API will run but there is no web page. Install Node 18+ and rerun.
  ) else (
    echo [studio] building the frontend...
    pushd "%ROOT%\frontend" && call npm install --no-audit --no-fund && call npm run build & popd
  )
)
if /i "%1"=="resetdb" ( "%STUDIO_VENV%\Scripts\python.exe" -m app.init_db --reset & exit /b %errorlevel% )
"%STUDIO_VENV%\Scripts\python.exe" -m app.init_db || exit /b 1
if /i "%1"=="initdb" exit /b 0
REM Training environment (torch + isaacsim + lerobot): installs what is missing; STUDIO_AUTO_SETUP_TRAIN_ENV=0 only checks.
if "%STUDIO_TRAIN_ENV%"=="" set "STUDIO_TRAIN_ENV=%ROOT%\env_isaacsim"
set "TRAIN_MODE=ensure"
if "%STUDIO_AUTO_SETUP_TRAIN_ENV%"=="0" set "TRAIN_MODE=check"
set "TRAIN_OK=0"
call "%ROOT%\setup_train_env.bat" %TRAIN_MODE% && set "TRAIN_OK=1"
if "%TRAIN_OK%"=="0" echo [studio] training environment not ready: train jobs may be unavailable, see README_STUDIO.md
if "%TRAIN_OK%"=="1" if "%STUDIO_TRAIN_PYTHON%"=="" set "STUDIO_TRAIN_PYTHON=%STUDIO_TRAIN_ENV%\Scripts\python.exe"
if "%TRAIN_OK%"=="1" if "%STUDIO_ISAAC_SIM_PYTHON%"=="" set "STUDIO_ISAAC_SIM_PYTHON=%STUDIO_TRAIN_PYTHON%"
if "%STUDIO_HOST%"=="" set "STUDIO_HOST=127.0.0.1"
if "%STUDIO_PORT%"=="" set "STUDIO_PORT=8700"
echo [studio] http://%STUDIO_HOST%:%STUDIO_PORT%   (Ctrl+C to stop)
"%STUDIO_VENV%\Scripts\python.exe" -m uvicorn app.main:app --host %STUDIO_HOST% --port %STUDIO_PORT%
