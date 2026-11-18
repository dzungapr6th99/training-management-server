@echo off
REM Training environment for Training Studio jobs: Python 3.12 with torch (CUDA), Isaac Sim (pip) and lerobot.
REM Installs only what is missing, so it is safe to rerun.
REM   setup_train_env.bat          install what is missing (several GB the first time, needs internet + NVIDIA driver)
REM   setup_train_env.bat ensure   quick check; installs only if torch / isaacsim / lerobot is missing (studio.bat calls this)
REM   setup_train_env.bat check    only report, install nothing
REM Target Python:
REM   STUDIO_TRAIN_PYTHON set and existing -> missing packages are installed into THAT environment
REM   otherwise                            -> venv STUDIO_TRAIN_ENV (default env_isaacsim in this folder) is created/used
REM Settings (environment variables or .env, see .env.example):
REM   STUDIO_TRAIN_ENV  keep it SHORT on Windows (e.g. C:\mhh_env): Isaac Sim has deep paths
REM   STUDIO_TORCH_INDEX (default https://download.pytorch.org/whl/cu128)  STUDIO_TORCH_VERSION (2.11.0)
REM   STUDIO_TORCHVISION_VERSION (0.26.0)  STUDIO_ISAACSIM_VERSION (6.0.1.0)
REM Never installs into an Isaac Sim install's own python.bat. Does not accept the NVIDIA Omniverse EULA for you:
REM Isaac Sim asks on the first "import isaacsim" (or set OMNI_KIT_ACCEPT_EULA=YES yourself after reading it).
setlocal
for %%I in ("%~dp0.") do set "ROOT=%%~fI"
if exist "%ROOT%\.env" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%\.env") do if not defined %%A if not "%%B"=="" set "%%A=%%B"
if "%STUDIO_TRAIN_ENV%"=="" set "STUDIO_TRAIN_ENV=%ROOT%\env_isaacsim"
if "%STUDIO_TORCH_INDEX%"=="" set "STUDIO_TORCH_INDEX=https://download.pytorch.org/whl/cu128"
if "%STUDIO_TORCH_VERSION%"=="" set "STUDIO_TORCH_VERSION=2.11.0"
if "%STUDIO_TORCHVISION_VERSION%"=="" set "STUDIO_TORCHVISION_VERSION=0.26.0"
if "%STUDIO_ISAACSIM_VERSION%"=="" set "STUDIO_ISAACSIM_VERSION=6.0.1.0"
REM relative paths in .env are taken from this folder (same rule as the backend)
if not "%STUDIO_TRAIN_ENV:~1,1%"==":" if not "%STUDIO_TRAIN_ENV:~0,2%"=="\\" set "STUDIO_TRAIN_ENV=%ROOT%\%STUDIO_TRAIN_ENV%"
if not "%STUDIO_TRAIN_PYTHON%"=="" if not "%STUDIO_TRAIN_PYTHON:~1,1%"==":" if not "%STUDIO_TRAIN_PYTHON:~0,2%"=="\\" set "STUDIO_TRAIN_PYTHON=%ROOT%\%STUDIO_TRAIN_PYTHON%"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=install"

set "PY=%STUDIO_TRAIN_ENV%\Scripts\python.exe"
set "OWN_VENV=1"
if not "%STUDIO_TRAIN_PYTHON%"=="" if exist "%STUDIO_TRAIN_PYTHON%" ( set "PY=%STUDIO_TRAIN_PYTHON%" & set "OWN_VENV=0" )
echo %PY% | find /i "python.bat" >nul && ( echo [train-env] %PY% is an Isaac Sim install's python.bat: packages are never installed there. Unset STUDIO_TRAIN_PYTHON or point it to a venv. & exit /b 1 )

call :probe
if /i "%MODE%"=="check" goto report
if /i "%MODE%"=="ensure" if "%HAVE_ALL%"=="1" exit /b 0
if /i "%MODE%"=="ensure" echo [train-env] the training environment is missing:%MISSING% - installing it now (set STUDIO_AUTO_SETUP_TRAIN_ENV=0 to skip)

echo [train-env] target python: %PY%
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | find "0x1" >nul
if errorlevel 1 echo [train-env] WARNING: Windows long paths are OFF. Isaac Sim may fail to install; enable LongPathsEnabled or use a short STUDIO_TRAIN_ENV.
where nvidia-smi >nul 2>nul || echo [train-env] WARNING: nvidia-smi not found: install the NVIDIA driver before training.

if exist "%PY%" goto have_python
if "%OWN_VENV%"=="0" goto fail
echo [train-env] creating Python 3.12 venv in %STUDIO_TRAIN_ENV%
where py >nul 2>nul && (py -3.12 -m venv "%STUDIO_TRAIN_ENV%") || (python -m venv "%STUDIO_TRAIN_ENV%")
if not exist "%PY%" ( echo [train-env] could not create the venv: install Python 3.12 from python.org and retry & exit /b 1 )
"%PY%" -m pip install --upgrade pip || goto fail
:have_python
"%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"
if errorlevel 1 ( echo [train-env] %PY% is not Python 3.12 ^(Isaac Sim 6 needs 3.12^) & exit /b 1 )
call :probe

set "CONSTRAINTS=%TEMP%\mhh_train_constraints.txt"
> "%CONSTRAINTS%" echo torch==%STUDIO_TORCH_VERSION%
>> "%CONSTRAINTS%" echo torchvision==%STUDIO_TORCHVISION_VERSION%

if "%HAVE_TORCH%"=="1" ( echo [train-env] torch %STUDIO_TORCH_VERSION%: already installed ) else (
  echo [train-env] installing torch %STUDIO_TORCH_VERSION% + torchvision %STUDIO_TORCHVISION_VERSION%
  "%PY%" -m pip install torch==%STUDIO_TORCH_VERSION% torchvision==%STUDIO_TORCHVISION_VERSION% --index-url %STUDIO_TORCH_INDEX% || goto fail
)
if "%HAVE_ISAAC%"=="1" ( echo [train-env] isaacsim: already installed ) else (
  echo [train-env] installing isaacsim %STUDIO_ISAACSIM_VERSION% ^(large download^)
  "%PY%" -m pip install "isaacsim[all,extscache]==%STUDIO_ISAACSIM_VERSION%" --extra-index-url https://pypi.nvidia.com -c "%CONSTRAINTS%" || goto fail
)
if "%HAVE_LEROBOT%"=="1" ( echo [train-env] lerobot: already installed ) else (
  if exist "%ROOT%\vendor\lerobot\pyproject.toml" (
    echo [train-env] installing lerobot[smolvla] from vendor\lerobot ^(has the Windows checkpoints\last junction patch^)
    "%PY%" -m pip install -e "%ROOT%\vendor\lerobot[smolvla]" -c "%CONSTRAINTS%" || goto fail
  ) else (
    echo [train-env] installing lerobot[smolvla] 0.6.2 from PyPI ^(vendor\lerobot not found^)
    "%PY%" -m pip install "lerobot[smolvla]==0.6.2" -c "%CONSTRAINTS%" || goto fail
  )
)
call :probe
if not "%HAVE_ALL%"=="1" goto fail

:report
if not exist "%PY%" ( echo [train-env] no Python at %PY%: run setup_train_env.bat & exit /b 1 )
"%PY%" -c "import sys, importlib.metadata as m; have={(d.metadata['Name'] or '').lower(): d.version for d in m.distributions()}; print('[train-env]', sys.executable, '| python', sys.version.split()[0], *[f'| {n} {have.get(n, 'MISSING')}' for n in ('torch', 'isaacsim', 'lerobot')])"
if not "%HAVE_ALL%"=="1" ( echo [train-env] missing:%MISSING% & exit /b 1 )
echo [train-env] OK. studio.bat uses it automatically; to pin it put in %ROOT%\.env:
echo     STUDIO_TRAIN_PYTHON=%PY%
echo     STUDIO_ISAAC_SIM_PYTHON=%PY%
exit /b 0

:probe
REM sets HAVE_TORCH / HAVE_ISAAC / HAVE_LEROBOT / HAVE_ALL / MISSING for %PY%
set "HAVE_TORCH=0" & set "HAVE_ISAAC=0" & set "HAVE_LEROBOT=0" & set "HAVE_ALL=0" & set "MISSING="
if not exist "%PY%" ( set "MISSING= python torch isaacsim lerobot" & exit /b 0 )
"%PY%" -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('torch').split('+')[0]=='%STUDIO_TORCH_VERSION%' else 1)" >nul 2>nul && set "HAVE_TORCH=1"
"%PY%" -c "import importlib.metadata as m; m.version('isaacsim')" >nul 2>nul && set "HAVE_ISAAC=1"
"%PY%" -c "import importlib.metadata as m; m.version('lerobot')" >nul 2>nul && set "HAVE_LEROBOT=1"
if "%HAVE_TORCH%"=="0" set "MISSING=%MISSING% torch==%STUDIO_TORCH_VERSION%"
if "%HAVE_ISAAC%"=="0" set "MISSING=%MISSING% isaacsim"
if "%HAVE_LEROBOT%"=="0" set "MISSING=%MISSING% lerobot"
if "%HAVE_TORCH%%HAVE_ISAAC%%HAVE_LEROBOT%"=="111" set "HAVE_ALL=1"
exit /b 0

:fail
echo [train-env] FAILED (see the output above). Rerun to resume: installed packages are kept.
exit /b 1
