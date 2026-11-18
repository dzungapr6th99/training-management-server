@echo off
REM Starts the policy server in its own window, waits for it to load, then launches Isaac Sim.
start "SmolVLA policy server" cmd /k ""%~dp0..\run_server.bat" --state-dim 8 --action-dim 8"
echo Waiting for the SmolVLA server on 127.0.0.1:5555 ...
:wait
powershell -NoProfile -Command "exit !(Test-NetConnection 127.0.0.1 -Port 5555 -InformationLevel Quiet -WarningAction SilentlyContinue)"
if errorlevel 1 (timeout /t 2 /nobreak >nul & goto wait)
call "%~dp0run_sim.bat" %*
