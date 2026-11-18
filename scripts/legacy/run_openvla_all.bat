@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
REM OpenVLA-7B (%ROOT%\models\openvla-7b, 4-bit) server in its own window, then Isaac Sim with the OpenArm task.
REM Extra args go to the sim, e.g.  run_openvla_all.bat --episodes 3 --task "put the red cube on the green square"
start "OpenVLA policy server" cmd /k ""%ROOT%\env_openvla\python.exe" -u "%~dp0openvla_server.py""
echo Waiting for the OpenVLA server on 127.0.0.1:5555 (loading 7B weights takes ~1-2 min) ...
:wait
powershell -NoProfile -Command "exit !(Test-NetConnection 127.0.0.1 -Port 5555 -InformationLevel Quiet -WarningAction SilentlyContinue)"
if errorlevel 1 (timeout /t 3 /nobreak >nul & goto wait)
call "%~dp0run_sim.bat" %*
