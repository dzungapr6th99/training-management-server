@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
REM Bimanual OpenArm "place the cube in the box" with the fine-tuned SmolVLA
REM (qualia-robotics/smolvla-openarm-cube-in-box-v2, in models\smolvla_openarm_cube_in_box_v2).
REM Server in its own window, then Isaac Sim (GUI, runs until you close it). Extra args go to the sim, e.g.
REM   run_bimanual_all.bat --episodes 20 --start-index 81 --grasp-depth 0.03
start "SmolVLA cube-in-box server" cmd /k ""%~dp0..\run_server.bat" --checkpoint %ROOT%\models\smolvla_openarm_cube_in_box_v2"
echo Waiting for the SmolVLA server on 127.0.0.1:5555 ...
:wait
powershell -NoProfile -Command "exit !(Test-NetConnection 127.0.0.1 -Port 5555 -InformationLevel Quiet -WarningAction SilentlyContinue)"
if errorlevel 1 (timeout /t 2 /nobreak >nul & goto wait)
call "%ISAAC_SIM_DIR%\python.bat" "%~dp0openarm_bimanual_sim.py" %*
