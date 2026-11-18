@echo off
setlocal
set "SCRIPTS=%~dp0"
if "%ISAAC_SIM_DIR%"=="" set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
cd /d "%SCRIPTS%"
call "%ISAAC_SIM_DIR%\python.bat" dual_robot_probe.py %*
