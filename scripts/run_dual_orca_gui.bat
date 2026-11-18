@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
REM Dual YellowArm+ORCA: watch the expert pick the cube in the Isaac Sim GUI (runs until you close the window).
REM Extra args pass through, e.g.  run_dual_orca_gui.bat --seed 5 --dart-noise 0.02
cd /d "%~dp0"
call "%ISAAC_SIM_DIR%\python.bat" "%~dp0dual_orca_pick_sim.py" --expert --loop %*
