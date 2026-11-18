@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
REM OpenArm + SmolVLA in Isaac Sim (GUI). Extra args pass through, e.g.
REM   run_sim.bat --task "pick up the red cube" --episodes 3 --episode-seconds 20
REM   run_sim.bat --headless --save-frames %ROOT%\logs\frames
call "%ISAAC_SIM_DIR%\python.bat" "%~dp0openarm_smolvla_sim.py" %*
