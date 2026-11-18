@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
REM Writes robots\yellowarm_orca_dual\usd\YellowArm_Two_ORCA_sim.usda + dual_orca_pick_scene.usda (open the scene in
REM Isaac Sim: File > Open, then Play). Re-run after run_dual_probe.bat --import.
cd /d "%~dp0"
call "%ISAAC_SIM_DIR%\python.bat" "%~dp0dual_orca_bake.py" %*
