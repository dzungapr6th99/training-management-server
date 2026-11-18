@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
REM Generate scripted-expert demonstrations (bimanual OpenArm, left arm, "place the cube in the box") and convert
REM them to a LeRobot v3 dataset.  Usage:  gen_data.bat [N_EPISODES] [extra sim args]
REM Add --headless for speed; leave it out to watch the robot in the Isaac Sim window.
set N=%1
if "%N%"=="" set N=100
set RAW=%ROOT%\datasets\raw\openarm_cube_in_box_isaac
set OUT=%ROOT%\datasets\lerobot\openarm_cube_in_box_isaac
shift
call "%ISAAC_SIM_DIR%\python.bat" "%~dp0openarm_bimanual_sim.py" --expert --episodes %N% --record "%RAW%" %1 %2 %3 %4 %5 %6 %7 %8
"%ROOT%\env\python.exe" "%~dp0..\npz_to_lerobot.py" --raw "%RAW%" --out "%OUT%" --overwrite
