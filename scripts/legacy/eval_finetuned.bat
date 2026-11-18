@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
REM Evaluate a fine-tuned SmolVLA checkpoint on the Isaac cube-in-box task, starting from the same condition as the
REM scripted demos (ready pose above the table, random cube). Uses port 5556 so it can run next to another server.
REM Usage: eval_finetuned.bat [CHECKPOINT_DIR] [extra sim args, e.g. --episodes 20 --headless]
REM (paths must not contain spaces)
set CKPT=%1
if "%CKPT%"=="" set CKPT=%ROOT%\outputs\smolvla_cube_in_box_isaac\checkpoints\last\pretrained_model
shift
start "SmolVLA eval server" "%~dp0..\run_server.bat" --checkpoint %CKPT% --port 5556
echo Waiting for the SmolVLA server on 127.0.0.1:5556 ...
:wait
powershell -NoProfile -Command "Start-Sleep 2; exit !(Test-NetConnection 127.0.0.1 -Port 5556 -InformationLevel Quiet -WarningAction SilentlyContinue)"
if errorlevel 1 goto wait
call "%ISAAC_SIM_DIR%\python.bat" "%~dp0openarm_bimanual_sim.py" --port 5556 --start-mode expert --episode-seconds 8 %1 %2 %3 %4 %5 %6 %7 %8
