@echo off
REM End-to-end check after the folder move: same checkpoint/conditions as results\baseline_cubebox_30k.json, seeds 1000-1004 only.
REM Output: logs\verify_pathfix_eval.txt (last line: === DONE) and results\verify_pathfix_5ep.json
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"
set "OUT=%ROOT%\logs\verify_pathfix_eval.txt"
> "%OUT%" echo === verify_pathfix_eval %DATE% %TIME%
"%ROOT%\env\python.exe" -u "%~dp0..\eval_harness.py" --checkpoint "outputs\smolvla_yellowarm_cube_in_box\checkpoints\030000\pretrained_model" --episodes 5 --seed 1000 --exec-steps 8 --tag verify_pathfix_5ep --overwrite >> "%OUT%" 2>&1
>> "%OUT%" echo harness exit code %ERRORLEVEL%
>> "%OUT%" echo === DONE
