@echo off
if not defined ISAAC_SIM_DIR set "ISAAC_SIM_DIR=D:\Isaac\isaacsim"
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
REM SmolVLA policy server (conda env at %ROOT%\env). Extra args pass through,
REM e.g.  run_server.bat --checkpoint someuser/smolvla-openarm-finetune
set HF_HOME=%ROOT%\models\hf_home
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%ROOT%\env\python.exe" -u "%~dp0policy_server.py" %*
