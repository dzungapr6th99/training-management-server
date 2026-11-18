@echo off
REM Web monitor of dual_orca_pipeline.py: http://localhost:8765  (dual_orca_pipeline.py starts it by itself).
REM   run_dual_orca_monitor.bat [--port 8765] [--set v1|smoke]
cd /d "%~dp0.."
start "" http://localhost:8765/
env\python.exe -u scripts\dual_orca_monitor.py %*
