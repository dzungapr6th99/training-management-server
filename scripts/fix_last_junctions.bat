@echo off
REM Re-create outputs\<run>\checkpoints\last as a junction to the newest numbered checkpoint of each run.
REM The old junctions still point at D:\Isaac\smolvla-openarm\... (moved away), so they are broken.
REM rmdir on a junction only removes the link, never the checkpoint it points to.
setlocal enabledelayedexpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
for /d %%R in ("%ROOT%\outputs\*") do (
  if exist "%%R\checkpoints\" (
    set "NEWEST="
    for /f "delims=" %%C in ('dir /b /ad /o:n "%%R\checkpoints" ^| findstr /r "^[0-9][0-9]*$"') do set "NEWEST=%%C"
    if defined NEWEST (
      rmdir "%%R\checkpoints\last" 2>nul
      mklink /J "%%R\checkpoints\last" "%%R\checkpoints\!NEWEST!" >nul && echo %%~nxR: last -^> !NEWEST!
    )
  )
)
endlocal
