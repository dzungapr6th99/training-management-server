@echo off
REM Smoke test after the folder move. Writes everything to logs\verify_pathfix.txt (last line: === DONE).
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"
set "OUT=%ROOT%\logs\verify_pathfix.txt"
set HF_HOME=%ROOT%\models\hf_home
set HF_HUB_OFFLINE=1
> "%OUT%" echo === verify_pathfix %DATE% %TIME% ROOT=%ROOT%
>> "%OUT%" echo --- 1 fix_last_junctions
call "%~dp0..\fix_last_junctions.bat" >> "%OUT%" 2>&1
>> "%OUT%" echo --- 2 mhh_paths
"%ROOT%\env\python.exe" "%~dp0..\mhh_paths.py" >> "%OUT%" 2>&1
>> "%OUT%" echo --- 3 imports (lerobot via relative .pth, train module, project modules)
"%ROOT%\env\python.exe" -c "import sys; sys.path.insert(0, r'%ROOT%\scripts'); import lerobot, mhh_paths, wire, humanoid_kin; import lerobot.scripts.lerobot_train as t; print('lerobot from', lerobot.__file__); print('train main', t.main); print('URDF', humanoid_kin.URDF)" >> "%OUT%" 2>&1
>> "%OUT%" echo --- 4 policy_server --selftest
"%ROOT%\env\python.exe" -u "%~dp0..\policy_server.py" --selftest >> "%OUT%" 2>&1
>> "%OUT%" echo selftest exit code %ERRORLEVEL%
>> "%OUT%" echo --- 5 fix_moved_paths dry run
"%ROOT%\env\python.exe" "%~dp0..\fix_moved_paths.py" >> "%OUT%" 2>&1
>> "%OUT%" echo --- 6 train_pipeline/pallet_pipeline import ROOT check
"%ROOT%\env\python.exe" -c "import runpy,sys; sys.path.insert(0, r'%ROOT%\scripts'); [print(f, 'ROOT=', runpy.run_path(r'%ROOT%\scripts\\'+f+'.py', run_name='x').get('ROOT')) for f in ('pallet_dataset_stats','expert_review')]" >> "%OUT%" 2>&1
>> "%OUT%" echo === DONE
