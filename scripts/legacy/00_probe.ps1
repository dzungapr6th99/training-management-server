# Probe the Windows host for GPU / conda / Isaac Sim python.
$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Log  = "$Root\logs\probe.log"
New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null
Start-Transcript -Path $Log -Force | Out-Null

Write-Host "=== HOST ==="
Write-Host ("computer : " + $env:COMPUTERNAME)
Write-Host ("os       : " + (Get-CimInstance Win32_OperatingSystem).Caption)
Write-Host ("ram GB   : " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1))

Write-Host "`n=== GPU (nvidia-smi) ==="
try { & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv } catch { Write-Host "nvidia-smi NOT FOUND: $_" }

Write-Host "`n=== DISK ==="
Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Name -in @('C','D')} |
  ForEach-Object { Write-Host ("{0}: free {1} GB / total {2} GB" -f $_.Name, [math]::Round($_.Free/1GB,1), [math]::Round(($_.Used+$_.Free)/1GB,1)) }

Write-Host "`n=== CONDA ==="
$condaExe = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
if (Test-Path $condaExe) {
  Write-Host "conda exe: $condaExe"
  & $condaExe --version
  & $condaExe env list
} else { Write-Host "conda NOT at $condaExe"; Get-Command conda -ErrorAction SilentlyContinue | Select-Object -Expand Source }

Write-Host "`n=== SYSTEM PYTHONS ==="
foreach ($p in @("python","py")) {
  $c = Get-Command $p -ErrorAction SilentlyContinue
  if ($c) { Write-Host "$p -> $($c.Source)"; & $p -V } else { Write-Host "$p not on PATH" }
}

Write-Host "`n=== ISAAC SIM ==="
$IsaacDir = if ($env:ISAAC_SIM_DIR) { $env:ISAAC_SIM_DIR } else { "D:\Isaac\isaacsim" }
Write-Host ("VERSION: " + (Get-Content "$IsaacDir\VERSION" -ErrorAction SilentlyContinue))
Write-Host ("python.bat exists: " + (Test-Path "$IsaacDir\python.bat"))
Write-Host ("post_install ran (kit dir): " + (Test-Path "$IsaacDir\kit\kit.exe"))
Write-Host "`n--- isaac python: torch / numpy ---"
& "$IsaacDir\python.bat" -c "import sys;print('exe',sys.executable);print('py',sys.version)
try:
    import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())
except Exception as e: print('torch MISSING:',e)
try:
    import numpy;print('numpy',numpy.__version__)
except Exception as e: print('numpy MISSING:',e)
try:
    import transformers;print('transformers',transformers.__version__)
except Exception as e: print('transformers MISSING:',e)"

Write-Host "`n=== HF CACHE ==="
foreach ($d in @("$env:USERPROFILE\.cache\huggingface\hub","$Root\models")) {
  if (Test-Path $d) { Write-Host "$d :"; Get-ChildItem $d -Name | Select-Object -First 20 } else { Write-Host "$d : none" }
}

Write-Host "`n=== PROBE DONE ==="
New-Item -ItemType File -Force -Path "$Root\logs\probe.DONE" | Out-Null
Stop-Transcript | Out-Null
