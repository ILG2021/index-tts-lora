param([string]$Config = "finetune_models/config.yaml")
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe train.py --config $Config --num-workers 0
if ($LASTEXITCODE -ne 0) { throw "Training failed." }
