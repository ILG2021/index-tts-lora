param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe webui.py `
    --model_dir finetune_models `
    --config finetune_models\config_finetuned.yaml `
    --host $HostAddress `
    --port $Port
if ($LASTEXITCODE -ne 0) { throw "WebUI failed." }
