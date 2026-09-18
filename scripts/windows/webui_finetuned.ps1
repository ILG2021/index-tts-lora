param(
    [Parameter(Mandatory=$true)][string]$LoraCheckpoint,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860,
    [string]$RootPath = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
& .\.venv\Scripts\python.exe webui.py `
    --model-dir checkpoints `
    --config checkpoints\config.yaml `
    --lora-checkpoint $LoraCheckpoint `
    --host $HostAddress `
    --port $Port `
    --root-path $RootPath
if ($LASTEXITCODE -ne 0) { throw "WebUI failed." }
